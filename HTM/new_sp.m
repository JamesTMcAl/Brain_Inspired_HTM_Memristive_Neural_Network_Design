function new_sp(dataset, train_data, train_label)
%NEW_SP  HTM Spatial Pooler with self-evolving parameter adaptation.
%   Self-evolution mechanism:
%     - Tracks a composite performance metric each epoch
%       (anomaly × entropy × sparsity_quality)
%     - If performance degrades over 3 consecutive epochs, triggers
%       re-optimisation of SP hyperparameters against the composite fitness
%     - TM anomaly feeds back to SP decay_scaling each epoch
%     - Cumulative sample counter preserves momentum ramp across epochs

    clc;
    clear compute_overlap adjust_synaptic_factors apply_kwta pi_controller

    cfg = sp_config.instance();
    entropy_threshold  = cfg.ENTROPY_THRESHOLD_INIT;
    sparsity_threshold = cfg.SPARSITY_THRESHOLD_INIT;
    if strcmp(dataset, 'CIFAR')
        sparsity_threshold = 20;
    end

    % Console logging
    logFile = 'new_sp_console_output.txt';
    if exist(logFile, 'file'), delete(logFile); end
    diary(logFile);
    fprintf('Logging started at %s\n', datestr(now));

    try

        % Data preparation

        train_epoch = 10;
        val_ratio   = 0.2;
        [train_data, train_label, val_data, val_label] = split_data(train_data, train_label, val_ratio);

        if strcmp(dataset, 'CIFAR')
            % CIFAR already in [0,1]
        else
            val_data = val_data / 255.0;
        end

        if ndims(val_data) == 4
            assert(size(val_data, 4) == numel(val_label), 'Val data/label count mismatch.');
        else
            assert(size(val_data, 3) == numel(val_label), 'Val data/label count mismatch.');
        end


        % SP geometry

        [h, w, ~]        = size(train_data);
        potential_radius  = floor(min(h, w) / 4);
        input_dimension   = [size(train_data, 1), size(train_data, 2)];


        % PCA initialisation

        numImgs   = size(train_data, 3);
        nPCA      = max(1, round(cfg.PCA_SAMPLE_FRAC * numImgs));
        selIdx    = randperm(numImgs, nPCA);
        pr        = potential_radius;
        maxPerImg = ceil(h/pr) * ceil(w/pr);
        patches   = zeros(pr^2, nPCA * maxPerImg, 'double');
        col       = 1;
        for k = 1:nPCA
            img = train_data(:, :, selIdx(k));
            img_patches = im2col(img, [pr pr], 'distinct');
            np  = size(img_patches, 2);
            patches(:, col:col+np-1) = img_patches;
            col = col + np;
        end
        patches(:, col:end) = [];
        [coeff, ~] = pca(patches', 'Centered', false);
        pca_coeff  = reshape(coeff(:, 1), [pr pr]);


        % Hyperparameter optimisation

        [best_params, adjusted_potential_radius] = optimize_hyperparameters( ...
            train_data, train_label, val_data, val_label, dataset, potential_radius, pca_coeff);

        base_area_density      = min(0.5, best_params.base_area_density * 1.15);
        syn_inc_base           = best_params.syn_inc_base;
        syn_dec_base           = best_params.syn_dec_base;
        syn_connected_thresh_base = 0.3;
        noise_std_base         = 0.02;
        decay_scaling          = best_params.decay_scaling;
        endurance_rate         = best_params.endurance_rate;

        cfg.OVERLAP_MIN_ACTIVE_FRAC = 0.8 * best_params.base_area_density;
        cfg.OVERLAP_MAX_ACTIVE_FRAC = 1.2 * best_params.base_area_density;

        potential_radius  = adjusted_potential_radius;
        overlap_dimension = input_dimension - potential_radius + 1;
        assert(all(overlap_dimension > 0), 'Invalid overlap dimensions.');


        % GPU setup

        use_gpu = safe_gpuDeviceCount() > 0;
        if use_gpu
            g = gpuDevice();
            fprintf('GPU detected: %s\n', g.Name);
            train_data = gpuArray(train_data);
        end


        % Initialise weights and history arrays

        w_permanence = initialize_permanence(pca_coeff, potential_radius, overlap_dimension, use_gpu);

        energy_history            = zeros(train_epoch, 1);
        write_cycle_history       = zeros(train_epoch, 1);
        sparsity_per_epoch        = zeros(train_epoch, 1);
        entropy_per_epoch         = zeros(train_epoch, 1);
        val_accuracy_history      = zeros(train_epoch, 1);
        inhibition_radius_history = zeros(train_epoch, 1);
        syn_inc_history           = zeros(train_epoch, 1);
        syn_dec_history           = zeros(train_epoch, 1);
        base_area_density_history = zeros(train_epoch, 1);
        fallbacks_per_epoch       = zeros(train_epoch, 1);
        anomaly_per_epoch         = zeros(train_epoch, 1);

        % Self-evolution performance tracking
        % perf_metric = entropy × (1 - |sparsity - target|) × (1 - anomaly)
        % Higher is better; degradation over 3 epochs triggers re-optimisation
        perf_history    = zeros(train_epoch, 1);
        reopt_cooldown  = 0;   % epochs remaining before re-optimisation allowed again

        fprintf('Starting Training...\n');
        tic;
        fprintf('[DEBUG] Before training: w_permanence Size=%s, Mean=%.4f\n', ...
                mat2str(size(w_permanence)), mean(w_permanence(:)));

        tm_state_main      = struct();
        cumulative_counter = 0;


        % Epoch loop

        for tep = 1:train_epoch
            fprintf('\nEpoch %d/%d\n', tep, train_epoch);

            sample_counter      = cumulative_counter;
            cfg.EPOCH_START_FLAG = true;

            % Noise injection (anneals to zero over epochs)
            noise_std        = noise_std_base * (1 - tep / train_epoch);
            train_data_noisy = max(0, min(1, train_data + noise_std * randn(size(train_data))));
            if use_gpu
                train_data_noisy = gpuArray(train_data_noisy);
            end

            %Spatial Pooler training -
            [w_permanence, sample_counter, returned_history, base_area_density, ...
             syn_inc_base, syn_dec_base, inh_rad, inc_rate, dec_rate, ...
             e, wc, epoch_sparsity, epoch_entropy, fallback_counter_epoch] = ...
                train_spatial_pooler( ...
                    train_data_noisy, train_label, ...
                    base_area_density, syn_inc_base, syn_dec_base, ...
                    sample_counter, syn_connected_thresh_base, use_gpu, ...
                    w_permanence, decay_scaling, endurance_rate, ...
                    pca_coeff, potential_radius, overlap_dimension, ...
                    val_data, val_label, val_accuracy_history, ...
                    entropy_threshold, sparsity_threshold);

            % Record epoch metrics
            final_val_acc             = returned_history(end);
            val_accuracy_history(tep) = final_val_acc;
            base_area_density_history(tep) = base_area_density;
            inhibition_radius_history(tep) = inh_rad;
            syn_inc_history(tep)           = inc_rate;
            syn_dec_history(tep)           = dec_rate;
            energy_history(tep)            = e;
            write_cycle_history(tep)       = wc;
            fallbacks_per_epoch(tep)       = fallback_counter_epoch;
            sparsity_per_epoch(tep)        = epoch_sparsity;
            entropy_per_epoch(tep)         = epoch_entropy;

            [~, stop_epoch] = max(val_accuracy_history);

            fprintf('[LOG] Epoch %d/%d: Sparsity=%.2f%% | Entropy=%.4f | ValAcc=%.1f%%\n', ...
                    tep, train_epoch, epoch_sparsity, epoch_entropy, final_val_acc);

            %TM pass on real SP SDRs -
            anomaly_sum = 0;
            tt_tm       = struct();
            kwta_tm     = struct();
            n_train_samples = size(train_data_noisy, 3);

            for tm_i = 1:n_train_samples
                [ov, ~, tt_tm] = compute_overlap(train_data_noisy, w_permanence, ...
                    overlap_dimension, potential_radius, tm_i, ...
                    syn_connected_thresh_base, false, tm_i, 1, tt_tm);
                [active_cols, ~, kwta_tm] = apply_kwta(ov, base_area_density, tm_i, ...
                    false, (tm_i == 1), kwta_tm);
                [~, ~, ~, tm_state_main] = temporal_memory(active_cols, tm_state_main, ...
                    true, sample_counter + tm_i);
                anomaly_sum = anomaly_sum + tm_state_main.anomaly_score;
            end

            avg_anomaly            = anomaly_sum / n_train_samples;
            anomaly_per_epoch(tep) = avg_anomaly;

            %TM anomaly feedback to SP decay -
            if numel(tm_state_main.anomaly_history) >= 10
                recent_anomaly = mean(tm_state_main.anomaly_history(end-9:end));
                if recent_anomaly > 0.5
                    decay_scaling = decay_scaling * 1.05;
                    fprintf('[TM-FB] Epoch %d: High anomaly %.3f - slowing SP decay\n', tep, recent_anomaly);
                elseif recent_anomaly < 0.2
                    decay_scaling = max(decay_scaling * 0.98, 1e-4);
                end
            end
            fprintf('[TM] Epoch %d avg anomaly: %.3f\n', tep, avg_anomaly);

            %Self-evolution: performance metric -
            % Measures how close sparsity is to the density target
            sparsity_quality = 1 - abs(epoch_sparsity/100 - base_area_density);
            sparsity_quality = max(0, sparsity_quality);

            % Combined metric: want high entropy, good sparsity, low anomaly
            perf_history(tep) = epoch_entropy * sparsity_quality * (1 - avg_anomaly);
            fprintf('[SELF-EVOLVE] Epoch %d perf_metric=%.4f\n', tep, perf_history(tep));

            %Self-evolution: re-optimise if degrading -
            if tep >= 3 && reopt_cooldown == 0
                recent_perf = perf_history(tep-2:tep);
                % Check for consistent decline over last 3 epochs
                if recent_perf(3) < recent_perf(2) && recent_perf(2) < recent_perf(1) ...
                        && (recent_perf(1) - recent_perf(3)) > 0.02
                    fprintf('[SELF-EVOLVE] Performance degrading (%.4f → %.4f → %.4f). Re-optimising...\n', ...
                            recent_perf(1), recent_perf(2), recent_perf(3));

                    % Re-run optimisation on current noisy data
                    try
                        % gather() is matlab in Octave data is already on CPU
                        train_data_for_opt = train_data_noisy;
                        if isa(train_data_noisy, 'gpuArray')
                            train_data_for_opt = gather(train_data_noisy);
                        end
                        [new_params, ~] = optimize_hyperparameters( ...
                            train_data_for_opt, train_label, ...
                            val_data, val_label, dataset, potential_radius, pca_coeff);

                        % Only accept if composite fitness improves
                        new_loss = evaluate_hyperparams( ...
                            train_data_for_opt, train_label, ...
                            val_data, val_label, new_params, pca_coeff, potential_radius);

                        current_loss = -final_val_acc/100 ...
                            + abs(epoch_sparsity/100 - base_area_density)*10 ...
                            + max(0, 0.2 - epoch_entropy)*5;

                        if new_loss < current_loss
                            decay_scaling = new_params.decay_scaling;
                            syn_inc_base  = new_params.syn_inc_base;
                            syn_dec_base  = new_params.syn_dec_base;
                            fprintf('[SELF-EVOLVE] Accepted new params (loss %.4f → %.4f)\n', ...
                                    current_loss, new_loss);
                        else
                            fprintf('[SELF-EVOLVE] New params not better (%.4f vs %.4f), keeping current\n', ...
                                    new_loss, current_loss);
                        end
                    catch ME
                        fprintf('[SELF-EVOLVE] Re-optimisation failed: %s\n', ME.message);
                    end

                    reopt_cooldown = 3;  % don't re-optimise again for 3 epochs
                end
            end

            if reopt_cooldown > 0
                reopt_cooldown = reopt_cooldown - 1;
            end

            cumulative_counter = cumulative_counter + sample_counter;
        end


        % Post-training

        if isa(w_permanence, 'gpuArray')
            w_permanence = gather(w_permanence);
        end

        config = struct( ...
            'overlap_dim',          overlap_dimension, ...
            'potential_radius',     potential_radius, ...
            'syn_connected_thresh', syn_connected_thresh_base, ...
            'base_area_density',    base_area_density);

        sample_idx    = 1;
        example_state = struct('sample_counter', 1, 'threshold_tracker', struct());
        [sdr_flat, ~] = generate_sdr(train_data, w_permanence, sample_idx, config, example_state);
        sample_sdr    = reshape(sdr_flat, overlap_dimension);
        sample_input  = gather(train_data(:, :, sample_idx));

        total_training_time = toc;
        fprintf('\nTraining Complete. Total Time: %.2f seconds\n', total_training_time);

        % Save results including self-evolution history
        threshold_tracker = struct();
		timestamp = datestr(now, 'yyyymmdd_HHMMSS');
		save(['training_results_' timestamp '.mat'], ...
             'w_permanence', 'pca_coeff', ...
             'sample_idx', 'sample_input', 'sample_sdr', ...
             'sparsity_per_epoch', 'entropy_per_epoch', ...
             'inhibition_radius_history', ...
             'syn_inc_history', 'syn_dec_history', ...
             'val_accuracy_history', 'energy_history', ...
             'threshold_tracker', 'write_cycle_history', 'stop_epoch', ...
             'base_area_density_history', 'fallbacks_per_epoch', ...
             'anomaly_per_epoch', 'perf_history', ...
             '-v7.3');

    catch ME
        fprintf('Error: %s\n', ME.message);
        fprintf('Stack:\n%s\n', ME.message);
    end

	fprintf('\nLogging ended at %s\n', datestr(now));    diary off;
end
