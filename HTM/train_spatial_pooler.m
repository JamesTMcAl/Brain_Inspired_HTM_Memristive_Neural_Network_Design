function [best_weights, sample_counter, val_accuracy_history, base_area_density, syn_inc_base, syn_dec_base, inhibition_radius, syn_inc_scalar, syn_dec_scalar, ...
         epoch_energy, epoch_write_cycles, epoch_sparsity, epoch_entropy, fallback_counter_epoch] = ...
         train_spatial_pooler(train_data, train_label, base_area_density, syn_inc_base, syn_dec_base, ...
         sample_counter, syn_connected_thresh, use_gpu, w_permanence, decay_scaling, endurance_rate, ...
         pca_coeff, potential_radius, overlap_dimension, val_data, val_label, val_accuracy_history, entropy_threshold, sparsity_threshold, dataset)
%TRAIN_SPATIAL_POOLER Processes a mini-batch of data to update the spatial pooler.

    fprintf('[DEBUG] SP call with: density=%.4f, LTP=%.4f, LTD=%.4f, decay=%.2e, endurance=%.2e\n', ...
            base_area_density, syn_inc_base, syn_dec_base, decay_scaling, endurance_rate);
    fprintf('[DEBUG] start train : train_data size: %s | train_labels size: %s\n', ...
            mat2str(size(train_data)), mat2str(size(train_label)));

    if ~use_gpu || isempty(use_gpu)
        if safe_gpuDeviceCount > 0
            g = gpuDevice();
            use_gpu = false;
        else
            use_gpu = false;
        end
    end

    cfg          = sp_config.instance();
    boost_factor = cfg.BOOST_FACTOR;
    epoch_energy       = 0;
    epoch_write_cycles = 0;
    epoch_sparsity     = NaN;
    epoch_entropy      = NaN;

    val_accuracy_history = val_accuracy_history(:);

    % Persistent variables
    persistent total_energy memristor_stats nominal_syn_inc_base nominal_syn_dec_base best_val_accuracy energy_history density_state;
    if isfield(cfg, 'EPOCH_START_FLAG') && cfg.EPOCH_START_FLAG
        total_energy      = 0;
        memristor_stats   = struct();
        energy_history    = [];
        best_val_accuracy = -Inf;
        cfg.EPOCH_START_FLAG = false;
        % preserve entropy history across epochs
    end
    if isempty(density_state)
        density_state = struct('I', 0, 'prevOutput', 0, 'hist', [], 'prev_error', 0, 'entropy_hist', [], 'target_activity', []);
    end
    if isempty(total_energy),        total_energy        = 0;     end
    if isempty(memristor_stats),     memristor_stats     = struct(); end
    if isempty(nominal_syn_inc_base), nominal_syn_inc_base = syn_inc_base; end
    if isempty(nominal_syn_dec_base), nominal_syn_dec_base = syn_dec_base; end
    if isempty(best_val_accuracy),   best_val_accuracy   = -Inf;  end
    if isempty(energy_history),      energy_history      = [];    end

    kwta_state   = struct();
    syn_state    = struct();
    best_weights = w_permanence;

    num_samples       = size(train_data, 3);
    train_sdrs        = false(num_samples, prod(overlap_dimension));
    active_count_total = 0;

    if use_gpu
        column_activity_sum = gpuArray.zeros(overlap_dimension, 'double');
    else
        column_activity_sum = zeros(overlap_dimension, 'double');
    end

    warmup_samples = min(200, ceil(0.1 * num_samples));
    window_size    = 50;
    entropy_hist   = [];
    sparsity_hist  = [];
    consecutive_viol = 0;
    last_recovery    = -100;
    target_energy    = 1e-10;

    if isa(w_permanence, 'gpuArray')
        noiseLTP = rand(size(w_permanence), 'like', w_permanence).^2 .* betarnd(2, 5);
        noiseLTD = rand(size(w_permanence), 'like', w_permanence).^2 .* betarnd(5, 2);
    else
        noiseLTP = reshape(betarnd(2, 5, numel(w_permanence), 1), size(w_permanence));
        noiseLTD = reshape(betarnd(5, 2, numel(w_permanence), 1), size(w_permanence));
    end
    noiseLTP = cast(noiseLTP, 'like', w_permanence);
    noiseLTD = cast(noiseLTD, 'like', w_permanence);

    if ~exist('threshold_tracker', 'var') || isempty(threshold_tracker)
        threshold_tracker = 1e-3;
    end

    sample_batch = cfg.OVERLAP_BATCH_SIZE;

    % Main training loop
    for start = 1:sample_batch:num_samples
        idxVec = start : min(start + sample_batch - 1, num_samples);

        if isempty(idxVec), break; end

        chunk_start_counter = sample_counter + 1;

        [overlap_batch, syn_connected_thresh_batch, threshold_tracker] = ...
            compute_overlap(train_data, w_permanence, overlap_dimension, ...
                            potential_radius, idxVec, syn_connected_thresh, ...
                            use_gpu, chunk_start_counter, cfg.OVERLAP_BATCH_SIZE, threshold_tracker);

        if ndims(overlap_batch) == 2
            overlap_batch = reshape(overlap_batch, size(overlap_batch,1), size(overlap_batch,2), 1);
        end

        for kk = 1:numel(idxVec)
            ii = idxVec(kk);
            sample_counter = sample_counter + 1;

            overlap = squeeze(overlap_batch(:, :, kk));
            thr     = syn_connected_thresh_batch(kk) + 0.1 * std(overlap(:));

            assert(isequal(size(overlap), overlap_dimension), ...
                   'Overlap slice %d size %s ~= expected %s', ...
                   kk, mat2str(size(overlap)), mat2str(overlap_dimension));

            train_sdrs(ii, :) = reshape(overlap > thr, 1, []);

            [active_columns, avg_act, kwta_state, base_area_density] = ...
                apply_kwta(overlap, base_area_density, sample_counter, use_gpu, (ii==1), kwta_state);

            % Structural rewiring
            if isfield(kwta_state, 'rewire_flag') && any(kwta_state.rewire_flag(:))
                [rr, cc] = find(kwta_state.rewire_flag);
                for k = 1:numel(rr)
                    r = rr(k); c = cc(k);
                    w_permanence(:, :, r, c) = 0.45 + 0.1 * rand(potential_radius, potential_radius);
                end
                kwta_state.rewire_flag = false(size(kwta_state.rewire_flag));
            end

            % Reject check
            if isfield(kwta_state, 'reject_flag') && kwta_state.reject_flag
                fprintf('[REJECT] Parameters rejected at sample_counter=%d.\n', sample_counter);
                best_weights = [];
                return;
            end

            if mod(sample_counter, cfg.DEBUG_INTERVAL) == 0
                fprintf('[DEBUG] Iter %d: Mean Overlap=%.4f\n', sample_counter, mean(overlap(:)));
            end

            active_count_total = active_count_total + nnz(active_columns);

            if use_gpu
                column_activity_sum = column_activity_sum + cast(active_columns, 'double');
            else
                column_activity_sum = column_activity_sum + double(active_columns);
            end

            % Weight update prep
            active_cols_4d  = reshape(active_columns, [1, 1, size(active_columns)]);
            expanded_active = repmat(active_cols_4d, [potential_radius, potential_radius, 1, 1]);
            assert(isequal(size(expanded_active), size(w_permanence)), ...
                   'Mismatch: expanded active columns do not match w_permanence size.');

            if ndims(train_data) == 3
                current = train_data(:, :, ii);
            else
                current = mean(train_data(:, :, :, ii), 3);
            end
            input_std = std(single(current(:)));

            if isempty(energy_history)
                avg_energy = target_energy;
            else
                avg_energy = mean(energy_history);
            end
            scaling_factor  = target_energy / (avg_energy + eps);
            scaling_factor  = max(0.5, min(2.0, scaling_factor));
            effective_syn_inc = nominal_syn_inc_base * scaling_factor;
            effective_syn_dec = nominal_syn_dec_base * scaling_factor;

            [syn_inc, syn_dec, syn_state] = adjust_synaptic_factors( ...
                expanded_active, effective_syn_inc, effective_syn_dec, ...
                boost_factor, sample_counter, decay_scaling, input_std, syn_state, w_permanence);

            inhibition_radius = kwta_state.inhibition_radius;
            syn_inc_scalar    = mean(syn_inc(:));
            syn_dec_scalar    = mean(syn_dec(:));

            [w_permanence, ~, energy, memristor_stats] = update_permanence( ...
                w_permanence, expanded_active, syn_inc, syn_dec, ...
                memristor_stats, sample_counter, endurance_rate, noiseLTP, noiseLTD);

            total_energy   = total_energy + energy;
            energy_history = [energy_history, energy];
            if length(energy_history) > window_size
                energy_history = energy_history(end-window_size+1:end);
            end

            syn_inc_base = effective_syn_inc;
            syn_dec_base = effective_syn_dec;

            if mod(sample_counter, cfg.DEBUG_INTERVAL) == 0
                fprintf('[ENERGY] Iter %d: Avg Energy=%.2e, Scaling=%.2f\n', ...
                        sample_counter, avg_energy, scaling_factor);
            end

            %  Entropy computed HERE
            current_entropy = calculate_entropy(active_columns);

            % Update density_state entropy history for adaptive target
            density_state.entropy_hist = [density_state.entropy_hist, current_entropy];
            if numel(density_state.entropy_hist) > 50
                density_state.entropy_hist = density_state.entropy_hist(end-49:end);
            end

            % PI sparsity controller
            if mod(sample_counter, cfg.KWTA_PI_PERIOD) == 0
                [base_area_density, density_state] = adjust_density( ...
                    base_area_density, overlap, sample_counter, (ii==1), density_state);
            end

            %  Entropy history (for reset logic) 
            if isempty(entropy_hist), entropy_hist = []; end
            entropy_hist = [entropy_hist, current_entropy];
            if numel(entropy_hist) > 100
                entropy_hist = entropy_hist(end-99:end);
            end

            % Entropy-proportional weight reset
            entropy_floor  = 0.15;
            decline_window = 50;
            if numel(entropy_hist) >= decline_window
                recent_entropy = entropy_hist(end-decline_window+1:end);
                if mean(diff(recent_entropy)) < 0 && mean(recent_entropy) < entropy_floor
                    reset_strength = min(0.5, 0.1 + (entropy_floor - mean(recent_entropy)) * 5);
                    fprintf('[LOG] Resetting weights (strength=%.2f) at iter %d\n', ...
                            reset_strength, sample_counter);
                    if use_gpu
                        noise = gpuArray.rand(size(w_permanence));
                    else
                        noise = rand(size(w_permanence));
                    end
                    w_permanence = (1 - reset_strength) * w_permanence + reset_strength * noise;
                    if isfield(syn_state, 'velocity_inc')
                        syn_state.velocity_inc = 0;
                        syn_state.velocity_dec = 0;
                    end
                end
            end

            fallback_counter_epoch = 0;
            if isfield(kwta_state, 'fallback_counter')
                fallback_counter_epoch = kwta_state.fallback_counter;
            end

            % Sparsity tracking
            sparsity_sample = mean(active_columns(:)) * 100;
            sparsity_hist   = [sparsity_hist, sparsity_sample];

            % Dynamic thresholds after warmup
            if sample_counter == warmup_samples
                entropy_threshold  = max(mean(entropy_hist), 0.15);
                sparsity_threshold = mean(sparsity_hist) * 2.0;
                consecutive_viol   = 0;
                fprintf('[LOG] Set dynamic thresholds: Entropy=%.4f, Sparsity=%.2f%%\n', ...
                        entropy_threshold, sparsity_threshold);
            end

            % Recovery 
            if sample_counter > warmup_samples && ((current_entropy < entropy_threshold) || (sparsity_sample < sparsity_threshold * 0.3))
                consecutive_viol = consecutive_viol + 1;
                if consecutive_viol >= 25
                    consecutive_viol = 0;
                    if sample_counter - last_recovery >= 50
                        last_recovery = sample_counter;
                        fprintf('[RECOVERY] at iter %d - triggering recovery\n', sample_counter);
                        base_area_density = min(base_area_density * 1.15, cfg.MAX_DENSITY);
                        entropy_threshold = entropy_threshold * 0.85;
                        if isfield(syn_state, 'velocity_inc')
                            syn_state.velocity_inc = 0;
                            syn_state.velocity_dec = 0;
                        end
                        if use_gpu
                            noise = gpuArray.rand(size(w_permanence));
                        else
                            noise = rand(size(w_permanence));
                        end
                        w_permanence = 0.92 * w_permanence + 0.08 * noise;
                    end
                end
            else
                consecutive_viol = 0;
            end

        end % inner for kk

        syn_connected_thresh = syn_connected_thresh_batch(end);

    end % outer for start

    % Gentle threshold drift back up after recovery
    if sample_counter > warmup_samples
        entropy_threshold = entropy_threshold * 1.001;
        entropy_threshold = min(entropy_threshold, max(mean(entropy_hist), 0.15));
    end

    % Epoch-end metrics
    processed_count = sample_counter;
    epoch_energy    = sum(energy_history);

    if isfield(memristor_stats, 'write_cycles')
        epoch_write_cycles = mean(memristor_stats.write_cycles(:));
    else
        epoch_write_cycles = 0;
    end

    column_activity_sum = safe_gather(column_activity_sum);
    epoch_sparsity = 100 * active_count_total / (processed_count * numel(column_activity_sum));

    p_active  = column_activity_sum / processed_count;
    p_inactive = 1 - p_active;
    column_entropy = -(p_active .* log2(p_active + eps) + p_inactive .* log2(p_inactive + eps));
    epoch_entropy  = mean(column_entropy(:));

    % Validation accuracy + plateau check
    if ~isempty(val_data)
        preds     = infer_labels(val_data, w_permanence, train_data, train_label, overlap_dimension);
        preds     = preds(:);
        val_label = val_label(:);
        correct   = (preds == val_label);
        val_acc   = sum(correct) / numel(correct) * 100;

        if cfg.DEBUG
            fprintf('DEBUG BEFORE APPEND: val_acc size = [%d %d]\n', size(val_acc));
            fprintf('DEBUG BEFORE APPEND: history size = [%d %d]\n', size(val_accuracy_history));
        end

        val_accuracy_history(end+1) = val_acc;

        if cfg.DEBUG
            fprintf('DEBUG AFTER APPEND: history size = [%d %d]\n', size(val_accuracy_history));
        end

        window = 10;
        if numel(val_accuracy_history) >= window
            recent = val_accuracy_history(end-window+1:end);
            flat   = max(recent) - min(recent) < 0.1;
            slope  = recent(end) - recent(1) < 0;
            epoch_idx = numel(val_accuracy_history);
            if flat && slope
                fprintf('[EARLY STOP] Validation plateau at epoch %d (%.2f%%)\n', epoch_idx, val_acc);
                return;
            end
        end

        if val_acc > best_val_accuracy
            best_val_accuracy = val_acc;
            best_weights      = w_permanence;
        end
    end

end