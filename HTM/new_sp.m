function new_sp(dataset, train_data,train_label)
    % HTM Spatial Pooler - Adaptive kWTA with Dynamic Sparsity and Entropy
    clc;
    clear compute_overlap adjust_synaptic_factors apply_kwta pi_controller 
    cfg = sp_config.instance();
    entropy_threshold  = cfg.ENTROPY_THRESHOLD_INIT;
    sparsity_threshold = cfg.SPARSITY_THRESHOLD_INIT;
        if strcmp(dataset, 'CIFAR')
    sparsity_threshold = 20; 
        end

    % Start logging console output
    logFile = 'new_sp_console_output.txt';   
    if exist(logFile, 'file')
    delete(logFile);
    end
    diary(logFile);


    fprintf('Logging started at %s\n', datetime('now'));

    try
        

        
        threshold_tracker = struct(); 
        train_epoch = 10;
        energy_history       = zeros(train_epoch,1);
        write_cycle_history  = zeros(train_epoch,1);

        % Split into training/validation 
        val_ratio = 0.2;
        [train_data, train_label, val_data, val_label] = split_data(train_data, train_label, val_ratio);
        
        if strcmp(dataset, 'CIFAR')
    % CIFAR data is [0,1], no need to normalize
        else
    % MNIST: Ensure val_data is [0,1]
    val_data = val_data / 255.0; 
        end
        if ndims(val_data)==4
           assert(size(val_data,4) == numel(val_label), ...
               'Validation data/label count mismatch');
       else
           assert(size(val_data,3) == numel(val_label), ...
               'Validation data/label count mismatch');
       end


        % Parameters for spatial pooler
        [h, w, ~] = size(train_data); 
        potential_radius = floor( min(h, w) / 4 );
        input_dimension = [size(train_data, 1), size(train_data, 2)];
       
        

        % Apply PCA using the reshaped data
        numImgs = size(train_data, 3);
        nPCA    = max(1, round(cfg.PCA_SAMPLE_FRAC * numImgs));   % e.g. 10%
            selIdx  = randperm(numImgs, nPCA);

        % Pre-allocate (rough upper bound) to avoid growing in the loop
        pr       = potential_radius;
        h        = size(train_data,1);
        w        = size(train_data,2);
        maxPerImg = ceil(h/pr) * ceil(w/pr);
        patches   = zeros(pr^2, nPCA * maxPerImg, 'double');
        col       = 1;

        for k = 1:nPCA
    img = train_data(:,:,selIdx(k));
    img_patches = im2col(img, [pr pr], 'distinct');
    np = size(img_patches, 2);
    patches(:, col:col+np-1) = img_patches;
    col = col + np;
        end
        patches(:, col:end) = [];   % trim unused columns

        % now PCA on the reduced set
        [coeff, ~] = pca(patches', 'Centered', false);
        pca_coeff  = reshape(coeff(:,1), [pr pr]);

        

        % Optimize hyperparameters 
        [best_params, adjusted_potential_radius] = optimize_hyperparameters(train_data, train_label, val_data, val_label, dataset, potential_radius, pca_coeff);
        base_area_density = best_params.base_area_density;
        base_area_density = min(0.5, base_area_density * 1.15);
        syn_inc_base = best_params.syn_inc_base;
        syn_dec_base = best_params.syn_dec_base;
        syn_connected_thresh_base = 0.3;
        noise_std_base = 0.02;
        decay_scaling  = best_params.decay_scaling;     
        endurance_rate = best_params.endurance_rate;
        cfg.OVERLAP_MIN_ACTIVE_FRAC = 0.8 * best_params.base_area_density;
        cfg.OVERLAP_MAX_ACTIVE_FRAC = 1.2 * best_params.base_area_density;
        potential_radius  = adjusted_potential_radius;
         overlap_dimension = input_dimension - potential_radius + 1;
        assert(all(overlap_dimension > 0), 'Invalid overlap dimensions.');
        % GPU availability check
        use_gpu = gpuDeviceCount > 0;
        if use_gpu
            g = gpuDevice();
            fprintf('GPU detected: %s\n', g.Name);
            train_data = gpuArray(train_data);
        end

        % Initialize permanence using PCA coefficients and overlap dimensions.
        w_permanence = initialize_permanence(pca_coeff, potential_radius, overlap_dimension, use_gpu);
        sparsity_per_epoch = zeros(train_epoch, 1);
        entropy_per_epoch = zeros(train_epoch, 1);
        val_accuracy_history = zeros(train_epoch,1);
        
        % Initialize histories
        inhibition_radius_history = zeros(train_epoch,1);
        syn_inc_history = zeros(train_epoch,1);
        syn_dec_history = zeros(train_epoch,1);
        base_area_density_history = zeros(train_epoch, 1);
        fallbacks_per_epoch = zeros(train_epoch, 1);

        

        fprintf('Starting Training...\n');
        tic;
        fprintf('[DEBUG] Before training call: w_permanence Size: %s, Mean: %.4f\n', ...
            mat2str(size(w_permanence)), mean(w_permanence(:)));

        for tep = 1:train_epoch
            fprintf('Epoch %d/%d\n', tep, train_epoch);
            
                sample_counter = 0;  % reset at epoch star
                cfg.EPOCH_START_FLAG = true;
            % Inject noise 
            noise_std = noise_std_base * (1 - tep / train_epoch);
            train_data_noisy = max(0, min(1, train_data + noise_std * randn(size(train_data))));
            if use_gpu
                train_data_noisy = gpuArray(train_data_noisy);
            end

            % Train spatial pooler 
            [w_permanence, sample_counter, returned_history ,base_area_density,  syn_inc_base, syn_dec_base, inh_rad, inc_rate, dec_rate, e, wc, epoch_sparsity, epoch_entropy, fallback_counter_epoch] = train_spatial_pooler(...
            train_data_noisy, train_label, ...
            base_area_density, syn_inc_base, syn_dec_base, ...
            sample_counter, syn_connected_thresh_base, use_gpu, ...
            w_permanence, decay_scaling, endurance_rate, ...
            pca_coeff, potential_radius,overlap_dimension , ...
            val_data, val_label, val_accuracy_history, ...
            entropy_threshold, sparsity_threshold);
            final_val_acc = returned_history(end);
                if cfg.DEBUG
                    fprintf('DEBUG new_sp BEFORE: history = [%d %d]\n', size(val_accuracy_history));
                end
            val_accuracy_history(tep) = final_val_acc;
                if cfg.DEBUG
                    fprintf('DEBUG new_sp AFTER : history = [%d %d]\n', size(val_accuracy_history));
                end

                base_area_density_history(tep) = base_area_density;

       

            inhibition_radius_history(tep) = inh_rad;
            syn_inc_history(tep)  = inc_rate;
            syn_dec_history(tep) = dec_rate;
            energy_history(tep)      = e;
            write_cycle_history(tep) = wc;
            fallbacks_per_epoch(tep) = fallback_counter_epoch;

            
            % metrics   
            sparsity_per_epoch(tep) = epoch_sparsity;
            entropy_per_epoch(tep)  = epoch_entropy;
            
            [~, stop_epoch] = max(val_accuracy_history);
            

            fprintf('[LOG] Epoch %d/%d: Sparsity = %.2f%%, Entropy = %.4f bits\n', tep, train_epoch, epoch_sparsity, epoch_entropy);

        end
        
        if isa(w_permanence,'gpuArray')
           w_permanence = gather(w_permanence);
        end

        config = struct( ...
    'overlap_dim',          overlap_dimension, ...
    'potential_radius',     potential_radius, ...
    'syn_connected_thresh', syn_connected_thresh_base, ...
    'base_area_density',    base_area_density ...
);


          sample_idx = 1;
    example_state = struct('sample_counter', 1, 'threshold_tracker', struct());                
    [sdr_flat, example_kwta_state] = generate_sdr( ...
    train_data, w_permanence, sample_idx, config, example_state );
    sample_sdr   = reshape(sdr_flat, overlap_dimension);
    sample_input = gather(train_data(:,:,sample_idx));


        total_training_time = toc;
        fprintf('Training Complete. Total Training Time: %.2f seconds\n', total_training_time);
        
        timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
        save(['training_results_' timestamp '.mat'],   'w_permanence', 'pca_coeff' , ...
     'sample_idx', 'sample_input', 'sample_sdr', ...
     'sparsity_per_epoch', 'entropy_per_epoch', ...
     'inhibition_radius_history', ...
     'syn_inc_history', 'syn_dec_history', ...
     'val_accuracy_history', 'energy_history', ...
        'threshold_tracker', 'write_cycle_history', 'stop_epoch', ...
        'base_area_density_history', 'fallbacks_per_epoch', ...
     '-v7.3');

    catch ME
        fprintf('Error encountered: %s\n', ME.message);
        fprintf('Stack Trace:\n%s\n', getReport(ME, 'extended'));
    end

    fprintf('Logging ended at %s\n', datetime('now'));
    diary off;
end
