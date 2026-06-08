function [loss, adjusted_potential_radius] = evaluate_hyperparams(train_data, train_label, val_data, val_label, params, pca_coeff, potential_radius)
%EVALUATE_HYPERPARAMS  One trial of SP training returning composite fitness.
%   Composite loss = -accuracy
%                  + sparsity_deviation_penalty
%                  + entropy_collapse_penalty
%                  + energy_cost_penalty
%   This replaces the flat -accuracy objective which produced a nearly
%   constant landscape (trivially separable patterns score ~100% for most
%   parameter sets, giving the optimizer nothing to search against).

    cfg = sp_config.instance();

    % Defaults for output in case of early error
    loss                     = Inf;
    adjusted_potential_radius = potential_radius;

    try
        %  Data subsetting 
        subset_train_count = min(200, size(train_data, 3));
        subset_val_count   = min(50,  size(val_data,   3));
        assert(subset_train_count > 0, 'Training subset is empty.');
        assert(subset_val_count   > 0, 'Validation subset is empty.');

        train_data_subset  = train_data(:, :, 1:subset_train_count);
        train_label_subset = train_label(1:subset_train_count);
        val_data_subset    = val_data(:, :, 1:subset_val_count);
        val_label_subset   = val_label(1:subset_val_count);

        assert(size(train_data_subset,1) == size(val_data_subset,1) && ...
               size(train_data_subset,2) == size(val_data_subset,2), ...
               'train_data and val_data spatial dimensions must match.');

        % GPU transfer if enabled
        if cfg.USE_GPU
            train_data_subset = gpuArray(double(train_data_subset));
            val_data_subset   = gpuArray(double(val_data_subset));
        end

        %  Overlap dimensions 
        input_size        = [size(train_data_subset,1), size(train_data_subset,2)];
        overlap_dimension = input_size - (potential_radius - 1);

        %  Permanence initialisation 
        initial_w = initialize_permanence(pca_coeff, potential_radius, overlap_dimension, cfg.USE_GPU);
        if cfg.USE_GPU
            initial_w = gpuArray(double(initial_w));
        end

        %  Train SP, capturing metrics 
        try
            [best_weights, ~, ~, ~, ~, ~, ~, ~, ~, epoch_energy, ~, epoch_sparsity, epoch_entropy] = ...
                train_spatial_pooler( ...
                    train_data_subset, train_label_subset, ...
                    params.base_area_density, params.syn_inc_base, params.syn_dec_base, ...
                    0, cfg.SYN_CONNECTED_INIT, false, ...
                    initial_w, params.decay_scaling, params.endurance_rate, ...
                    pca_coeff, potential_radius, overlap_dimension, ...
                    val_data_subset, val_label_subset, [], ...
                    cfg.ENTROPY_THRESHOLD_INIT, cfg.SPARSITY_THRESHOLD_INIT);
        catch ME
            fprintf('[ERROR] train_spatial_pooler failed in evaluate_hyperparams:\n%s\n', ...
                    getReport(ME, 'extended', 'hyperlinks', 'off'));
            loss = Inf;
            return;
        end

        %  Compute loss 
        if isempty(best_weights)
            warning('evaluate_hyperparams:EmptyWeights', ...
                    'train_spatial_pooler returned empty weights.');
            loss = Inf;
            return;
        end

        % Classification accuracy
        preds = infer_labels(val_data_subset, best_weights, ...
                             train_data_subset, train_label_subset, overlap_dimension);
        acc   = mean(preds(:) == val_label_subset(:));

        % Sparsity deviation penalty - penalise both collapse and over-density
        % Target is params.base_area_density since that's what the SP was asked for
        target_sparsity = max(0.02, params.base_area_density);
        sparsity_pen    = abs(epoch_sparsity / 100 - target_sparsity) * 10;

        % Entropy collapse penalty - penalise representations with no diversity
        entropy_pen = max(0, 0.2 - epoch_entropy) * 5;

        % Energy cost penalty - normalised write energy (memristor endurance proxy)
        % epoch_energy is in Joules; 1e-2 J/epoch is a soft budget
        energy_pen = min(1.0, epoch_energy / 1e-2);

        % Composite loss (minimised by optimizer)
        loss = -acc + sparsity_pen + entropy_pen + 0.1 * energy_pen;

        fprintf('[EVAL] acc=%.3f | sparsity=%.2f%% | entropy=%.4f | energy=%.2e | loss=%.4f\n', ...
                acc, epoch_sparsity, epoch_entropy, epoch_energy, loss);

    catch ME
        fprintf('[ERROR] evaluate_hyperparams failed:\n%s\n', ...
                getReport(ME, 'extended', 'hyperlinks', 'off'));
        loss = Inf;
    end
end