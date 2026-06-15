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
        %  Data subsetting. scales with dataset
        n_tr = size(train_data, 3);
        n_vl = size(val_data, 3);
        subset_train_count = min(max(150, round(n_tr * 0.10)), n_tr);
        subset_val_count   = min(max(40,  round(n_vl * 0.10)), n_vl);
        assert(subset_train_count > 0, 'Training subset is empty.');
        assert(subset_val_count   > 0, 'Validation subset is empty.');
        % Normalise labels to row vectors indexing is shape-agnostic
        train_label = train_label(:).';
        val_label   = val_label(:).';
        rng(cfg.HYPER_SEED);  % reproducible subset across evals
        tr_idx = sort(randperm(n_tr, subset_train_count));
        vl_idx = sort(randperm(n_vl, subset_val_count));
        train_data_subset  = train_data(:, :, tr_idx);
        train_label_subset = train_label(tr_idx);
        val_data_subset    = val_data(:, :, vl_idx);
        val_label_subset   = val_label(vl_idx);

        assert(size(train_data_subset,1) == size(val_data_subset,1) && ...
               size(train_data_subset,2) == size(val_data_subset,2), ...
               'train_data and val_data spatial dimensions must match.');

        % GPU transfer if enabled
        if cfg.USE_GPU  && safe_gpuDeviceCount() > 0
            train_data_subset = gpuArray(double(train_data_subset));
            val_data_subset   = gpuArray(double(val_data_subset));
        end

        %  Overlap dimensions
        input_size        = [size(train_data_subset,1), size(train_data_subset,2)];
        overlap_dimension = input_size - (potential_radius - 1);

        %  Permanence initialisation
        initial_w = initialize_permanence(pca_coeff, potential_radius, overlap_dimension, cfg.USE_GPU);
        if cfg.USE_GPU && safe_gpuDeviceCount() > 0
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
            fprintf('[ERROR] train_spatial_pooler failed in evaluate_hyperparams:\n%s\n', ME.message);
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

        % Sparsity deviation penalty - anchored to an absolute target band, not params.base_area_density.
        % Anchoring to the self-chosen density lets the optimiser score well by simply lowering its own target
        sparsity_frac   = epoch_sparsity / 100;
        target_sparsity = 0.10;                       % absolute task-agnostic ideal
        sparsity_band   = 0.05;                        % tolerance before penalty bites
        sparsity_dev    = max(0, abs(sparsity_frac - target_sparsity) - sparsity_band);
        sparsity_pen    = sparsity_dev * 8;

        % Hard floor penalty
        if sparsity_frac < 0.02
            sparsity_pen = sparsity_pen + (0.02 - sparsity_frac) * 50;
        end

        % Entropy collapse penalty, penalise representations with no diversity
        entropy_pen = max(0, 0.2 - epoch_entropy) * 5;

        % Energy cost penalty, ormalised write energy epoch_energy is in Joules; 1e-2 J/epoch is soft budget
        energy_pen = min(1.0, epoch_energy / 1e-2);

        % Composite loss
        loss = -acc * 3 + sparsity_pen + entropy_pen + 0.1 * energy_pen;

        fprintf('[EVAL] acc=%.3f | sparsity=%.2f%% | entropy=%.4f | energy=%.2e | loss=%.4f\n', ...
                acc, epoch_sparsity, epoch_entropy, epoch_energy, loss);

    catch ME
        fprintf('[ERROR] evaluate_hyperparams failed:\n%s\n', ME.message);
        loss = Inf;
    end
end
