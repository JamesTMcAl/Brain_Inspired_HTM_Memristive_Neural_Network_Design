function [loss, adjusted_potential_radius] = evaluate_hyperparams(train_data, train_label, val_data, val_label, params, pca_coeff, potential_radius)
    % EVALUATE_HYPERPARAMS  Run one trial of spatial pooler training & return negative accuracy.
    % Implements fixed RNG seed and PCA-based permanence initialization for determinism.
    %
    %   loss = evaluate_hyperparams(train_data, train_label, ...
    %                                val_data,   val_label, ...
    %                                params,     pca_coeff)
    %
    % Inputs:
    %   train_data    — [H×W×Ntrain] input patterns
    %   train_label   — [Ntrain×1]  labels
    %   val_data      — [H×W×Nval]   validation patterns
    %   val_label     — [Nval×1]    labels
    %   params        — struct with fields:
    %                    .base_area_density, .syn_inc_base, .syn_dec_base,
    %                    .decay_scaling, .endurance_rate
    %   pca_coeff     — [R×R] PCA patch used for initialization
    %
    % Output:
    %   loss — negative validation accuracy (to be minimized)

    % Grab our global config
    cfg = sp_config.instance();
    
    try
        % Subset for fast prototyping & debug prints
        subset_train_count = min(200, size(train_data, 3));
        subset_val_count   = min(50,  size(val_data,   3));
        assert(subset_train_count > 0, 'Training subset is empty!');
        assert(subset_val_count   > 0, 'Validation subset is empty!');
        
        train_data_subset  = train_data(:, :, 1:subset_train_count);
        train_label_subset = train_label(1:subset_train_count);
        val_data_subset    = val_data(:, :, 1:subset_val_count);
        val_label_subset   = val_label(1:subset_val_count);

        assert(isequal(size(train_data_subset,1), size(val_data_subset,1)) && ...
       isequal(size(train_data_subset,2), size(val_data_subset,2)), ...
       'train_data and val_data must have the same height and width for now.');

        if cfg.USE_GPU
    if ~isa(train_data_subset, 'double')
        train_data_subset = double(train_data_subset);
    end
    if ~isa(val_data_subset, 'double')
        val_data_subset = double(val_data_subset);
    end
    train_data_subset = gpuArray(train_data_subset);
    val_data_subset = gpuArray(val_data_subset);
        end
        if cfg.DEBUG
        fprintf('[DEBUG] Training: %d samples, Validation: %d samples\n', ...
                subset_train_count, subset_val_count);
        end
       
        
        % Compute overlap dimensions
        
        adjusted_potential_radius = potential_radius;
        input_size       = [size(train_data_subset,1), size(train_data_subset,2)];
        overlap_dimension      = input_size - (potential_radius - 1);
        
        % PCA-based permanence initialization
        % initialize_permanence handles USE_PCA_INIT and fixed fallback
        initial_w = initialize_permanence( ...
            pca_coeff, ...                % PCA patches
            potential_radius, ...
            overlap_dimension, ...
            cfg.USE_GPU );           
        if cfg.USE_GPU
            if ~isa(initial_w, 'double')
                initial_w = double(initial_w);
            end
                initial_w = gpuArray(initial_w);
        end
        % Train spatial pooler
        try
            fprintf('[DEBUG] train_data_subset size: %s | train_label_subset size: %s\n', mat2str(size(train_data_subset)), mat2str(size(train_label_subset)));
        fprintf('[DEBUG] val_data_subset size: %s | val_label_subset size: %s\n', mat2str(size(val_data_subset)), mat2str(size(val_label_subset)));

            [best_weights, ~, ~,  ~, ~, ~, ~, ~, ~,~, ~, ~, ~] = train_spatial_pooler( ...
        train_data_subset, train_label_subset, ...
        params.base_area_density, params.syn_inc_base, params.syn_dec_base, ...
        0, cfg.SYN_CONNECTED_INIT, false, ...
        initial_w, params.decay_scaling, params.endurance_rate, ...
        pca_coeff, potential_radius, overlap_dimension, ...
        val_data_subset, val_label_subset, [], ...
        cfg.ENTROPY_THRESHOLD_INIT, ...
        cfg.SPARSITY_THRESHOLD_INIT );

        catch ME
             fprintf('[ERROR] Error during train_spatial_pooler:\n');
    fprintf('[ERROR] Message: %s\n', ME.message);
    fprintf('[ERROR] Error ID: %s\n', ME.identifier);
    fprintf('[ERROR] Stack Trace:\n');
    for k = 1:length(ME.stack)
        fprintf('[ERROR] %s in %s at line %d\n', ME.stack(k).name, ME.stack(k).file, ME.stack(k).line);
    end

    % log  error object for full report
    fullErrorReport = getReport(ME, 'extended', 'hyperlinks', 'off');
    fprintf('[ERROR] Full Error Report:\n%s\n', fullErrorReport);
            return;
        end
        
        % Computes validation accuracy or return Inf on failure
        if isempty(best_weights)
            warning('evaluate_hyperparams:EmptyWeights', ...
                    'train_spatial_pooler returned empty weights');
            loss = Inf;
        else
            preds = infer_labels( ...
                val_data_subset, best_weights, ...
                train_data_subset, train_label_subset, overlap_dimension);
            acc  = mean(preds(:) == val_label_subset(:));
            loss = -acc;   % BO minimizes
        end
            catch ME
        fprintf(2, '[ERROR] evaluate_hyperparams failed:\n%s\n', ...
                getReport(ME,'extended','hyperlinks','off'));
        loss = Inf;
    end
end
