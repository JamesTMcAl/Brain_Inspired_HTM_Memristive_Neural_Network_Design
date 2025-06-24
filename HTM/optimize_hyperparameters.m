function [best_params, adjusted_potential_radius] = optimize_hyperparameters(train_data, train_label, val_data, val_label, dataset, potential_radius, pca_coeff )
    fprintf('[DEBUG] optimize_hyperparameters: potential_radius=%d\n', potential_radius);
fprintf('[DEBUG] optimize_hyperparameters: pca_coeff size=%s\n', mat2str(size(pca_coeff)));

cfg = sp_config.instance();
        rng(cfg.HYPER_SEED);
        if isempty(val_data) || isempty(val_label)
        val_ratio = 0.2;  % 20% validation split
        [train_data, train_label, val_data, val_label] = split_data(train_data, train_label, val_ratio);
        end


% Handle range constraints depending on the dataset
    if strcmp(dataset, 'CIFAR')
        density_var = optimizableVariable('base_area_density', [0.3, 0.5]);
    else
        density_var = optimizableVariable('base_area_density', [0.3, 0.45]);
    end

    syn_inc_var  = optimizableVariable('syn_inc_base', [0.04, 0.2]);
    syn_dec_var  = optimizableVariable('syn_dec_base', [0.002, 0.02], 'Transform', 'log'); 
    decay_var    = optimizableVariable('decay_scaling', [5000, 20000], 'Transform', 'log');
    endurance_var = optimizableVariable('endurance_rate', [1e5, 1e6], 'Transform', 'log');

    param_ranges = [density_var, syn_inc_var, syn_dec_var, decay_var, endurance_var];

    

    
    
    adjusted_potential_radius = potential_radius;
    


    function loss = safeEvaluate(params)
        
        try
        id = getCurrentTask.ID;
    catch
        id = 0;
        end
        rng(cfg.HYPER_SEED + id);

    fprintf('Worker %d evaluating params: %s\n', id, jsonencode(params));

    try
        loss = evaluate_hyperparams( ...
            train_data, train_label, ...
            val_data, val_label, ...
            params, pca_coeff, potential_radius );
    catch ME
        fprintf('--- ERROR on worker %d ---\n', id);
        fprintf('Params: %s\n', jsonencode(params));
        fprintf('%s\n', getReport(ME,'extended','hyperlinks','off'));

        save('lastHyperparamError.mat','ME','params');

        loss = Inf;  
    end
    end

        use_parallel = ~isempty(gcp('nocreate'));
            % Perform Bayesian optimization
    try
        results = bayesopt(@safeEvaluate, param_ranges, ...
            'AcquisitionFunctionName','expected-improvement-plus', ...
            'MaxObjectiveEvaluations',200, ...
            'ExplorationRatio',0.8, ...
            'UseParallel',use_parallel, ...
            'PlotFcn',{@plotObjectiveModel,@plotMinObjective});
    catch ME
        % Fallback to serial if something outside safeEvaluate goes wrong
        warning(ME.identifier, '%s', ME.message);
        results = bayesopt(@safeEvaluate, param_ranges, ...
            'AcquisitionFunctionName','expected-improvement-plus', ...
            'MaxObjectiveEvaluations',200, ...
            'ExplorationRatio',0.8, ...
            'UseParallel',false, ...
            'PlotFcn',{@plotObjectiveModel,@plotMinObjective});
    end
        
    
        % If MATLAB exposes ObjectiveMinimumTrace, use it directly
        if isprop(results, 'ObjectiveMinimumTrace')
    bestSoFar = results.ObjectiveMinimumTrace;
        else
    % Otherwise, compute it from the raw trace
    raw = results.ObjectiveTrace;
    bestSoFar = cummin(raw);
        end

plot(1:numel(bestSoFar), bestSoFar, '-o', 'MarkerSize', 4);
xlabel('Evaluation #');
ylabel('Min Objective (–Accuracy)');
title('Bayesian Optimization Convergence');
grid on;




    if ~isempty(results.XAtMinObjective)
        best_params = results.XAtMinObjective;
        save('best_hyperparameters.mat', 'best_params');
        fprintf('Optimized Params: Density=%.4f, LTP=%.4f, LTD=%.4f, Decay=%.2e, Endurance=%.2e\n', ...
            best_params.base_area_density, ...
            best_params.syn_inc_base, ...
            best_params.syn_dec_base, ...
            best_params.decay_scaling, ...
            best_params.endurance_rate);
    else
        error('Optimization failed: No feasible points found.');
    end
    

end
