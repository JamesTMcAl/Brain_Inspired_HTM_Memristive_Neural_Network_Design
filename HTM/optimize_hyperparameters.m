function [best_params, adjusted_potential_radius] = optimize_hyperparameters(train_data, train_label, val_data, val_label, dataset, potential_radius, pca_coeff)
%OPTIMIZE_HYPERPARAMETERS  Evolutionary random search over SP hyperparameters.
%   Replaces bayesopt (MATLAB Statistics Toolbox only) with a two-phase
%   Octave-compatible search:
%     Phase 1 - Latin hypercube random sampling across the full space
%     Phase 2 - Local refinement around the best point found in phase 1
%   The fitness is the composite loss from evaluate_hyperparams:
%     loss = -accuracy + sparsity_penalty + entropy_penalty + energy_penalty
%   This gives the search real gradients to follow rather than the flat
%   -accuracy landscape where most parameter sets score ~100%.

    fprintf('[DEBUG] optimize_hyperparameters: potential_radius=%d\n', potential_radius);
    fprintf('[DEBUG] optimize_hyperparameters: pca_coeff size=%s\n', mat2str(size(pca_coeff)));

    cfg = sp_config.instance();
    rng(cfg.HYPER_SEED);

    adjusted_potential_radius = potential_radius;

    % Validation split if not provided
    if isempty(val_data) || isempty(val_label)
        [train_data, train_label, val_data, val_label] = split_data(train_data, train_label, 0.2);
    end

    %  Parameter bounds 
    if strcmp(dataset, 'CIFAR')
        density_bounds = [0.10, 0.40];
    else
        density_bounds = [0.05, 0.35];
    end
    syn_inc_bounds   = [0.04,  0.20];
    syn_dec_bounds   = [0.002, 0.02];   % sampled in log space
    decay_bounds     = [3000,  25000];  % sampled in log space
    endurance_bounds = [5e4,   1e6];    % sampled in log space

    %  Phase 1: random sampling 
    n_phase1 = 40;
    fprintf('[OPT] Phase 1: %d random evaluations\n', n_phase1);

    candidates = generate_candidates(n_phase1, density_bounds, syn_inc_bounds, ...
                                     syn_dec_bounds, decay_bounds, endurance_bounds);

    [best_loss, best_params] = run_evaluations(candidates, train_data, train_label, ...
                                               val_data, val_label, pca_coeff, ...
                                               potential_radius, cfg);

    fprintf('[OPT] Phase 1 best loss: %.4f\n', best_loss);

    %  Phase 2: local refinement around best point 
    n_phase2 = 20;
    fprintf('[OPT] Phase 2: %d local refinement evaluations\n', n_phase2);

    local_candidates = generate_local_candidates(n_phase2, best_params, ...
                                                  density_bounds, syn_inc_bounds, ...
                                                  syn_dec_bounds, decay_bounds, ...
                                                  endurance_bounds);

    [local_best_loss, local_best_params] = run_evaluations(local_candidates, train_data, ...
                                                            train_label, val_data, val_label, ...
                                                            pca_coeff, potential_radius, cfg);

    if local_best_loss < best_loss
        best_loss   = local_best_loss;
        best_params = local_best_params;
        fprintf('[OPT] Phase 2 improved: loss=%.4f\n', best_loss);
    else
        fprintf('[OPT] Phase 2 did not improve (best=%.4f)\n', best_loss);
    end

    %  Save and report 
    save('best_hyperparameters.mat', 'best_params');
    fprintf('[OPT] Best params: Density=%.4f | LTP=%.4f | LTD=%.4f | Decay=%.2e | Endurance=%.2e\n', ...
            best_params.base_area_density, best_params.syn_inc_base, ...
            best_params.syn_dec_base, best_params.decay_scaling, best_params.endurance_rate);
    fprintf('[OPT] Best composite loss: %.4f\n', best_loss);
end



function candidates = generate_candidates(n, density_bounds, syn_inc_bounds, ...
                                           syn_dec_bounds, decay_bounds, endurance_bounds)
%GENERATE_CANDIDATES  Latin hypercube-style random sampling across param space.

    candidates = cell(n, 1);
    for i = 1:n
        p = struct();
        p.base_area_density = density_bounds(1)  + rand() * diff(density_bounds);
        p.syn_inc_base      = syn_inc_bounds(1)   + rand() * diff(syn_inc_bounds);
        % Log-space sampling for parameters spanning orders of magnitude
        p.syn_dec_base      = exp(log(syn_dec_bounds(1))   + rand() * log(syn_dec_bounds(2)/syn_dec_bounds(1)));
        p.decay_scaling     = exp(log(decay_bounds(1))     + rand() * log(decay_bounds(2)/decay_bounds(1)));
        p.endurance_rate    = exp(log(endurance_bounds(1)) + rand() * log(endurance_bounds(2)/endurance_bounds(1)));
        candidates{i}       = p;
    end

    % Always include the dissertation-optimised point as candidate 1
    candidates{1} = struct('base_area_density', 0.3085, 'syn_inc_base', 0.0975, ...
                           'syn_dec_base', 0.0040, 'decay_scaling', 5797, ...
                           'endurance_rate', 179340);
end


function candidates = generate_local_candidates(n, centre, density_bounds, ...
                                                  syn_inc_bounds, syn_dec_bounds, ...
                                                  decay_bounds, endurance_bounds)
%GENERATE_LOCAL_CANDIDATES  Gaussian perturbations around the current best point.

    candidates = cell(n, 1);
    candidates{1} = centre;  % always include the centre itself

    for i = 2:n
        p = struct();
        % Perturb by ~15% of each parameter range, clamp to bounds
        p.base_area_density = clamp(centre.base_area_density + 0.15*diff(density_bounds)*randn(), ...
                                    density_bounds(1), density_bounds(2));
        p.syn_inc_base      = clamp(centre.syn_inc_base + 0.15*diff(syn_inc_bounds)*randn(), ...
                                    syn_inc_bounds(1), syn_inc_bounds(2));
        % Log-space perturbation for log-scale params
        p.syn_dec_base      = clamp(centre.syn_dec_base * exp(0.3*randn()), ...
                                    syn_dec_bounds(1), syn_dec_bounds(2));
        p.decay_scaling     = clamp(centre.decay_scaling * exp(0.3*randn()), ...
                                    decay_bounds(1), decay_bounds(2));
        p.endurance_rate    = clamp(centre.endurance_rate * exp(0.3*randn()), ...
                                    endurance_bounds(1), endurance_bounds(2));
        candidates{i}       = p;
    end
end


function [best_loss, best_params] = run_evaluations(candidates, train_data, train_label, ...
                                                     val_data, val_label, pca_coeff, ...
                                                     potential_radius, cfg)
%RUN_EVALUATIONS  Evaluate each candidate and return the best.

    best_loss   = Inf;
    best_params = candidates{1};

    for i = 1:numel(candidates)
        p = candidates{i};
        fprintf('[OPT] Eval %d/%d: density=%.3f inc=%.4f dec=%.4f decay=%.0f end=%.0e\n', ...
                i, numel(candidates), p.base_area_density, p.syn_inc_base, ...
                p.syn_dec_base, p.decay_scaling, p.endurance_rate);
        try
            loss = evaluate_hyperparams(train_data, train_label, val_data, val_label, ...
                                        p, pca_coeff, potential_radius);
            if loss < best_loss
                best_loss   = loss;
                best_params = p;
                fprintf('[OPT] New best at eval %d: loss=%.4f\n', i, best_loss);
            end
        catch ME
            fprintf('[OPT] Eval %d failed: %s\n', i, ME.message);
        end
    end
end


function val = clamp(val, lo, hi)
    val = min(max(val, lo), hi);
end