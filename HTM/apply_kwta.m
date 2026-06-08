function [active_columns, avg_activity, state, base_area_density] = apply_kwta(overlap, base_area_density, sample_counter, use_gpu, reset, state)
% APPLY_KWTA  Global hard k-WTA inhibition with homeostatic duty-cycle boosting.
%   Selects exactly k = round(N * base_area_density) columns by rank on the
%   boosted overlap score. Replaces local imdilate inhibition which cannot
%   reach k on small grids (e.g. 8x8 = 64 columns with radius 3 blocks ~49
%   neighbours per winner, making k > 2 physically unreachable).
%   Keeps: radius tracking (read by train_spatial_pooler), duty-cycle boosting,
%          dead-column tracking, structural rewiring flag, refractory map,
%          stochastic exploration, fallback safety net.
%   Removes: local imdilate loop, k_adjust_factor rescaling, density auto-inflation.

    validateattributes(overlap, {'numeric'}, {'2d','nonnegative','finite'}, mfilename, 'overlap', 1);
    validateattributes(base_area_density, {'numeric'}, {'scalar','>=',0,'<=',1}, mfilename, 'base_area_density', 2);

    % Ensure outputs always defined (Octave compatibility)
    active_columns = false(size(overlap));
    avg_activity   = 0;

    cfg = sp_config.instance();

    
    % State initialisation
    
    if reset || ~isfield(state, 'fallback_counter')
        state.fallback_counter = 0;
    end
    if reset || ~isfield(state, 'avg_overlap')
        state.avg_overlap = mean(overlap(:));
    end
    if reset || ~isfield(state, 'inhibition_radius')
        state.inhibition_radius         = [];
        state.radius_history            = [];
        state.prev_overlap              = [];
        state.refractory_map            = zeros(size(overlap));
        state.inhibition_radius_history = [];
        state.activity_map              = 0.5 * ones(size(overlap));
        state.duty_cycle_map            = 0.01 * ones(size(overlap), 'like', overlap);
    end

    
    % Inhibition radius tracking (kept for train_spatial_pooler logging)
    
    if sample_counter == 1 || isempty(state.inhibition_radius)
        if std(overlap(:)) < 1e-6
            spatial_corr  = 0;
            temporal_corr = 0;
        else
            tmp = corr(overlap(1:end-1,:)', overlap(2:end,:)');
            tmp(isnan(tmp)) = 0;
            spatial_corr = mean(tmp(:));
            tmp = corr(overlap(:,1:end-1), overlap(:,2:end));
            tmp(isnan(tmp)) = 0;
            temporal_corr = mean(tmp(:));
        end
        initR = cfg.KWTA_INIT_RADIUS;
        scale = cfg.KWTA_RADIUS_SCALE;
        state.inhibition_radius = round(initR - scale * (spatial_corr + temporal_corr));
        state.radius_history    = repmat(state.inhibition_radius, 1, cfg.KWTA_RADIUS_HISTORY);
        state.prev_overlap      = mean(overlap(:));
        state.inhibition_radius_history = state.inhibition_radius;
    end

    % Local variance -> adaptive radius (kept for radius history logging)
    w0     = max(cfg.KWTA_LOCAL_MIN, round(sqrt(numel(overlap) / 10)));
    kernel = ones(w0, 'like', overlap) / (w0^2);
    mu     = conv2(overlap, kernel, 'same');
    mu2    = conv2(overlap.^2, kernel, 'same');
    local_var = mu2 - mu.^2;

    R = max(cfg.KWTA_LOCAL_MIN, ...
            round(cfg.KWTA_LOCAL_MAX - cfg.KWTA_LOCAL_SCALE * (local_var ./ (max(local_var(:)) + eps))));
    state.inhibition_radius = round(mean(R(:)));

    % Smooth radius
    state.radius_history = [state.radius_history(2:end), state.inhibition_radius];
    final_radius = max(cfg.KWTA_LOCAL_MIN, round(mean(state.radius_history)));

    % Correlation tweak
    curr = mean(overlap(:));
    if curr < state.prev_overlap && state.inhibition_radius < cfg.INHIB_RADIUS_MAX
        state.inhibition_radius = state.inhibition_radius + 1;
    elseif curr > state.prev_overlap && state.inhibition_radius > cfg.INHIB_RADIUS_MIN
        state.inhibition_radius = state.inhibition_radius - 1;
    end
    state.prev_overlap = curr;
    state.inhibition_radius_history(end+1) = state.inhibition_radius;

    
    % DETERMINE k     
    min_active = max(round(cfg.KWTA_MIN_ACTIVE_FRAC * numel(overlap)), 2);
    k = max(round(numel(overlap) * base_area_density), min_active + 1);
    k = min(k, numel(overlap) - 1);

    
    % Normalise + duty-cycle boosting
    
    mn     = min(overlap(:));
    vr     = max(overlap(:)) - mn + eps;
    normed = (overlap - mn) / vr;

    boost_factors = exp(cfg.KWTA_BETA_BOOST * (cfg.KWTA_TARGET_DUTY - state.duty_cycle_map));
    normed = normed .* boost_factors;

    if cfg.Debug_Overlap_Tracking
        fprintf('[DEBUG] normed min=%.4f, mean=%.4f, max=%.4f\n', ...
                min(normed(:)), mean(normed(:)), max(normed(:)));
    end

    
    % Symmetry-breaking noise
    
    if sample_counter < 1000
        base_noise = 0.1;
    else
        base_noise = cfg.KWTA_NOISE_LEVEL + 0.03 * exp(-sample_counter / 25000);
    end

    noise_scale = base_noise + 0.1 * (1 - state.activity_map);
    noisy = normed + noise_scale .* randn(size(normed));

    [~, sorted_idx] = sort(noisy(:), 'descend');

    
    % GLOBAL k-WTA: select exactly k columns by rank
    % Refractory map respected 
    
    active_columns = false(size(overlap));
    selected = 0;

    % Pass 1: top-k among non-refractory columns
    for idx = sorted_idx'
        if state.refractory_map(idx) == 0
            active_columns(idx) = true;
            state.refractory_map(idx) = cfg.REFRACTORY_PERIOD;
            selected = selected + 1;
            if selected >= k, break; end
        end
    end

    % Pass 2: if refractory blocked too many, fill from top remaining
    if selected < k
        for idx = sorted_idx'
            if ~active_columns(idx)
                active_columns(idx) = true;
                selected = selected + 1;
                if selected >= k, break; end
            end
        end
    end

    state.refractory_map = max(0, state.refractory_map - 1);
    state.activity_map   = 0.99 * state.activity_map + 0.01 * double(active_columns);

    if cfg.DEBUG && mod(sample_counter, cfg.DEBUG_INTERVAL) == 0
        fprintf('[DEBUG] Global k-WTA: selected %d/%d columns (k=%d)\n', ...
                nnz(active_columns), numel(active_columns), k);
    end

    
    % Dead-column tracking + structural rewiring flag
    
    if ~isfield(state, 'dead_counter')
        state.dead_counter = zeros(size(overlap));
    end

    state.dead_counter = state.dead_counter + double(~active_columns);
    state.dead_counter(active_columns) = 0;

    if mod(sample_counter, 300) == 0
        % Soft boost for columns dead > 100 steps
        dead_cols = state.dead_counter > 100;
        if any(dead_cols(:))
            fprintf('[BOOST] Reactivating %d dead columns.\n', nnz(dead_cols));
            state.refractory_map(dead_cols) = 0;
            state.dead_counter(dead_cols)   = 0;
        end

        % Structural rewiring for persistently dead columns (> 500 steps)
        rewire_cols = state.dead_counter > 500;
        if any(rewire_cols(:))
            fprintf('[REWIRE] Flagging %d persistently dead columns for rewiring\n', nnz(rewire_cols));
            state.rewire_flag = rewire_cols;
            state.dead_counter(rewire_cols) = 0;
        else
            state.rewire_flag = false(size(overlap));
        end
    else
        if ~isfield(state, 'rewire_flag')
            state.rewire_flag = false(size(overlap));
        end
    end

    
    % Fallback safety net (should rarely fire with hard-k)
    
    if nnz(active_columns) < min_active
        nf = round(min_active * cfg.KWTA_FALLBACK_FACTOR);
        nz = find(overlap(:) > 0);
        if isempty(nz)
            sel = randperm(numel(overlap), nf);
        else
            [~, ord] = sort(overlap(nz), 'descend');
            sel = nz(ord(1:min(nf, numel(ord))));
        end
        active_columns(sel) = true;

        if cfg.fallback_Debug
            fprintf('[WARNING] Fallback @ iter %d: %d -> %d active (min=%d)\n', ...
                    sample_counter, nnz(active_columns) - nf, nnz(active_columns), min_active);
        end

        state.fallback_counter = state.fallback_counter + 1;

        % Drive boosting rather than inflating density target
        if state.fallback_counter > 50 && mod(state.fallback_counter, 10) == 0
            fprintf('[PATCH] Too many fallbacks - reducing duty-cycle map to boost underactive columns.\n');
            state.duty_cycle_map = state.duty_cycle_map * 0.95;
        end
    end

    
    % Borderline boost + homeostasis bookkeeping
    
    margin    = 0.1;
    max_val   = max(normed(:));
    borderline = (normed > (max_val - margin)) & ~active_columns;
    boost_prob = 0.3;
    active_columns = active_columns | (rand(size(active_columns)) < boost_prob & borderline);

    state.activity_map   = 0.99 * state.activity_map   + 0.01 * double(active_columns);
    state.duty_cycle_map = 0.99 * state.duty_cycle_map + 0.01 * double(active_columns);

    avg_activity = mean(active_columns(:));

    if use_gpu
        active_columns = gpuArray(active_columns);
    end

    
    % Stochastic exploration (fires rarely only when very sparse)
    
    if mod(sample_counter, cfg.KWTA_STOCH_INTERVAL) == 0 && avg_activity < 0.08
        rate = max(cfg.KWTA_STOCH_MIN, cfg.KWTA_STOCH_BASE - avg_activity);
        active_columns = active_columns | (rand(size(active_columns)) < rate);
        fprintf('[LOG] Stochastic @ %d\n', sample_counter);
    end

    
    % Debug print + reject flag
    
    if cfg.DEBUG && mod(sample_counter, cfg.DEBUG_INTERVAL) == 0
        fprintf('[DEBUG] Iter %d | Active=%d | Rad=%d | k=%d | MeanOv=%.4f\n', ...
                sample_counter, nnz(active_columns), final_radius, k, mean(overlap(:)));
    end

    fallback_rate = state.fallback_counter / max(sample_counter, 1);
    if fallback_rate > 0.6 && sample_counter > 50
        state.reject_flag = true;
        fprintf('[REJECT] Fallback rate too high (%.2f)\n', fallback_rate);
    else
        state.reject_flag = false;
    end

    if state.fallback_counter > 10 && mod(state.fallback_counter, 25) == 0
        fprintf('[KWTA] Fallback count=%d - duty-cycle boosting active\n', state.fallback_counter);
    end

end