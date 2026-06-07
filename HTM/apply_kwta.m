function [active_columns, avg_activity, state, base_area_density] = apply_kwta(overlap, base_area_density, sample_counter, use_gpu, reset, state)
% APPLY_KWTA  Adaptive k‑WTA inhibition ↔ lateral inhibition in cortex
    validateattributes(overlap,{'numeric'}, {'2d','nonnegative','finite'}, mfilename,'overlap',1);
    validateattributes(base_area_density,{'numeric'},{'scalar','>=',0,'<=',1}, mfilename,'base_area_density',2);
    % Ensure output is always defined (Octave compatibility)
    active_columns = false(size(overlap)); avg_activity = 0;
    active_columns = false(size(overlap));
avg_activity = 0;
    cfg = sp_config.instance();
    
    if reset || ~isfield(state,'fallback_counter')
    state.fallback_counter = 0;
    end
    if reset || ~isfield(state,'avg_overlap')
    state.avg_overlap = mean(overlap(:));
    end


    % Reset state if requested
    if reset || ~isfield(state,'inhibition_radius')
        state.inhibition_radius           = [];
        state.radius_history              = [];
        state.prev_overlap                = [];
        state.refractory_map              = zeros(size(overlap));
        state.inhibition_radius_history   = [];
        state.activity_map = 0.5 * ones(size(overlap));
        state.duty_cycle_map = 0.01 * ones(size(overlap), 'like', overlap);  

    end

    % INITIALIZATION on first call
    if sample_counter==1 || isempty(state.inhibition_radius)
        % global spatial & temporal corr
        if std(overlap(:))<1e-6
            spatial_corr  = 0;
            temporal_corr = 0;
        else
            tmp = corr(overlap(1:end-1,:)', overlap(2:end,:)');
            tmp(isnan(tmp)) = 0;
            spatial_corr = mean(tmp(:));
            tmp = corr(overlap(:,1:end-1), overlap(:,2:end));
            tmp(isnan(tmp)) = 0;
            temporal_corr = mean(tmp(:));
        end  % <-- closes the std check
        initR = cfg.KWTA_INIT_RADIUS;
        scale = cfg.KWTA_RADIUS_SCALE;
        state.inhibition_radius = round(initR - scale*(spatial_corr + temporal_corr));
        state.radius_history    = repmat(state.inhibition_radius,1,cfg.KWTA_RADIUS_HISTORY);
        state.prev_overlap      = mean(overlap(:));
        state.inhibition_radius_history = state.inhibition_radius;
    end

    % LOCAL variance -> adaptive radius
    w0 = max(cfg.KWTA_LOCAL_MIN, round(sqrt(numel(overlap)/10)));
    kernel = ones(w0, 'like', overlap) / (w0^2);
    mu = conv2(overlap, kernel, 'same');
    mu2 = conv2(overlap.^2, kernel, 'same');
    local_var = mu2 - mu.^2;

    R = max(cfg.KWTA_LOCAL_MIN, ...
            round(cfg.KWTA_LOCAL_MAX - cfg.KWTA_LOCAL_SCALE*(local_var./(max(local_var(:))+eps))));
    state.inhibition_radius = round(mean(R(:)));

    % smooth radius
    state.radius_history = [state.radius_history(2:end), state.inhibition_radius];
    final_radius = max(cfg.KWTA_LOCAL_MIN, round(mean(state.radius_history)));

    % backup correlation tweak
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
    
        if sample_counter < 1000
    ta = min(0.6, base_area_density + 0.15);  
        else
    ta = base_area_density + cfg.KWTA_TEMP_GROWTH_RATE * (1 - exp(-sample_counter/cfg.KWTA_TIME_CONST));
        end
    k  = max(round(numel(overlap)*ta), min_active + 5);

    % NORMALIZE + NOISE
    mn = min(overlap(:));
    vr = max(overlap(:)) - mn + eps;
    normed = (overlap - mn)/vr;
    % BOOSTING
    target_duty_cycle = cfg.KWTA_TARGET_DUTY;   
    beta_boost = cfg.KWTA_BETA_BOOST;          
    % Compute
    boost_factors = exp(beta_boost * (target_duty_cycle - state.duty_cycle_map));
    % Apply
    normed = normed .* boost_factors;
    if cfg.Debug_Overlap_Tracking
            fprintf('[DEBUG] normed min=%.4f, mean=%.4f, max=%.4f\n', min(normed(:)), mean(normed(:)), max(normed(:)));
    end
        % Adaptive k based on smoothed overlap mean 
        
        
    state.avg_overlap = mean(normed(:));

        

        % Exponential-moving average of the current overlap 
        state.avg_overlap = 0.9*state.avg_overlap + 0.1*mean(normed(:));

        % Set adjustment factor based on avg_overlap
        k_adjust_factor = 1.0;
        if state.avg_overlap < 0.15
    k_adjust_factor = 0.8;   % Gently loosen inhibition
        elseif state.avg_overlap > 0.6
    k_adjust_factor = 1.1;   % Gently tighten inhibition
        end

        % Adjust k
        old_k = k;
        k = round(k * k_adjust_factor);

        % Clamp k to safe range
        k = max(min_active+5, min(k, numel(overlap)-5));
        if cfg.DEBUG && mod(sample_counter, cfg.DEBUG_INTERVAL) == 0
        fprintf('[DEBUG] kWTA adaptive k: %d (k_adjust=%.2f, state.avg_overlap=%.4f)\n', k, k_adjust_factor, state.avg_overlap);
        end
        if cfg.DEBUG && mod(sample_counter, cfg.DEBUG_INTERVAL) == 0
            fprintf('[kWTA] mean_overlap=%.2f → k=%d (adjusted from %d by %.2f)\n',  state.avg_overlap, k, old_k, k_adjust_factor);
        end

    

            if sample_counter < 1000
    base_noise = 0.1; % high noise early
            else
    base_noise = cfg.KWTA_NOISE_LEVEL + 0.03*exp(-sample_counter/25000);
            end

    noise_scale = base_noise + 0.1 * (1 - state.activity_map);
    noisy = normed + noise_scale .* randn(size(normed));

    [~, sorted_idx] = sort(noisy(:), 'descend');


    % INHIBITION LOOP
    
    avail = state.refractory_map==0;
    mask = avail;
    active_columns = false(size(overlap));
    SE = strel('square', 2 * final_radius + 1);

    if cfg.DEBUG && mod(sample_counter, cfg.DEBUG_INTERVAL) == 0
        fprintf('[DEBUG] Starting inhibition loop with %d candidates\n', numel(sorted_idx));
    end
        for idx = sorted_idx'
    [r, c] = ind2sub(size(overlap), idx);
    if mask(r, c)
        active_columns(r, c) = true;
        state.refractory_map(r, c) = cfg.REFRACTORY_PERIOD;

        temp_mask = false(size(overlap));
        temp_mask(r, c) = true;
        temp_mask = imdilate(temp_mask, SE);
        mask = mask & ~temp_mask;

        if cfg.Debug_Overlap_Tracking
            fprintf('[DEBUG] Activated column at (%d,%d), total active=%d\n', r, c, nnz(active_columns));
        end
        
        if nnz(active_columns) >= k
            break;
        end
    end
        end
        
    state.refractory_map = max(0, state.refractory_map-1);
    state.activity_map = 0.99 * state.activity_map + 0.01 * double(active_columns);

if cfg.DEBUG && mod(sample_counter, cfg.DEBUG_INTERVAL) == 0
     fprintf('[DEBUG] Final active columns: %d/%d\n', nnz(active_columns), numel(active_columns));
end
        if ~isfield(state,'dead_counter')
    state.dead_counter = zeros(size(overlap));
        end

state.dead_counter = state.dead_counter + ~active_columns;

if mod(sample_counter,300)==0
    dead_cols = state.dead_counter > 100;
    if any(dead_cols(:))
        fprintf('[BOOST] Reactivating %d dead columns.\n', nnz(dead_cols));
        overlap(dead_cols) = overlap(dead_cols) + 0.1*rand();
        state.refractory_map(dead_cols) = 0;
        state.dead_counter(dead_cols) = 0;
    end
end
% Structural rewiring for persistently dead columns (separate threshold)
if mod(sample_counter, 300) == 0
    rewire_cols = state.dead_counter > 500;
    if any(rewire_cols(:))
        fprintf('[REWIRE] Rewiring %d persistently dead columns\n', nnz(rewire_cols));
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
    % FALLBACK if too few
    if nnz(active_columns)<min_active
        active_before_fallback = nnz(active_columns);
        nf = round(min_active * cfg.KWTA_FALLBACK_FACTOR);
        nz = find(overlap(:)>0);
        if isempty(nz)
            sel = randperm(numel(overlap),nf);
        else
            [~,ord] = sort(overlap(nz),'descend');
            sel = nz(ord(1:min(nf,numel(ord))));
        end
        active_columns(sel)=true;
        if cfg.fallback_Debug
        fprintf('[WARNING] Fallback @ iter %d: %d after %d active (min=%d)\n', sample_counter, active_before_fallback, nnz(active_columns), min_active);
        end
        base_noise = base_noise + 0.05;
        state.fallback_counter = state.fallback_counter + 1;
        if state.fallback_counter > 50 && mod(state.fallback_counter,10)==0
    fprintf('[PATCH] Too many fallbacks, forcing density bump.\n');
    base_area_density = min(base_area_density * 1.05, 0.6);
        end


    end
    margin = 0.1;
    max_val = max(normed(:));
    borderline = (normed > (max_val - margin)) & ~active_columns;
    boost_prob = 0.3;
    active_columns = active_columns | (rand(size(active_columns)) < boost_prob & borderline);
    state.activity_map = 0.99 * state.activity_map + 0.01 * double(active_columns);
    state.duty_cycle_map = 0.99 * state.duty_cycle_map + 0.01 * double(active_columns);
    avg_activity = mean(active_columns(:));
    if use_gpu, active_columns = gpuArray(active_columns); end

    % STOCHASTIC exploration
    if mod(sample_counter,cfg.KWTA_STOCH_INTERVAL)==0 && avg_activity<0.08
        rate = max(cfg.KWTA_STOCH_MIN, cfg.KWTA_STOCH_BASE - avg_activity);
        active_columns = active_columns | (rand(size(active_columns))<rate);
        fprintf('[LOG] Stochastic @ %d\n', sample_counter);
    end

    % DEBUG print
    if cfg.DEBUG && mod(sample_counter,cfg.DEBUG_INTERVAL)==0
        fprintf('[DEBUG] Iter %d | Active=%d | Rad=%d | k=%d | MeanOv=%.4f\n', ...
                sample_counter, nnz(active_columns), final_radius, k, mean(overlap(:)));
    end


    fallback_rate = state.fallback_counter / sample_counter;

        if fallback_rate > 0.6 && sample_counter > 50
    state.reject_flag = true;
    fprintf('[REJECT] Fallback rate too high (%.2f)\n', fallback_rate);
        end
        if state.fallback_counter > 10 && sample_counter < 1000
    base_area_density = min(base_area_density * 1.15, 0.6); % big bump early
        elseif state.fallback_counter > 50

    base_area_density = min(base_area_density * 1.05, 0.6); % gentle bump later
        end

end
