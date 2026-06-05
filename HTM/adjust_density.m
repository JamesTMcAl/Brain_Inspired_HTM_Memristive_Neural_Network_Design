function [base_area_density, state] = adjust_density( base_area_density, overlap, sample_counter, reset, state, use_mvavg)
%ADJUST_DENSITY  PI‑controller or moving‑average for dynamic sparsity
%
% [newDensity, state] = adjust_density(oldDensity, overlap, iter, reset, state, use_mvavg)

    if nargin < 4, reset = false; end
    if nargin < 5, state = struct('I',0,'prevOutput',0,'hist',[]); end
    if nargin < 6, use_mvavg = false; end

    % On reset, initialize integral, previous output, and history buffer
    if (reset == true) || (sample_counter == 1)
        state = struct('I',0,'prevOutput',0,'hist',[]);
    end

    cfg = sp_config.instance();

    if use_mvavg
        % moving‐average over last 100 activities
        state.hist = [state.hist, mean(overlap(:))];
        if numel(state.hist) > 100
            state.hist = state.hist(end-99:end);
        end
        target    = cfg.TARGET_ACTIVITY;
        error_val = target - mean(state.hist);
        % proportional update
        adjustment = cfg.KP_DENSITY * error_val;
    else
        % compute error vs target
        current_activity = mean(overlap(:));
        error_val        = cfg.TARGET_ACTIVITY - current_activity;

        % adaptive PI gains
        std_o = std(overlap(:));
        if sample_counter < 2000
    Kp_boost = 2.0;  % Boost early gain
    Ki_boost = 2.0;
        else
    Kp_boost = 1.0;
    Ki_boost = 1.0;
        end
        Kp  = min(max(cfg.KP_DENSITY*(1+0.5*std_o)*Kp_boost, 0.01), 0.4);
        Ki  = min(max(cfg.KI_DENSITY*(1+0.2*std_o)*Ki_boost, 0.005), 0.1);

        % call shared PI controller
        [adjustment, state] = pi_controller( error_val, state, Kp, Ki, 'clampI',[-2,2], 'momentum', true);
    end

    % apply exponential decay + clamp to valid density range
    alpha = 0.02;
        base_area_density = base_area_density + alpha*adjustment;
    base_area_density = min(max(base_area_density, cfg.MIN_DENSITY), cfg.MAX_DENSITY);

    % debug print
    if cfg.DEBUG && cfg.DEBUG_ADJUST && mod(sample_counter,100)==0
        fprintf('[DENSITY] Iter %d | Density=%.4f | Adj=%.4f\n', ...
                sample_counter, base_area_density, adjustment);
    end
end
