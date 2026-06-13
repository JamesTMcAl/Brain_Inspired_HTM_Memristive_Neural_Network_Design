function [base_area_density, state] = adjust_density(base_area_density, overlap, sample_counter, reset, state, use_mvavg)
%ADJUST_DENSITY  PI-controller or moving-average for dynamic sparsity.
%   Measures the fraction of columns with any overlap (not raw overlap magnitude) so the error signal is in the same units as TARGET_ACTIVITY.
%   Self-adaptive target: shifts toward current activity when entropy is healthy and stable, loosens when entropy is collapsing. This lets the system find its own sparsity equilibrium rather than chasing a fixed biological ideal that may not suit the input statistics.


    if nargin < 4, reset     = false; end
    if nargin < 5, state     = struct('I', 0, 'prevOutput', 0, 'hist', [], ...
                                      'prev_error', 0, 'entropy_hist', [], ...
                                      'target_activity', []); end
    if nargin < 6, use_mvavg = false; end

    % Reset on request or first iteration
    if reset || sample_counter == 1
        state = struct('I', 0, 'prevOutput', 0, 'hist', [], ...
                       'prev_error', 0, 'entropy_hist', [], ...
                       'target_activity', []);
    end

    % Backwards-compatibility: ensure all fields exist on old state structs
    if ~isfield(state, 'prev_error'),      state.prev_error     = 0;  end
    if ~isfield(state, 'entropy_hist'),    state.entropy_hist   = []; end
    if ~isfield(state, 'target_activity'), state.target_activity = []; end

    cfg = sp_config.instance();

    % Initialise adaptive target from config on first call
    if isempty(state.target_activity)
        state.target_activity = cfg.TARGET_ACTIVITY;
    end


    % Entropy-driven adaptive target update

    if numel(state.entropy_hist) >= 10
        recent_entropy = mean(state.entropy_hist(end-9:end));
        entropy_trend  = state.entropy_hist(end) - state.entropy_hist(max(1, end-9));

        if recent_entropy > 0.4 && entropy_trend >= 0
            % healthy and stable/improving: pull target toward current activity
            current_act = mean(overlap(:) > 0);
	    % adaptive rate scales with PI call frequency
            % more calls per epoch = slower rate per call
            % fewer calls = faster rate so it converges within the epoch
            if ~isfield(state, 'adapt_rate'), state.adapt_rate = 0.20; end
            if numel(state.entropy_hist) == 10
                % Use faster rate for small datasets, slower for large
                estimated_calls = max(1, round(sample_counter / 10));
                state.adapt_rate = min(0.20, max(0.02, 2.0 / estimated_calls));
            end
            state.target_activity = (1 - state.adapt_rate) * state.target_activity +  state.adapt_rate * current_act;
        elseif recent_entropy < 0.15 || recent_entropy < 0.5 * cfg.ENTROPY_THRESHOLD_INIT
            % Collapsing or entropy below half its healthy initial level, loosen target aggressively to pull activity back up
            state.target_activity = min(state.target_activity * 1.10, 0.45);
        end
        state.target_activity = max(state.target_activity, cfg.MIN_DENSITY * 2);
        state.target_activity = min(max(state.target_activity, cfg.MIN_DENSITY), cfg.MAX_DENSITY);
    end

    effective_target = state.target_activity;


    % Moving-average mode (proportional only)

    if use_mvavg
        state.hist = [state.hist, mean(overlap(:) > 0)];
        if numel(state.hist) > 100
            state.hist = state.hist(end-99:end);
        end
        current_activity = mean(state.hist);
        error_val        = effective_target - current_activity;
        adjustment       = cfg.KP_DENSITY * error_val;


    % PI mode (default)

    else
        current_activity = mean(overlap(:) > 0);
        error_val        = effective_target - current_activity;

        % Integral windup guard: halve integral when error flips sign
        if sign(error_val) ~= sign(state.prev_error) && sample_counter > 100
            state.I = state.I * 0.5;
        end
        state.prev_error = error_val;

        % Adaptive PI gains
        std_o = std(overlap(:));
        if sample_counter < 2000
            Kp_boost = 2.0;
            Ki_boost = 2.0;
        else
            Kp_boost = 1.0;
            Ki_boost = 1.0;
        end
        Kp = min(max(cfg.KP_DENSITY * (1 + 0.5*std_o) * Kp_boost, 0.01), 0.4);
        Ki = min(max(cfg.KI_DENSITY * (1 + 0.2*std_o) * Ki_boost, 0.005), 0.1);

        [adjustment, state] = pi_controller(error_val, state, Kp, Ki, ...
                                            'clampI', [-2, 2], 'momentum', true);
    end


    % Proportional step size

    error_magnitude = abs(error_val);
    if error_magnitude > 0.2
        alpha = 0.015;
    elseif error_magnitude > 0.05
        alpha = 0.005;
    else
        alpha = 0.001;
    end

    base_area_density = base_area_density + alpha * adjustment;
    base_area_density = min(max(base_area_density, cfg.MIN_DENSITY), cfg.MAX_DENSITY);

    if cfg.DEBUG && cfg.DEBUG_ADJUST && mod(sample_counter, 100) == 0
        fprintf('[DENSITY] Iter %d | Activity=%.4f | Target=%.4f | Density=%.4f | Adj=%.4f\n', ...
                sample_counter, current_activity, effective_target, ...
                base_area_density, adjustment);
    end
end
