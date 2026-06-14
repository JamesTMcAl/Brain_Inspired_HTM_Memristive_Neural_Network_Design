function [syn_inc, syn_dec, state] = adjust_synaptic_factors( ...
    active_cols, syn_inc_base, syn_dec_base, boost_factor, ...
    sample_counter, decay_scaling, input_std, state, w_permanence)
%ADJUST_SYNAPTIC_FACTORS Adjust the synaptic increment/decrement factors based on activity.

    cfg = sp_config.instance();

    % Expand active_cols if needed
    if isequal(size(active_cols), size(w_permanence(:,:,1,1)))
        active_cols = repmat(active_cols, [size(w_permanence,1), size(w_permanence,2), 1, 1]);
    end
    assert(isequal(size(active_cols), size(w_permanence)), ...
        'Size mismatch between active_cols and w_permanence');

    % Early exit for no activity
    if isempty(active_cols) || ~any(active_cols(:))
        syn_inc = zeros(size(w_permanence), 'like', w_permanence);
        syn_dec = zeros(size(w_permanence), 'like', w_permanence);
        return;
    end

    % Initialize momentum if needed
    if ~isfield(state, 'velocity_inc')
        state.velocity_inc = 0;
        state.velocity_dec = 0;
    end

    if  cfg.DEBUG && mod(sample_counter, 100) == 0
        fprintf('[DEBUG] sample_counter: %d | active_cols size: %s | w_permanence size: %s\n', ...
                sample_counter, mat2str(size(active_cols)), mat2str(size(w_permanence)));
    end

    % Compute statistics
    original_active   = squeeze(any(any(active_cols,1),2));
    active_ratio      = mean(original_active(:));
    persistence_ratio = mean(w_permanence(:) > 0.5);

    % Momentum update
    tr = cfg.SYN_TARGET_RANGE;
    m  = min(0.95, 0.5 + 0.45*tanh(0.1*sample_counter));
    state.velocity_inc = m*state.velocity_inc + (1-m)*(active_ratio - tr(1));
    state.velocity_dec = 0.9*state.velocity_dec + 0.1*(tr(2) - persistence_ratio);

    % Clamp velocities
    state.velocity_inc = max(-0.3, min(0.3, state.velocity_inc));
    state.velocity_dec = max(-0.3, min(0.3, state.velocity_dec));

    % Homeostatic scaling
    activity_error = tr(1) - mean(active_cols(:));
    Kp_syn = 0.1;
    homeo_scale = 1 + Kp_syn * activity_error;
    if cfg.DEBUG && mod(sample_counter, 100) == 0
        fprintf('[DEBUG] synaptic scaling_factor=%.4f\n', homeo_scale);
    end

    % Base updates
    inc_base = (syn_inc_base + 0.005*input_std) * homeo_scale;
    dec_base = syn_dec_base * homeo_scale;
    epoch_position = mod(sample_counter, decay_scaling);
    decay = 0.3 + 0.7 * exp(-epoch_position / (decay_scaling * 1.5));

    % Compute scalar updates
    scalar_inc = inc_base * (1 + boost_factor * state.velocity_inc) * decay * 1.5;
    scalar_dec = dec_base * (1 + boost_factor * state.velocity_dec) * decay * 0.8;

    % Broadcast scalars safely
    syn_inc = scalar_inc * ones(size(w_permanence), 'like', w_permanence);
    syn_dec = scalar_dec * ones(size(w_permanence), 'like', w_permanence);

    % Clamp to safe operational ranges
    SYN_INC_MIN_ABS = 1e-4;  % prevent zero update
    SYN_INC_MAX_ABS = 1e-1;  % prevent exploding updates
    syn_inc = max(SYN_INC_MIN_ABS, min(SYN_INC_MAX_ABS, syn_inc));

        syn_dec = max(0.001, min(0.02, syn_dec));
end
