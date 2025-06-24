function [sdr_flat, state] = generate_sdr(data, w_permanence, idx, config, state)
    %GENERATE_SDR  Generate sparse distributed representation using adaptive kWTA
    arguments
        data            {mustBeNumeric}
        w_permanence    {mustBeNumeric}
        idx             (1,1) {mustBeInteger}
        config          struct  % overlap_dim, potential_radius, syn_connected_thresh, base_area_density
        state           struct  % sample_counter (int), kwta_state (struct)
    end
    cfg = sp_config.instance();

    % Move data to GPU if enabled
    if cfg.USE_GPU
        data = gpuArray(data);
        w_permanence = gpuArray(w_permanence);
    end

    % Compute overlap

    input_size = [size(data,1), size(data,2)];
    dynamic_overlap_dim = input_size - (config.potential_radius - 1);
        

    if ~isfield(state, 'threshold_tracker') || isempty(state.threshold_tracker)
    state.threshold_tracker = struct();
    end

    [overlap, syn_connected_thresh, state.threshold_tracker] = compute_overlap(data, w_permanence, ...
    dynamic_overlap_dim, config.potential_radius, idx, ...
    config.syn_connected_thresh, cfg.USE_GPU, state.sample_counter, ...
    cfg.OVERLAP_BATCH_SIZE, state.threshold_tracker);

    % Initialize kwta_state on first call if missing
    if ~isfield(state, 'kwta_state') || isempty(state.kwta_state)
        state.kwta_state = struct();
    end

    % Apply adaptive kWTA inhibition
    reset_flag = (state.sample_counter == 1);
    [active_columns, ~, state.kwta_state, base_area_density] = apply_kwta( overlap, config.base_area_density, state.sample_counter, cfg.USE_GPU, reset_flag, state.kwta_state);

    % Flatten SDR
    sdr_flat = active_columns(:);
    state.last_active_columns = active_columns;   
    % Debug print
    if cfg.DEBUG
        fprintf('[SDR] Sample %d | Active=%d/%d\n', idx, nnz(sdr_flat), numel(sdr_flat));
    end

    % Increment sample counter
    state.sample_counter = state.sample_counter + 1;
end
