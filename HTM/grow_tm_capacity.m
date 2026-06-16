function tm_state = grow_tm_capacity(tm_state, cfg)
% GROW_TM_CAPACITY  Add cells per column when anomaly stays high (neurogenesis)
%   Preserves all learned segments; new cells start blank so existing
%   sequence memory is untouched and the network simply gains headroom.
    old_C = tm_state.cells_per_col;
    new_C = min(cfg.EVOLVE_TM_CELLS_MAX, round(old_C * cfg.EVOLVE_GROW_FACTOR));
    if new_C <= old_C
        return;  % already at ceiling
    end

    [n_cols, ~, max_segs, max_syns] = size(tm_state.seg_permanences);
    added = new_C - old_C;

    % Segment arrays
    % flattened layout (n_cols, C, ...), cell dim = 2
    tm_state.seg_permanences = cat(2, tm_state.seg_permanences, ...
        0.1 * rand(n_cols, added, max_segs, max_syns));
    tm_state.seg_presynaptic = cat(2, tm_state.seg_presynaptic, ...
        zeros(n_cols, added, max_segs, max_syns, 'uint32'));
    tm_state.seg_count = cat(2, tm_state.seg_count, ...
        zeros(n_cols, added, 'uint8'));
    if isfield(tm_state, 'write_counts')
        tm_state.write_counts = cat(2, tm_state.write_counts, ...
            zeros(n_cols, added, max_segs, max_syns, 'uint32'));
    end

    %Grid-shaped state arrays
    % (H, W, C), cell dim = 3, These MUST also grow or the next TM call mismatches dimensions
    H = tm_state.col_size(1);
    W = tm_state.col_size(2);
    tm_state.predicted_state = cat(3, tm_state.predicted_state, false(H, W, added));
    tm_state.prev_active     = cat(3, tm_state.prev_active,     false(H, W, added));
    tm_state.prev_winner     = cat(3, tm_state.prev_winner,     false(H, W, added));

    tm_state.cells_per_col = new_C;
    fprintf('[EVOLVE-GROW] TM capacity grown: %d -> %d cells/column\n', old_C, new_C);
end
