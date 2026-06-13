function [active_cells, winner_cells, predicted_cells, tm_state] = temporal_memory(...
    active_columns, tm_state, learn, sample_counter)
%TEMPORAL_MEMORY HTM Temporal Memory - sequence learning extension of EA-MHTM
% Adds predictive cell firing to the spatial pooler output.
% Local Hebbian learning only - no backpropagation.

    cfg = sp_config.instance();

    if ~isfield(tm_state, 'initialised')
        tm_state = tm_init(size(active_columns), cfg);
    end

    [H, W] = size(active_columns);
    C      = tm_state.cells_per_col;

    active_cells    = false(H, W, C);
    winner_cells    = false(H, W, C);
    predicted_cells = false(H, W, C);

    % Phase 1 - activate cells in active columns
    bursting_cols = false(H, W);
    for col_idx = find(active_columns)'
        [r, c] = ind2sub([H, W], col_idx);
        predicted_in_col = squeeze(tm_state.predicted_state(r, c, :));

        if any(predicted_in_col)
            active_cells(r, c, :) = predicted_in_col;
            winner_cells(r, c, :) = predicted_in_col;
        else
            % burst - no prediction was made for this column
            active_cells(r, c, :)  = true;
            bursting_cols(r, c)    = true;
            best_cell = tm_find_best_cell(r, c, tm_state);
            winner_cells(r, c, best_cell) = true;
        end
    end

    % Phase 2 - synaptic learning
    if learn
        tm_state = tm_learn(active_cells, winner_cells, tm_state, sample_counter, cfg);
    end

    % Phase 3 - predict next timestep
    [predicted_cells, tm_state] = tm_predict(active_cells, tm_state, cfg);

    % Phase 4 - anomaly score (unpredicted active columns)
    n_active = sum(active_columns(:));
    n_correct = sum(sum(any(active_cells & repmat(tm_state.predicted_state, 1, 1, 1), 3) & active_columns));
    tm_state.anomaly_score = 1 - (n_correct / max(n_active, 1));
	if tm_state.anomaly_score > 0.5
        tm_state.activation_thresh = max(2, tm_state.activation_thresh - 0.01);
    elseif tm_state.anomaly_score < 0.15
        tm_state.activation_thresh = min(6, tm_state.activation_thresh + 0.005);
    end
    tm_state.anomaly_history = [tm_state.anomaly_history, tm_state.anomaly_score];
    if numel(tm_state.anomaly_history) > 100
        tm_state.anomaly_history = tm_state.anomaly_history(end-99:end);
    end

    tm_state.predicted_state = predicted_cells;
    tm_state.prev_active     = active_cells;
    tm_state.prev_winner     = winner_cells;

end

function state = tm_init(col_size, cfg)
% Initialise TM state

    C = 32;   % cells per column
    H = col_size(1);
    W = col_size(2);
    n_cols = H * W;

    state.cells_per_col    = C;
    state.col_size         = col_size;

    % Distal synapses - each cell has max_segs segments
    % each segment has max_syns potential synapses
    max_segs = 128;
    max_syns = 32;

    state.seg_permanences  = 0.1 * rand(n_cols, C, max_segs, max_syns);
    state.seg_presynaptic  = zeros(n_cols, C, max_segs, max_syns, 'uint32');
    state.seg_count        = zeros(n_cols, C, 'uint8');

    % Activation threshold - adaptive like SP density controller
    state.activation_thresh     = 3;
    state.learning_thresh       = 2;
    state.initial_permanence    = 0.51;
    state.connected_permanence  = 0.50;
    state.perm_inc              = 0.10;
    state.perm_dec              = 0.05;

    % Memristor endurance tracking (same pattern as SP)
    state.write_counts     = zeros(n_cols, C, max_segs, max_syns, 'uint32');
    state.endurance_limit  = 1e5;

    % Persistent state
    state.predicted_state  = false(H, W, C);
    state.prev_active      = false(H, W, C);
    state.prev_winner      = false(H, W, C);

    % Anomaly tracking
    state.anomaly_score    = 0;
    state.anomaly_history  = [];

    state.initialised      = true;
    fprintf('[TM] Initialised: %dx%d grid, %d cells/col, %d segs/cell\n', ...
            H, W, C, max_segs);
end


function best_cell = tm_find_best_cell(r, c, state)
% Find best matching cell, or least-used cell if no match exists
% Standard HTM rule: least-used-cell prevents all context collapsing onto cell 1

    col_idx = sub2ind(state.col_size, r, c);
    C = state.cells_per_col;

    best_cell     = 1;
    best_score    = -1;
    found_match   = false;

    prev_flat = state.prev_active(:);

    % First pass: find best matching segment across all cells
    for cell = 1:C
        n_segs = state.seg_count(col_idx, cell);
        for seg = 1:n_segs
            pre  = squeeze(state.seg_presynaptic(col_idx, cell, seg, :));
            perm = squeeze(state.seg_permanences(col_idx, cell, seg, :));
            valid = pre > 0;
            if ~any(valid), continue; end
            connected = perm(valid) >= state.connected_permanence;
            pre_valid = pre(valid);
            score = sum(prev_flat(pre_valid(connected)));
            if score > best_score
                best_score = score;
                best_cell  = cell;
                found_match = (score > 0);
            end
        end
    end


        if ~found_match
        % least used cell with random tie break critical for variable order memory. Without the tie-break all early context collapses onto cell 1 since every cell starts with 0 segments.
        seg_counts = double(state.seg_count(col_idx, :));   % 1xC
        candidates = find(seg_counts == min(seg_counts));
        best_cell  = candidates(randi(numel(candidates)));
    end
end


function state = tm_learn(active_cells, winner_cells, state, sample_counter, cfg)
% Hebbian learning - strengthen synapses from prev active to current winner

    [H, W, C]  = size(active_cells);
    prev_flat  = state.prev_winner(:);
    n_prev     = sum(prev_flat);

    if n_prev == 0, return; end
    prev_indices = find(prev_flat);

    for col_idx = 1:H*W
        [r, c] = ind2sub([H, W], col_idx);
        for cell = 1:C
            if ~winner_cells(r, c, cell), continue; end

            n_segs = state.seg_count(col_idx, cell);
            found_seg = 0;

            % Find matching segment
            for seg = 1:n_segs
                pre  = squeeze(state.seg_presynaptic(col_idx, cell, seg, :));
                perm = squeeze(state.seg_permanences(col_idx, cell, seg, :));
                valid = pre > 0;
                if ~any(valid), continue; end
                score = sum(prev_flat(pre(valid)));
                if score >= state.learning_thresh
                    found_seg = seg;
                    break;
                end
            end

            % Grow new segment if none found
            if found_seg == 0
                max_segs = size(state.seg_permanences, 3);
                if n_segs >= max_segs
                    % Prune lowest scoring segment to make room
                    seg_scores = zeros(max_segs, 1);
                    for s = 1:max_segs
                        pre_s = squeeze(state.seg_presynaptic(col_idx, cell, s, :));
                        perm_s = squeeze(state.seg_permanences(col_idx, cell, s, :));
                        valid_s = pre_s > 0;
                        if any(valid_s)
                            seg_scores(s) = mean(perm_s(valid_s));
                        end
                    end
                    [~, worst] = min(seg_scores);
                    state.seg_permanences(col_idx, cell, worst, :) = 0;
                    state.seg_presynaptic(col_idx, cell, worst, :) = 0;
                    state.seg_count(col_idx, cell) = max_segs - 1;
                    n_segs = max_segs - 1;
                end
                if n_segs < max_segs
                    n_segs = n_segs + 1;
                    state.seg_count(col_idx, cell) = n_segs;
                    found_seg = n_segs;

                    % Sample synapses from prev winners
                    max_syns = size(state.seg_permanences, 4);
                    n_new = min(max_syns, numel(prev_indices));
                    chosen = prev_indices(randperm(numel(prev_indices), n_new));
                    state.seg_presynaptic(col_idx, cell, found_seg, 1:n_new) = chosen;
                    state.seg_permanences(col_idx, cell, found_seg, 1:n_new) = state.initial_permanence;
                end
            end

            if found_seg == 0, continue; end

            % LTP/LTD update with memristor endurance check
            pre  = squeeze(state.seg_presynaptic(col_idx, cell, found_seg, :));
            perm = squeeze(state.seg_permanences(col_idx, cell, found_seg, :));
            wc   = squeeze(state.write_counts(col_idx,   cell, found_seg, :));

            valid = pre > 0;
            for syn = find(valid)'
                if wc(syn) >= state.endurance_limit, continue; end
                if prev_flat(pre(syn))
                    perm(syn) = min(1.0, perm(syn) + state.perm_inc);
                else
                    perm(syn) = max(0.0, perm(syn) - state.perm_dec);
                end
                wc(syn) = wc(syn) + 1;
            end

            state.seg_permanences(col_idx, cell, found_seg, :) = perm;
            state.write_counts(col_idx,   cell, found_seg, :) = wc;
        end
    end
end


function [predicted, state] = tm_predict(active_cells, state, cfg)
% Compute predicted cells for next timestep based on current active cells

    [H, W, C] = size(active_cells);
    active_flat = active_cells(:);
    predicted   = false(H, W, C);

    for col_idx = 1:H*W
        [r, c] = ind2sub([H, W], col_idx);
        for cell = 1:C
            n_segs = state.seg_count(col_idx, cell);
            for seg = 1:n_segs
                pre  = squeeze(state.seg_presynaptic(col_idx, cell, seg, :));
                perm = squeeze(state.seg_permanences(col_idx, cell, seg, :));
                valid = pre > 0;
                if ~any(valid), continue; end
                connected = perm(valid) >= state.connected_permanence;
                pre_valid = pre(valid);
                score = sum(active_flat(pre_valid(connected)));
                if score >= state.activation_thresh
                    predicted(r, c, cell) = true;
                    break;
                end
            end
        end
    end




end


