function [overlap, syn_connected_thresh_batch, threshold_tracker] = compute_overlap( ...
    train_data, w_permanence, overlap_dimension, potential_radius, ii, ...
    syn_connected_thresh, use_gpu, sample_counter, batch_size, threshold_tracker)
%COMPUTE_OVERLAP Compute spatial overlaps between input data and permanence weights.
%   This function calculates the overlap score between input image data and
%   a set of permanence weights (w_permanence) for the Spatial Pooler. It
%   supports single or batch modes and optional GPU acceleration. Overlaps
%   are non-negative and adaptively thresholded using quantiles from sp_config.

    % Get configuration instance for parameters and debug flags
    cfg = sp_config.instance();
    DEBUG_INTERVAL = cfg.DEBUG_INTERVAL;
    
    % Validate required inputs
    if nargin < 5
        error('compute_overlap:NotEnoughInputs', 'Insufficient input arguments.');
    end
    validateattributes(overlap_dimension, {'numeric'}, {'vector','numel',2}, mfilename, 'overlap_dimension', 3);
    validateattributes(potential_radius, {'numeric'}, {'scalar','integer','>=',1}, mfilename, 'potential_radius', 4);
    validateattributes(ii, {'numeric'}, {'vector','integer','>=',1}, mfilename, 'ii', 5);
    
    % Determine data dimensionality and extract the requested sample(s)
    d = ndims(train_data);
    if ismatrix(train_data)
        % 2D image (HxW) -> treat as 3D [H x W x 1]
        train_data = reshape(train_data, size(train_data,1), size(train_data,2), 1);
        d = 3;
    end
    
    % Handle GPU flag and availability
    if nargin < 7 || isempty(use_gpu)
        use_gpu = false;
    end
    if use_gpu && gpuDeviceCount == 0
        use_gpu = false;  % fallback if no GPU present
    end
    
    % Extract the specified samples from train_data
    if d == 3
        % 3D data: [H x W x N]
        slice = train_data(:,:,ii);  % can handle scalar or vector ii
    elseif d == 4
        % 4D data: [H x W x C x N] ( multi-channel)
        % Average across channels to get a single [H x W] per sample
        slice = mean(train_data(:,:,:,ii), 3);  % handles scalar or vector ii
    else
        error('compute_overlap:BadInputDims', 'train_data must be 2D, 3D, or 4D.');
    end
    
    % Ensure slice is 3D for consistency (H x W x num_samples)
    if ismatrix(slice)
        slice = reshape(slice, size(slice,1), size(slice,2), 1);
    end
    
    % Get dimensions
    [H, W, num_samples] = size(slice);
    p = potential_radius;
    outH = overlap_dimension(1);
    outW = overlap_dimension(2);
    
    % Check that input size matches expected overlap dimension + patch
    if H ~= outH + p - 1 || W ~= outW + p - 1
        error('compute_overlap:InputSizeMismatch', ...
            'Input size [%d %d] does not match overlap dimension [%d %d] with patch %d.', ...
            H, W, outH, outW, p);
    end
    
    % Validate w_permanence dimensions and range
    if ~isnumeric(w_permanence) || isempty(w_permanence)
        error('compute_overlap:BadWPerm', 'w_permanence must be a non-empty numeric array.');
    end
    if ndims(w_permanence) ~= 4
        error('compute_overlap:BadWPermDims', 'w_permanence must be a 4-D array.');
    end
    [p1, p2, hW, hW2] = size(w_permanence);
    if p1~=p || p2~=p || hW~=outH || hW2~=outW
        error('compute_overlap:WPermSizeMismatch', ...
            'w_permanence size mismatch: expected [%d %d %d %d], got [%d %d %d %d].', ...
            p, p, outH, outW, p1, p2, hW, hW2);
    end
    % Clip permanence values to [0,1] valid range
    w_permanence = min(max(w_permanence, 0), 1);
    
    % If using GPU, move data to GPU
    if use_gpu
        slice = gpuArray(slice);
        w_permanence = gpuArray(w_permanence);
    end
    
    % Compute overlaps by extracting patches and dot-product with weights
    origType = class(slice);           % remember original data type
    slice_double = double(slice);      % work in double for accuracy
    Wperm = double(w_permanence);  % gather weights to CPU double for computation
    
    if num_samples > 1
        % Batch mode: compute overlap for each sample in loop
        overlap_mat = zeros(outH, outW, num_samples);
        for j = 1:num_samples
            patch_cols = im2col(slice_double(:,:,j), [p, p], 'sliding');  % [p*p x outH*outW]
            Wmat = reshape(Wperm, p*p, []);                               % [p*p x (outH*outW)]
            ovlp = sum(patch_cols .* Wmat, 1);                            % [1 x (outH*outW)]
            overlap_mat(:,:,j) = reshape(ovlp, outH, outW);
        end
    else
        % Single-sample mode
        patch_cols = im2col(slice_double(:,:,1), [p, p], 'sliding');
        Wmat = reshape(Wperm, p*p, []);
        ovlp = sum(patch_cols .* Wmat, 1);
        overlap_mat = reshape(ovlp, outH, outW);
    end
    
    % Cast overlap back to original data type (single or double)
    if isa(w_permanence, 'single') || strcmp(origType,'single')
        overlap_mat = single(overlap_mat);
    end
    if use_gpu
        overlap_mat = gpuArray(overlap_mat);
    end
    
    % Eliminate any NaN or Inf (shouldn't occur normaly)
    overlap_mat(isnan(overlap_mat) | isinf(overlap_mat)) = 0;
    
    % Output the overlap score
    overlap = overlap_mat;
    
    %  Quantile-based Threshold Adaptation
    % Initialize threshold_tracker struct if it was numeric or empty
    if ~isstruct(threshold_tracker)
        if ~isempty(threshold_tracker) && isnumeric(threshold_tracker)
            base_thresh = threshold_tracker;
        elseif exist('syn_connected_thresh','var') && ~isempty(syn_connected_thresh)
            base_thresh = syn_connected_thresh;
        else
            base_thresh = 0;
        end
        threshold_tracker = struct();
        for j = 1:numel(ii)
            threshold_tracker.(['s_' num2str(ii(j))]) = base_thresh;
        end
    end
    
    % Prepare output threshold array
    syn_connected_thresh_batch = zeros(1, numel(ii));
    for k = 1:numel(ii)
        sample_idx = ii(k);
        field = ['s_' num2str(sample_idx)];
        if isfield(threshold_tracker, field) && ~isempty(threshold_tracker.(field))
            prev_thresh = threshold_tracker.(field);
        else
            prev_thresh = 0;
        end
        % Compute new low quantile from current overlap distribution
        if use_gpu
            ov_vals = overlap_mat(:,:,k);
        else
            ov_vals = overlap_mat(:,:,k);
        end
        ov_vals = ov_vals(:);
        q_new = quantile(ov_vals, cfg.OVERLAP_QUANTILE_UPDATE);
        % Exponential moving average update
        new_thresh = cfg.OVERLAP_QUANTILE_DECAY * prev_thresh + (1 - cfg.OVERLAP_QUANTILE_DECAY) * q_new;
        new_thresh = max(new_thresh, 0);  % ensure non-negative threshold
        threshold_tracker.(field) = new_thresh;
        % Synapse-connected threshold is half of overlap threshold
        syn_val = 0.5 * new_thresh;
        % Clamp upper bound to avoid extreme values (no lower bound clamp)
        syn_val = min(syn_val, 0.6);
        syn_connected_thresh_batch(k) = syn_val;
    end
    
    % Return scalar if only one sample
    if numel(syn_connected_thresh_batch) == 1
        syn_connected_thresh_batch = syn_connected_thresh_batch(1);
    end
    
    % If GPU mode, output threshold array as GPU array
    if use_gpu
        syn_connected_thresh_batch = gpuArray(syn_connected_thresh_batch);
        
    end
    
    %  Debug Logging
    if cfg.DEBUG_OVERLAP && mod(sample_counter, DEBUG_INTERVAL) == 0
        ov_all = overlap_mat(:);
        if use_gpu, ov_all = ov_all; end
        fprintf('[DEBUG_OVERLAP] Iter %d | mean=%.4f | max=%.4f | nnz=%d\n', ...
                sample_counter, mean(ov_all), max(ov_all), nnz(ov_all>0));
    end
    if cfg.Debug_Overlap_Tracking
        fprintf('[DEBUG_OVERLAP_TRACK] Sample(s) %s | Thresh fields: %s\n', mat2str(ii), ...
                strjoin(string(struct2cell(threshold_tracker)), ', '));
    end

end
