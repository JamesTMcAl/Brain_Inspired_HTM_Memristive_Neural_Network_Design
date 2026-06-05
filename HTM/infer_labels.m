function predicted_labels = infer_labels(test_data, w_permanence, train_data, train_labels, overlap_dimension)
    % infer_labels  Generate SDRs and classify via k-NN (Hamming) with SVM fallback
    
    cfg = sp_config.instance();
    

    validateattributes(train_data, {'numeric'},{'nonempty'}, mfilename, 'train_data', 1);
    validateattributes(test_data,  {'numeric'},{'nonempty'}, mfilename, 'test_data', 2);
    assert(ndims(train_data) <= 4, 'infer_labels:BadInput', 'train_data must have at most 4 dimensions.');
    assert(ndims(test_data)  <= 4, 'infer_labels:BadInput', 'test_data must have at most 4 dimensions.');

    % Convert 4-D (H×W×C×N) to grayscale 3-D (H×W×N)
    if ndims(train_data) == 4
        train_data = squeeze(mean(train_data, 3));
    end
    if ndims(test_data) == 4
        test_data = squeeze(mean(test_data, 3));
    end

    % Ensure 3-D data by adding singleton sample dimension if needed
    if ismatrix(train_data)
        train_data = reshape(train_data, size(train_data,1), size(train_data,2), 1);
    end
    if ismatrix(test_data)
        test_data = reshape(test_data, size(test_data,1), size(test_data,2), 1);
    end
    

    potential_radius = size(w_permanence, 1);
    assert(all(overlap_dimension > 0), 'infer_labels:BadInput', 'Invalid overlap dimensions');
    syn_connected_thresh = 0.5;
    threshold_tracker = struct();
    use_gpu = cfg.USE_GPU;
    batch_size = cfg.OVERLAP_BATCH_SIZE;

    % Determine sample counts for 3-D or 4-D data
    if ndims(train_data) == 3
        num_train = size(train_data, 3);
    else
        num_train = size(train_data, 4);
    end
    if ndims(test_data) == 3
        num_test = size(test_data, 3);
    else
        num_test = size(test_data, 4);
    end
    
    % Preallocate SDR matrices (rows = samples, columns = flattened overlap)
    train_sdrs = false(num_train, prod(overlap_dimension));
    test_sdrs  = false(num_test,  prod(overlap_dimension));

    % Compute SDRs for all training samples
    for i = 1:num_train
        % Computee overlap for sample i (batch of size 1)
        
        [ov, syn_connected_thresh_batch, threshold_tracker] = ...
            compute_overlap(train_data, w_permanence, overlap_dimension, potential_radius, i:i, syn_connected_thresh, false, i, batch_size, threshold_tracker);

        % If compute_overlap returned a 3-D array for this single sample, squeeze to 2-D
        if ndims(ov) == 3
            assert(size(ov,3) == 1, 'infer_labels:TrainOverlapBadDims', 'Expected third dim size = 1 for single sample overlap.');
            ov = ov(:,:,1);
        end
        assert(isequal(size(ov), overlap_dimension), ...
            'infer_labels:TrainOverlapMismatch', ...
            'Overlap size mismatch during training: got %s expected %s.', ...
            mat2str(size(ov)), mat2str(overlap_dimension));

        % Determine threshold and generate binary SDR
        thr = syn_connected_thresh_batch(1) + 0.1 * std(double(ov(:)));
        train_sdrs(i, :) = reshape(ov > thr, 1, []);

        % Update threshold for next iteration
        syn_connected_thresh = syn_connected_thresh_batch(end);
        
    end

    % Compute SDRs for all test samples (using batch processing)
    for b = 1:batch_size:num_test
        idx = b : min(b+batch_size-1, num_test);
        
        if isempty(idx) || idx(1) > size(test_data, 3)
            break; % no more samples
        end
        
        % Compute overlap for the batch of test samples
        [ov_batch, syn_connected_thresh_batch, threshold_tracker] = ...
            compute_overlap(test_data, w_permanence, overlap_dimension, potential_radius, idx, syn_connected_thresh, false, idx(1), batch_size, threshold_tracker);

        % Ensure ov_batch is 3-D (H×W×N_batch)
        if ndims(ov_batch) == 2
            ov_batch = reshape(ov_batch, size(ov_batch,1), size(ov_batch,2), 1);
        end
       
       

        % Process each sample in the batch
        for kk = 1:numel(idx)
            ov = squeeze(ov_batch(:,:,kk));
            thr = syn_connected_thresh_batch(kk) + 0.1 * std(double(ov(:)));
            test_sdrs(idx(kk), :) = reshape(ov > thr, 1, []);
            
        end

        % Update threshold for next batch
        syn_connected_thresh = syn_connected_thresh_batch(end);
    end

    % k-NN classification (Hamming distance, K=3)
    K = 3;
    predicted_labels = zeros(num_test,1);
    assert(length(predicted_labels) == num_test, 'infer_labels:LabelVectorSizeMismatch', 'Label vector size mismatch');
    try
        for j = 1:num_test
            % Hamming distance between test SDR j and all train SDRs
            diffs = sum(abs(train_sdrs - reshape(test_sdrs(j,:), 1, [])), 2);
            [~, sorted_idx] = sort(diffs); idxK = sorted_idx(1:K);
            % majority vote among nearest neighbors
            if isempty(idxK)
                predicted_labels(j) = mode(train_labels);  % fallback if no neighbors
            else
                votes = train_labels(idxK);
                predicted_labels(j) = mode(votes);
            end
            
        end
    catch ME
        % fallback to SVM if k-NN fails
        fprintf('[WARNING] k-NN classification failed: %s\n Falling back to SVM.\n', ME.message);
        predicted_labels = classify_using_sdrs(test_sdrs, train_sdrs, train_labels);
    end

    
end
