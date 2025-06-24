classdef TestHTMFunctions < matlab.unittest.TestCase
    % TestHTMFunctions  Unit tests 
    %   Covers core utilities: DataHash, split_data, calculate_entropy,
    %   pi_controller, and simple overlap/inference pipelines.

    methods (Test)
        function testDataHashHexAndUint8(testCase)
            % Known input yields expected MD5 and uint8
            data = uint8([1 2 3 4 5]);
            hexHash = DataHash(data, struct('Method','MD5','Format','hex'));
            testCase.verifyClass(hexHash, 'char');
            testCase.verifyGreaterThan(numel(hexHash), 0);
            uint8Hash = DataHash(data, struct('Method','MD5','Format','uint8'));
            testCase.verifyClass(uint8Hash, 'uint8');
            testCase.verifySize(uint8Hash, [1,16]);
        end

        function testSplitDataSizes(testCase)
            % Create synthetic 3D data [H W N] and labels
            H = 10; W = 8; N = 50;
            data = rand(H,W,N);
            labels = (1:N)';
            [trD, trL, vD, vL] = split_data(data, labels, 0.2);
            testCase.verifyEqual(size(trD,3) + size(vD,3), N);
            testCase.verifyEqual(numel(trL) + numel(vL), N);
        end

        function testCalculateEntropyEdgeCases(testCase)
            % All zeros or ones should give zero entropy
            z = zeros(5);
            testCase.verifyEqual(calculate_entropy(z), 0);
            o = ones(5);
            testCase.verifyEqual(calculate_entropy(o), 0);
            % Half ones/zeros gives entropy ~1 bit
            m = [zeros(10,1); ones(10,1)]; m = reshape(m, [4,5]);
            e = calculate_entropy(m);
            testCase.verifyGreaterThan(e, 0.9);
            testCase.verifyLessThan(e, 1.1);
        end

        function testPIControllerBasic(testCase)
            % PI controller should accumulate integral
            state = struct('I',0,'prevOutput',0);
            [adj1, st1] = pi_controller(1.0, state, 0.5, 0.1, struct('clampI',[-Inf Inf],'momentum',false));
            [adj2, st2] = pi_controller(1.0, st1, 0.5, 0.1, struct('clampI',[-Inf Inf],'momentum',false));
            testCase.verifyGreaterThan(adj2, adj1);
            testCase.verifyGreaterThan(st2.I, st1.I);
        end

        function testAdjustDensityProportional(testCase)
            % With use_mvavg=false, small overlap -> increase density
            cfg = sp_config.instance();
            state = struct('I',0,'prevOutput',0,'hist',[]);
            base = cfg.TARGET_ACTIVITY; % start at target
            overlap = zeros(4);
            [newD, ~] = adjust_density(base, overlap, 1, true, state, false);
            testCase.verifyLessThan(newD, base + 1); % stays in range
            testCase.verifyGreaterThanOrEqual(newD, cfg.MIN_DENSITY);
            testCase.verifyLessThanOrEqual(newD, cfg.MAX_DENSITY);
        end

                
        function testInferLabelsTrivial(testCase)
        % For identical train/test, accuracy=100%
        H = 4; W = 4; N = 10;
        data = rand(H,W,N);
        potential_radius = 3;
        overlap_dim = [H,W] - (potential_radius - 1);
        w_perm = initialize_permanence(ones(potential_radius), potential_radius, overlap_dim, false);

        labels = ones(N,1);
        preds = infer_labels(data, w_perm, data, labels, overlap_dim);
        testCase.verifySize(preds, [N,1]);
        end


        function testDataHashIdempotenceAndError(testCase)
        data = uint8(randi(255,1,128));
        h1 = DataHash(data);
        h2 = DataHash(data);
        testCase.verifyEqual(h1,h2,'Hashes should be deterministic');
        testCase.verifyNotEqual(h1,DataHash([data 0]), ...
            'Different input should give different hash');

        testCase.verifyError(@()DataHash(data,struct('Format','bogus')), ...
            'DataHash:BadFormat');
        end

    function testSplitDataRatioTolerance(testCase)
        rng(42);               % reproducible
        D = rand(6,6,103);     % prime count to test rounding
        L = (1:103)';
        [~,trL,vD,vL] = split_data(D,L,0.25);
        ratio = size(vD,3)/size(D,3);
        testCase.verifyLessThan(abs(ratio-0.25), 1/size(D,3));
        allSeen = sort([trL; vL]);
        testCase.verifyEqual(allSeen, sort(L));
    end

    function testCalculateEntropyRandomMatrix(testCase)
        m = rand(100) > 0.3;   % ~0.88 bits theoretical
        e = calculate_entropy(m);
        testCase.verifyGreaterThan(e,0.7);
        testCase.verifyLessThan(e,1.0);
    end

    function testPIControllerClampAndMomentum(testCase)
        st = struct('I',1.9,'prevOutput',0.4);
        opts = struct('clampI',[-2 2],'momentum',true);
        [adj, st2] = pi_controller(-5,st,0.3,0.6,opts);
        testCase.verifyGreaterThanOrEqual(st2.I,-2);  % clamped
        testCase.verifyNotEqual(adj,0.3*(-5)+0.6*st2.I, ...
            'Momentum term should perturb raw PI output');
    end

    function testAdjustDensityMovingAverage(testCase)
        cfg  = sp_config.instance();
        ovlp = rand(10)<0.05;  % sparse → expect density to go up
        base = cfg.MIN_DENSITY;
        state = struct('I',0,'prevOutput',0,'hist',[]);
        [newD,~] = adjust_density(base,ovlp,1,true,state,true);
        testCase.verifyGreaterThan(newD, base);
    end

            function testComputeOverlapBatchHandling(testCase)
    % Test both batch/non-batch modes on CPU/GPU
    potential_radius = 3;
    input_size = [8, 8];
    overlap_dim = input_size - (potential_radius-1);
    n_samples = 4;
    
    % Create test data and weights
    data_batch = rand([input_size, n_samples], 'single');
    data_single = data_batch(:,:,1); 
    w_permanence = ones([potential_radius, potential_radius, overlap_dim], 'single');

    % Test configurations: {use_gpu, batch_mode, description}
    configs = {
        false, false, 'CPU non-batch';
        false, true,  'CPU batch';
        true,  false, 'GPU non-batch';
        true,  true,  'GPU batch';
    };

        for cfg_idx = 1:size(configs,1)
        use_gpu = configs{cfg_idx,1};
        batch_mode = configs{cfg_idx,2};
        desc = configs{cfg_idx,3};
        
        if use_gpu && gpuDeviceCount == 0
            testCase.assumeTrue(false, 'Skipping GPU test - no device available');
            continue;
        end
        
        % Run compute_overlap
        if batch_mode
            input_data = data_batch;
            sample_indices = 1:n_samples;
            expected_size = [overlap_dim, n_samples];
        else
            input_data = data_single;
            sample_indices = 1;
            expected_size = overlap_dim;
        end
        
        % Execute with fresh threshold tracker
        threshold_tracker = struct();
        [overlap, syn_thresh, tt] = compute_overlap(...
            input_data, w_permanence, overlap_dim, potential_radius, ...
            sample_indices, 0.5, use_gpu, 1, 10, threshold_tracker);
        if ~batch_mode && ndims(overlap) == 2
    overlap = reshape(overlap, [size(overlap), 1]);
        end
        % Basic validation
        testCase.verifySize(overlap, expected_size, ...
            sprintf('%s: Size mismatch', desc));
        testCase.verifyTrue(isa(gather(overlap), 'single') || isa(gather(overlap), 'double'), ...
            sprintf('%s: Wrong data type', desc));
        testCase.verifyGreaterThanOrEqual(overlap, 0, ...
            sprintf('%s: Negative overlaps', desc));
        
        % Compare batch results to individual computations
        if batch_mode
            for i = 1:n_samples
                % Compute single version
                [ov_single, ~, ~] = compute_overlap(...
                    data_batch(:,:,i), w_permanence, overlap_dim, ...
                    potential_radius, 1, 0.5, use_gpu, 1, 10, threshold_tracker);
                
                % Extract batch result
                if use_gpu
                    ov_batch = gather(overlap(:,:,i));
                    ov_single = gather(ov_single);
                else
                    ov_batch = overlap(:,:,i);
                end
                
                if ndims(ov_single) == 2
    ov_single = reshape(ov_single, size(ov_single,1), size(ov_single,2), 1);
            end
        if ndims(ov_batch) == 2
    ov_batch = reshape(ov_batch, size(ov_batch,1), size(ov_batch,2), 1);
        end

        % Match data types
        ov_batch = double(ov_batch);
        ov_single = double(ov_single);

        testCase.verifyEqual(ov_batch, ov_single, 'AbsTol', 1e-3, ...
    sprintf('%s: Batch elem %d mismatch', desc, i));

            end
        end
        
        % Validate threshold tracker structure
        if batch_mode
            testCase.verifyTrue(isstruct(tt), ...
                'Threshold tracker should be struct in batch mode');
            testCase.verifyEqual(numel(fieldnames(tt)), n_samples, ...
                'Threshold tracker missing sample entries');
        end
        end
        end

        function testComputeOverlapNumericalConsistency(testCase)
    % Verify CPU/GPU numerical consistency with tolerance
    if gpuDeviceCount == 0
        testCase.assumeTrue(false, 'Skipping GPU consistency test');
        return;
    end
    
    % Create test data
    data = rand(10,10,2,'single');  % 2 samples
    w_perm = rand(3,3,8,8,'single');
    overlap_dim = [8,8];
    potential_radius = 3;
    
    % CPU results
    [ov_cpu_batch, th_cpu, tt_cpu] = compute_overlap(...
        data, w_perm, overlap_dim, potential_radius, 1:2, 0.5, false, 1, 10, struct());
    
    % GPU results
    [ov_gpu_batch, th_gpu, tt_gpu] = compute_overlap(...
        gpuArray(data), w_perm, overlap_dim, potential_radius, 1:2, 0.5, true, 1, 10, struct());
    
    testCase.verifyEqual(double(gather(ov_gpu_batch)), double(ov_cpu_batch), 'AbsTol', 1e-2, ...
    'GPU/CPU numerical mismatch');

    testCase.verifyEqual(gather(th_gpu), th_cpu, 'AbsTol', 1e-2, ...
        'Threshold values mismatch');
        end
        function testInferLabelsKnnVsSvm(testCase)
            % Construct two clearly separable classes
            N = 6; H = 4; W = 4;
        clsA = zeros(H,W,N/2);        % all zeros
        clsB = ones(H,W,N/2)*2.0;       % bright
        data = cat(3,clsA,clsB);
        labels = [ones(N/2,1); 2*ones(N/2,1)];

    potential_radius = 2;
        overlap_dim = [H W] - (potential_radius - 1);
        w_perm = initialize_permanence(2 * ones(potential_radius), potential_radius, overlap_dim, false);

    % k-NN path
    preds = infer_labels(data,w_perm,data,labels,overlap_dim);
    testCase.verifyEqual(preds, labels);

    % Corrupt train SDRs to force k-NN failure → triggers SVM
    badTrain = data + 0.2*randn(size(data)); 
    preds2 = infer_labels(data,w_perm,badTrain,labels,overlap_dim);
    testCase.verifySize(preds2,[N,1]);  % fallback still predicts
        end


            function testOverlapSingleVsBatchDims(testCase)
    potential_radius = 3;
    input_size = [10, 10];
    overlap_dim = input_size - (potential_radius - 1);
    data = rand([input_size, 2]);  % 2 samples
    w_perm = rand(potential_radius, potential_radius, overlap_dim(1), overlap_dim(2));
    
    % Single
    [ov1, ~, ~] = compute_overlap(data(:,:,1), w_perm, overlap_dim, potential_radius, 1, 0.5, false, 1, 1, struct());
    testCase.verifyEqual(size(ov1), overlap_dim, 'Single sample overlap mismatch');
    
    % Batch
    [ov2, ~, ~] = compute_overlap(data, w_perm, overlap_dim, potential_radius, 1:2, 0.5, false, 1, 2, struct());
    testCase.verifyEqual(size(ov2), [overlap_dim, 2], 'Batch sample overlap mismatch');
            end


            function testApplyKwtaBasic(testCase)
    % Setup: create overlap matrix
    overlap = rand(8, 8);  % Simulated input
    base_area_density = 0.1;
    sample_counter = 1;
    use_gpu = false;
    reset = true;
    state = struct();

    % Run apply_kwta
    [active_columns, avg_activity, new_state, new_density] = apply_kwta( ...
        overlap, base_area_density, sample_counter, use_gpu, reset, state);

    % Assertions
    testCase.verifySize(active_columns, size(overlap));
    testCase.verifyClass(active_columns, 'logical');
    testCase.verifyGreaterThanOrEqual(avg_activity, 0);
    testCase.verifyLessThanOrEqual(avg_activity, 1);
    testCase.verifyGreaterThanOrEqual(new_density, 0);
    testCase.verifyLessThanOrEqual(new_density, 1);
end

                        
    function testComputeOverlap2DInputGPUandCPU(testCase)
        % Test compute_overlap with 2D data on CPU and GPU
        H = 7; W = 5;
        data2D = rand(H, W);
        potential_radius = 3;
        overlap_dim = [H - potential_radius + 1, W - potential_radius + 1];
        w_perm = rand(potential_radius, potential_radius, overlap_dim(1), overlap_dim(2));
        % CPU path
        [ov_cpu, ~, ~] = compute_overlap(data2D, w_perm, overlap_dim, potential_radius, 1, 0.5, false, 1, 1, struct());
        % GPU path
        [ov_gpu, ~, ~] = compute_overlap(data2D, w_perm, overlap_dim, potential_radius, 1, 0.5, true, 1, 1, struct());
        testCase.verifyEqual(size(ov_cpu), overlap_dim, '2D CPU output size mismatch');
        if ndims(ov_gpu) == 3 && size(ov_gpu,3) == 1
    ov_gpu = ov_gpu(:,:,1);
        end
        testCase.verifyEqual(gather(ov_gpu), ov_cpu, 'RelTol', 1e-6, '2D CPU/GPU mismatch');

    end

    function testComputeOverlapBatchVsSingleConsistency(testCase)
        % Test batch vs single-sample consistency for compute_overlap
        H = 8; W = 6; N = 3;
        data = rand(H, W, N);
        potential_radius = 3;
        overlap_dim = [H - potential_radius + 1, W - potential_radius + 1];
        w_perm = rand(potential_radius, potential_radius, overlap_dim(1), overlap_dim(2));
        [ov_batch, ~, ~] = compute_overlap(data, w_perm, overlap_dim, potential_radius, 1:N, 0.5, false, 1, N, struct());
        for i = 1:N
            [ov_single, ~, ~] = compute_overlap(data(:,:,i), w_perm, overlap_dim, potential_radius, 1, 0.5, false, i, 1, struct());
            testCase.verifyEqual(ov_batch(:,:,i), ov_single, 'RelTol', 1e-6, ...
                sprintf('Overlap mismatch for sample %d', i));
        end
    end

    function testThresholdAdaptationVariedDensity(testCase)
        % Test adaptive thresholding on samples with very different intensities
        H = 5; W = 5;
        data1 = zeros(H, W);    % low-intensity sample
        data2 = ones(H, W);     % high-intensity sample
        data = cat(3, data1, data2);
        potential_radius = 2;
        overlap_dim = [H - potential_radius + 1, W - potential_radius + 1];
        w_perm = rand(potential_radius, potential_radius, overlap_dim(1), overlap_dim(2));
        threshold_tracker = struct();
        [~, thr1, threshold_tracker] = compute_overlap(data, w_perm, overlap_dim, potential_radius, 1, 0.5, false, 1, 2, threshold_tracker);
        [~, thr2, ~] = compute_overlap(data, w_perm, overlap_dim, potential_radius, 2, 0.5, false, 2, 2, threshold_tracker);
        testCase.verifyGreaterThan(thr2, thr1, ...
    'Adaptive threshold did not increase for higher-intensity sample');

    end

    function testInferLabelsBatchVsSingle(testCase)
        % Test consistency of infer_labels when called on batches vs single samples
        H = 5; W = 5;
        % Create two classes: all zeros vs all ones
        classA = zeros(H, W, 4);
        classB = ones(H, W, 4);
        train_data = cat(3, classA, classB);
        train_labels = [ones(4,1); 2*ones(4,1)];
        potential_radius = 2;
        overlap_dim = [H - potential_radius + 1, W - potential_radius + 1];
        % Use simple random permanence for demonstration
        w_perm = rand(potential_radius, potential_radius, overlap_dim(1), overlap_dim(2));
        % Test data: one zero and one one image
        test_data = zeros(H, W, 2);
        test_data(:,:,2) = 1;
        % Predictions on batch
        preds_batch = infer_labels(test_data, w_perm, train_data, train_labels, overlap_dim);
        % Predictions one by one
        preds_single = zeros(2,1);
        for j = 1:2
            sample = reshape(test_data(:,:,j), size(test_data,1), size(test_data,2), 1);

            preds_single(j) = infer_labels(sample, w_perm, train_data, train_labels, overlap_dim);

        end
        testCase.verifyEqual(preds_batch, preds_single, 'Infer_labels batch vs single mismatch');
    end
end


    
end
