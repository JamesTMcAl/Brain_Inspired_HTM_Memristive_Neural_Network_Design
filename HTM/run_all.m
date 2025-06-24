cfg = sp_config.instance();
disp('Starting HTM Spatial Pooler Training...');
% dataset (MNIST/CIFAR)
    dataset = 'CIFAR'; 
    if strcmp(dataset, 'MNIST')
    [train_data, train_label, test_data, test_label] = get_mnist();
    train_data = train_data(:,:,1:cfg.SubsetTraining);
    train_label = train_label(1:cfg.SubsetTraining);
    else
    if ~exist(fullfile('cifar-10-batches-mat', 'cifar-10.mat'), 'file')
        preprocess_cifar10(); 
    end
    [train_data, train_label, test_data, test_label] = get_cifar_subset(5000, 1000, ...
        fullfile('cifar-10-batches-mat', 'cifar-10.mat'));
    end
    

rng(12345);

useParallel = false;     
p = gcp('nocreate');
if ~useParallel && ~isempty(p)
    delete(p);         
end
if useParallel
    if isempty(p)
        parpool('local',1);   
    end
    spmd, gpuDevice(1); end
else
    gpuDevice(1);
end

% Check for Training Results 
files = dir('training_results_*.mat'); %default  training_results_*.mat - change name of mat file before run to get new results 
if isempty(files)
    fprintf('No training results found. Starting training process...\n');
    clear compute_overlap adjust_synaptic_factors apply_kwta pi_controller train_spatial_pooler
            % Train
     new_sp(dataset, train_data, train_label);
 
    
    % Re-check training
    files = dir('training_results_*.mat');
    if isempty(files)
        error('Training failed');
    end
end

% Load Latest Results
[~, idx] = max([files.datenum]);
latest_file = files(idx).name;
fprintf('Loading training results from: %s\n', latest_file);
load(latest_file, 'w_permanence')
potential_radius = size(w_permanence,1);
overlap_dim = [ size(w_permanence,3), size(w_permanence,4) ];


% Evaluate and print accuracy
disp('Evaluating Accuracy...');
syn_connected_thresh = 0.5;
num_train  = size(train_data, 3);
feature_len = prod(overlap_dim);
train_sdrs = zeros(num_train, feature_len);
batch_size = cfg.OVERLAP_BATCH_SIZE;
threshold_tracker = struct();

for i = 1:num_train
    [overlap, syn_connected_thresh_batch, threshold_tracker] = compute_overlap( ...
        train_data, ...
        w_permanence, ...
        overlap_dim, ...
        potential_radius, ...
        i, ...
        syn_connected_thresh, ...
        cfg.USE_GPU, ...    
        i, ...  
        batch_size, threshold_tracker);
    dynamic_thresh = syn_connected_thresh + 0.1 * std(overlap(:));
    train_sdrs(i, :) = overlap(:) > dynamic_thresh;
end


[accuracy, sparsity, entropy] = evaluate_accuracy( ...
    test_data, test_label, ...
    w_permanence, potential_radius, overlap_dim, ...
    syn_connected_thresh, ...
    train_sdrs, train_label);




disp('Accuracy Evaluation Complete.');




% Metrics Analysis
disp('Starting Metrics Analysis...');
metrics_analysis(dataset, files);
disp('Metrics Analysis Completed.');
