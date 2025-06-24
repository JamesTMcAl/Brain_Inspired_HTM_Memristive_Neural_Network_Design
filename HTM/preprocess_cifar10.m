function preprocess_cifar10()
    % Directory containing CIFAR-10 batches
    cifar_dir = 'cifar-10-batches-mat';
    
    % Initialize containers for training and testing data
    train_data = [];
    train_labels = [];
    
    % Load training batches
    for batch_num = 1:5
        batch_file = fullfile(cifar_dir, sprintf('data_batch_%d.mat', batch_num));
        if isfile(batch_file)
            batch = load(batch_file);
            train_data = [train_data; batch.data];
            train_labels = [train_labels; batch.labels];
        else
            error('Batch file %s not found.', batch_file);
        end
    end
    
    % Reshape and normalize training data
    train_data = reshape(train_data, [], 32, 32, 3); % CIFAR-10 is 32x32 with 3 color channels
    train_data = permute(train_data, [2, 3, 4, 1]); % Rearrange to [H, W, C, N]
    train_data = double(train_data) / 255; % Normalize to [0, 1]

    % Load test batch
    test_file = fullfile(cifar_dir, 'test_batch.mat');
    if isfile(test_file)
        test_batch = load(test_file);
        test_data = reshape(test_batch.data, [], 32, 32, 3);
        test_data = permute(test_data, [2, 3, 4, 1]);
        test_data = double(test_data) / 255;
        test_labels = test_batch.labels;
    else
        error('Test batch file not found.');
    end
    
    % Save as a single .mat file
    save(fullfile(cifar_dir, 'cifar-10.mat'), 'train_data', 'train_labels', 'test_data', 'test_labels', '-v7.3');
    fprintf('CIFAR-10 dataset has been preprocessed and saved to cifar-10.mat.\n');
end
