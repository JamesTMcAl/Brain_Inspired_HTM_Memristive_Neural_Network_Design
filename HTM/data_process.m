function [train, train_l, test, test_l, no_train, no_test] = data_process(train_sample, train_label, test_sample, test_label)
    % Sort training data and labels
    [train_l, train_index] = sort(train_label);
    train = train_sample(:, :, train_index);
    assert(any(train(:) > 0), 'Training data is all zeros!');

    % Normalize data for HTM sparsity optimization
    train = train / 255.0; % Scale values to [0, 1]
    test = test_sample / 255.0;

    % Apply noise
    cfg = sp_config.instance();
    noise_std = cfg.NOISE_STD;    

    train = max(0, min(1, train + noise_std * randn(size(train))));

    test = max(0, min(1, test + noise_std * randn(size(test))));

    % Verify intensity range after normalization
    assert(all(train(:) >= 0 & train(:) <= 1), 'Train data out of range!');
    assert(all(test(:) >= 0 & test(:) <= 1), 'Test data out of range!');

    % Debugging visualization
    disp('Preprocessing Complete.');
    figure;
    imagesc(train(:, :, 1));
    title('First Training Sample After Preprocessing');
    colormap gray;

    % Label statistics
    unique_labels = unique(train_l);
    no_train = histcounts(train_l, [unique_labels; max(unique_labels)+1]);
    fprintf('Number of training samples per label: \n');
    disp(no_train);
end
