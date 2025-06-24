function [train_data, train_label, test_data, test_label] = get_mnist_subset(num_train, num_test)
    % Load the full MNIST dataset
    [train_data_full, train_label_full, test_data_full, test_label_full] = get_mnist();
    
    % Extract the requested number of training samples
    assert(size(train_data_full, 3) >= num_train, 'Not enough training samples in MNIST dataset.');
    train_data = train_data_full(:, :, 1:num_train);
    train_label = train_label_full(1:num_train);
    
    % Extract the requested number of testing samples
    assert(size(test_data_full, 3) >= num_test, 'Not enough testing samples in MNIST dataset.');
    test_data = test_data_full(:, :, 1:num_test);
    test_label = test_label_full(1:num_test);
end
