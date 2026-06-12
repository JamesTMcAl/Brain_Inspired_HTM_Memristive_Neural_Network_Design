function [train_data, train_label, test_data, test_label] = get_mnist_subset(num_train, num_test)
    if nargin < 1, num_train = 4000; end
    if nargin < 2, num_test  = 1000; end

    [train_data_full, train_label_full, test_data_full, test_label_full] = get_mnist();

    assert(size(train_data_full, 3) >= num_train, ...
           'Not enough training samples. Requested %d, have %d.', num_train, size(train_data_full,3));
    assert(size(test_data_full, 3) >= num_test, ...
           'Not enough test samples. Requested %d, have %d.', num_test, size(test_data_full,3));

    train_data  = train_data_full(:, :, 1:num_train);
    train_label = train_label_full(1:num_train);
    test_data   = test_data_full(:, :, 1:num_test);
    test_label  = test_label_full(1:num_test);
end
