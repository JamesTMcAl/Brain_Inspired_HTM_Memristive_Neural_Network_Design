function [train_data, train_label, test_data, test_label] = get_mnist()

    % Find mnist folder relative to this function's location
    % Works regardless of working directory or who clones the repo
    this_dir  = fileparts(mfilename('fullpath'));
    mnist_dir = fullfile(this_dir, 'mnist');

    % Load training images
    f = fopen(fullfile(mnist_dir, 'train-images-idx3-ubyte'), 'r');
    if f == -1
        error('Cannot find MNIST files. Expected folder: %s\nDownload from: http://yann.lecun.com/exdb/mnist/', mnist_dir);
    end
    train_data = fread(f, inf, 'uint8');
    fclose(f);
    train_data = train_data(17:end) ./ 255;
    train_data = permute(reshape(train_data, 28, 28, 60e3), [2 1 3]);

    % Load test images
    f = fopen(fullfile(mnist_dir, 't10k-images-idx3-ubyte'), 'r');
    if f == -1, error('Cannot find MNIST test images in %s', mnist_dir); end
    test_data = fread(f, inf, 'uint8');
    fclose(f);
    test_data = test_data(17:end) ./ 255;
    test_data = permute(reshape(test_data, 28, 28, 10e3), [2 1 3]);

    % Load training labels
    f = fopen(fullfile(mnist_dir, 'train-labels-idx1-ubyte'), 'r');
    if f == -1, error('Cannot find MNIST training labels in %s', mnist_dir); end
    train_label = fread(f, inf, 'uint8');
    fclose(f);
    train_label = double(train_label(9:end)');

    % Load test labels
    f = fopen(fullfile(mnist_dir, 't10k-labels-idx1-ubyte'), 'r');
    if f == -1, error('Cannot find MNIST test labels in %s', mnist_dir); end
    test_label = fread(f, inf, 'uint8');
    fclose(f);
    test_label = double(test_label(9:end)');
end
