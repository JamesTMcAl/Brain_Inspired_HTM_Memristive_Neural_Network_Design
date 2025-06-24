function [train_data, train_label, test_data, test_label] = get_mnist()

    % Load and process MNIST training images
    f = fopen(fullfile('mnist', 'train-images-idx3-ubyte'), 'r');
    if f == -1
        error('Error opening training image file');
    end
    train_data = fread(f, inf, 'uint8');
    fclose(f);
    train_data = train_data(17:end) ./ 255;  % Skips the 16-byte header and normalize
    train_data = permute(reshape(train_data, 28, 28, 60e3), [2 1 3]);  % Reshapes to 28x28 images

    % Load and process MNIST test images
    f = fopen(fullfile('mnist', 't10k-images-idx3-ubyte'), 'r');
    if f == -1
        error('Error opening test image file');
    end
    test_data = fread(f, inf, 'uint8');
    fclose(f);
    test_data = test_data(17:end) ./ 255;  % Skips the 16-byte header and normalize
    test_data = permute(reshape(test_data, 28, 28, 10e3), [2 1 3]);  % Reshapes to 28x28 images

    % Load and process MNIST training labels
    f = fopen(fullfile('mnist', 'train-labels-idx1-ubyte'), 'r');
    if f == -1
        error('Error opening training label file');
    end
    train_label = fread(f, inf, 'uint8');
    fclose(f);
    train_label = double(train_label(9:end)');  % Skip the 8-byte header 

    % Load and process MNIST test labels
    f = fopen(fullfile('mnist', 't10k-labels-idx1-ubyte'), 'r');
    if f == -1
        error('Error opening test label file');
    end
    test_label = fread(f, inf, 'uint8');
    fclose(f);
    test_label = double(test_label(9:end)');  % Skips the 8-byte header 

    if ndims(train_data) == 3
        % MNIST: [28x28xN]
    elseif ndims(train_data) == 4
        % CIFAR: [32x32x3xN]
    else
        error('Unsupported data dimensions: %s', mat2str(size(train_data)));
    end

end
