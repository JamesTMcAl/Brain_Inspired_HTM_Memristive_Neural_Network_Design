function [train_data, train_label, val_data, val_label] = split_data(data, labels, val_ratio)
    % Determine whether to stratify or not
    N = numel(labels);
    if numel(unique(labels)) < N
        % True classification data: preserve class proportions
        cv = cvpartition(labels, 'HoldOut', val_ratio);
    else
        % All labels unique ( in unit tests): simple random hold-out
        cv = cvpartition(N,      'HoldOut', val_ratio);
    end

    % Split the data
    if ndims(data) == 3
        % MNIST-style [H x W x N]
        train_data = data(:, :, cv.training);
        val_data   = data(:, :, cv.test);
    elseif ndims(data) == 4
        % CIFAR-style [H x W x C x N]
        train_data = data(:, :, :, cv.training);
        val_data   = data(:, :, :, cv.test);
    else
        error('Unsupported data dimensions: %s', mat2str(size(data)));
    end

    % Split the labels
    train_label = labels(cv.training);
    val_label   = labels(cv.test);

    % Debug info
    fprintf('[DEBUG] After split_data: train_data shape %s, val_data shape %s\n', ...
        mat2str(size(train_data)), mat2str(size(val_data)));
    fprintf('[DEBUG] train_label: %d, val_label: %d\n', ...
        numel(train_label), numel(val_label));
end
