function validate_inputs(train_data, train_label, val_data, val_label)
    % Validate training data
    if isempty(train_data) || isempty(train_label)
        error('Training data or labels are empty.');
    end

    % Ensure training data dimensions match labels
    if size(train_data, 3) ~= numel(train_label)
        error('Mismatch between training data samples and labels.');
    end

    % Validate validation data only if provided
    if nargin == 4
        if isempty(val_data) || isempty(val_label)
            error('Validation data or labels are empty.');
        end
        if size(val_data, 3) ~= numel(val_label)
            error('Mismatch between validation data samples and labels.');
        end
    end
end
