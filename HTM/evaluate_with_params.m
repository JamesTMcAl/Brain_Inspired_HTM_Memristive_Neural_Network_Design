function accuracy = evaluate_with_params(train_data, train_label, test_data, test_label, density, inc, dec, potential_radius, pca_coeff)
   
input_dim      = [size(train_data,1) size(train_data,2)];
    overlap_dimension    = input_dim - (potential_radius-1);
    initial_w      = initialize_permanence(pca_coeff,            ...
                                           potential_radius,     ...
                                           overlap_dimension, cfg.USE_GPU);  

    try
        [w_permanence,~,~,~,~,~, ~, ~] = train_spatial_pooler( ...
            train_data,   train_label, ...
            density,      inc,         dec, ...
            0,            0.3,         false, ...        % sampleCounter, thresh, GPU
            initial_w,    1e4,         1e6,   ...        % decayScaling, enduranceRate
            pca_coeff,    potential_radius, overlap_dimension, ...
            [],           [],           [],    ...       % no validation data
            0.5,          25);                            % entropy & sparsity thresholds

        predicted_labels = infer_labels(test_data,w_permanence,train_data,train_label, overlap_dimension);

        % Compute accuracy
        accuracy = mean(predicted_labels == test_label) * 100;
        fprintf('Validation Accuracy: %.2f%%\n', accuracy);
    catch ME
        fprintf('Error in evaluate_with_params: %s\n', ME.message);
        fprintf('Traceback: %s\n', getReport(ME, 'extended'));
        accuracy = NaN;
    end
end
