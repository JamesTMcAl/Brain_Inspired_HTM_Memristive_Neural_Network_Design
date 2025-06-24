function [accuracy, sparsity, entropy] = evaluate_accuracy(data, labels, w_permanence, potential_radius, overlap_dimension, syn_connected_thresh, train_sdrs, train_labels)   
fprintf('Evaluating HTM Model Performance...\n');
    num_samples = size(data, 3);
    sdrs = zeros(num_samples, prod(overlap_dimension));
    persistent sparsity_window;
cfg = sp_config.instance();
threshold_tracker = struct();

    for i = 1:num_samples
        [overlap, syn_connected_thresh, threshold_tracker] = compute_overlap(data,w_permanence,overlap_dimension, ...
                              potential_radius,i,syn_connected_thresh,cfg.USE_GPU,i, cfg.OVERLAP_BATCH_SIZE, threshold_tracker);

        % Dynamic threshold consistent with training:
        dynamic_thresh       = syn_connected_thresh + 0.1*std(overlap(:));
        sdrs(i, :) = overlap(:) > dynamic_thresh;

        if i == 1
            fprintf('Sample SDR Binary Matrix (Sample 1): %s\n', mat2str(sdrs(i, :)));
        end
    end

    % Classification using SDRs.
    model       = fitcecoc(train_sdrs, train_labels);
    predictions = predict(model, sdrs);
    accuracy    = mean(predictions == labels)*100;

    sparsity = mean(sum(sdrs, 2)) / size(sdrs, 2) * 100;
    probabilities = mean(sdrs, 1);
    entropy = -sum(probabilities .* log2(probabilities + eps));
    
    fprintf('Evaluation Metrics - Accuracy: %.2f%%, Sparsity: %.2f%%, Entropy: %.4f\n', accuracy, sparsity, entropy);
    
    % if sparsity consistently decreases over multiple evaluations
        if isempty(sparsity_window)
    sparsity_window = sparsity * ones(1, 5);
        end

        % Update the trend window
        sparsity_window = [sparsity_window(2:end), sparsity];
        sparsity_trend = mean(diff(sparsity_window));  % Measure change over time

        % Trigger warning only if sparsity keeps decreasing
        if sparsity_trend < -0.5
    fprintf('[WARNING] SDR sparsity dropping (%.2f%%). Consider adjusting inhibition radius.\n', sparsity);
        end

end
