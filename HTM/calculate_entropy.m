function entropy = calculate_entropy(binary_matrix, show_visualization)
    if nargin < 2, show_visualization = false; end

    % Enforce binary values with fixed threshold
    binary_matrix = binary_matrix > 0.5; 

    % Calculate entropy
    active_ratio = mean(binary_matrix(:));
    if active_ratio > 0 && active_ratio < 1
        p1 = active_ratio;
        p0 = 1 - active_ratio;
        entropy = -(p1 * log2(p1 + 1e-10) + p0 * log2(p0 + 1e-10));
    else
        entropy = 0; % Uniform states have zero entropy
    end

    % Visualization
    if show_visualization
        figure; imagesc(binary_matrix); colormap gray; title('Binary Matrix'); colorbar;
    end
end
