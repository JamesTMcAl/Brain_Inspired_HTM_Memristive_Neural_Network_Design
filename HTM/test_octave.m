% test_octave.m verify core SP functions work in Octave without GPU
pkg load image

% Add path
addpath(pwd);

% Load config
cfg = sp_config.instance();

cfg.USE_PCA_INIT = false;
cfg.USE_GPU = false;

fprintf('Config loaded OK\n');
fprintf('TARGET_ACTIVITY: %.2f\n', cfg.TARGET_ACTIVITY);

% Test initialize_permanence
fprintf('\nTesting initialize_permanence...\n');
try
    pca_coeff = zeros(3, 3);  % dummy, not used when PCA off
    potential_radius = 3;
    overlap_dimension = [8, 8];
    perm = initialize_permanence(pca_coeff, potential_radius, overlap_dimension, false);
    fprintf('initialize_permanence OK: size %s\n', mat2str(size(perm)));
catch ME
    fprintf('initialize_permanence FAILED: %s\n', ME.message);
end

% Test adjust_density
fprintf('\nTesting adjust_density...\n');
try
    pi_state = struct('I', 0, 'prevOutput', 0, 'hist', []);
    overlap_mat = rand(8, 8) * 0.4;
    [new_density, pi_state] = adjust_density(0.3, overlap_mat, 1, false, pi_state, false);
    fprintf('adjust_density OK: %.4f\n', new_density);
catch ME
    fprintf('adjust_density FAILED: %s\n', ME.message);
end

% Test calculate_entropy
fprintf('\nTesting calculate_entropy...\n');
try
    test_sdr = rand(100, 1) > 0.8;  % ~20% sparsity
    ent = calculate_entropy(test_sdr);
    fprintf('calculate_entropy OK: %.4f\n', ent);
catch ME
    fprintf('calculate_entropy FAILED: %s\n', ME.message);
end
% Test validateattributes
try
    validateattributes(0.5, {'numeric'}, {'scalar'}, 'test', 'x', 1);
    fprintf('validateattributes OK\n');
catch ME
    fprintf('validateattributes FAILED: %s\n', ME.message);
end

% Test 'like' syntax
try
    x = rand(3,3);
    y = ones(size(x), 'like', x);
    fprintf('ones like OK\n');
catch ME
    fprintf('ones like FAILED: %s\n', ME.message);
end

% Test apply_kwta
fprintf('\nTesting apply_kwta...\n');
try
    overlap_mat = rand(8, 8) * 0.5;
    density = 0.3;
    kwta_state = struct();
    [active_cols, avg_act, kwta_state, new_density] = apply_kwta(overlap_mat, density, 1, false, true, kwta_state);
    fprintf('apply_kwta OK: active=%d, density=%.4f\n', sum(active_cols(:)), new_density);
catch ME
    fprintf('apply_kwta FAILED: %s\n', ME.message);
end

fprintf('\nTesting compute_overlap...\n');
try
    pkg load image
    potential_radius = 3;
    overlap_dimension = [8, 8];
    H = overlap_dimension(1) + potential_radius - 1;
    W = overlap_dimension(2) + potential_radius - 1;
    train_data = rand(H, W);
    w_perm = rand(potential_radius, potential_radius, overlap_dimension(1), overlap_dimension(2));
    threshold_tracker = 0;
    ii = [1];
    sample_counter = 1;
    batch_size = 1;
    syn_thresh = 0.2;

    [overlap, thresh, threshold_tracker] = compute_overlap(train_data, w_perm, overlap_dimension, potential_radius, ii, syn_thresh, false, sample_counter, batch_size, threshold_tracker);
    fprintf('compute_overlap OK: size=%s, mean=%.4f\n', mat2str(size(overlap)), mean(overlap(:)));
catch ME
    fprintf('compute_overlap FAILED: %s\n', ME.message);
end

% Test update_permanence
fprintf('\nTesting update_permanence...\n');
try
    potential_radius = 3;
    overlap_dimension = [8, 8];
    w_perm = rand(potential_radius, potential_radius, overlap_dimension(1), overlap_dimension(2)) * 0.6 + 0.2;
    active_cols = rand(potential_radius, potential_radius, overlap_dimension(1), overlap_dimension(2)) > 0.7;
    memristor_stats = struct('write_counts', zeros(size(w_perm)), 'endurance_limit', 1e6);

    [w_new, params, energy, mem_stats] = update_permanence(w_perm, active_cols, 0.1, 0.05, memristor_stats, 1, 0.001, 0.01, 0.01);
    fprintf('update_permanence OK: mean_w=%.4f, energy=%.4f\n', mean(w_new(:)), energy);
catch ME
    fprintf('update_permanence FAILED: %s\n', ME.message);
end


fprintf('\nDone.\n');
