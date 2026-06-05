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


fprintf('\nDone.\n');
