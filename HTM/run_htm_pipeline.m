% run_htm_pipeline.m
% End-to-end SP + TM pipeline on synthetic temporal data
% Tests the complete EA-MHTM system including the new TM layer

pkg load image
pkg load statistics
addpath(pwd);

fprintf('=== EA-MHTM Full Pipeline Test ===\n\n');

% Config
cfg = sp_config.instance();
cfg.USE_GPU = false;
cfg.USE_PCA_INIT = false;
cfg.DEBUG = false;

% Dataset params
potential_radius  = 3;
overlap_dimension = [8, 8];
H = overlap_dimension(1) + potential_radius - 1;
W = overlap_dimension(2) + potential_radius - 1;

% Generate synthetic temporal sequences
% 3 classes, each with a characteristic spatial pattern
% Sequence repeats: class 1 -> 2 -> 3 -> 1 -> 2 -> 3 ...
rng(42);
n_classes    = 3;
n_train      = 90;   % 30 per class, 10 full sequences
n_val        = 30;

fprintf('Generating synthetic temporal dataset...\n');
base_patterns = zeros(H, W, n_classes);
base_patterns(1:4, 1:4, 1) = 1;
base_patterns(1:4, 6:10, 2) = 1;
base_patterns(6:10, 3:7, 3) = 1;
train_data  = zeros(H, W, n_train);
train_labels = zeros(n_train, 1);
for i = 1:n_train
    cls = mod(i-1, n_classes) + 1;
    train_data(:,:,i) = base_patterns(:,:,cls) + 0.1*randn(H,W);
    train_data(:,:,i) = max(0, min(1, train_data(:,:,i)));
    train_labels(i)   = cls;
end

val_data   = zeros(H, W, n_val);
val_labels = zeros(n_val, 1);
for i = 1:n_val
    cls = mod(i-1, n_classes) + 1;
    val_data(:,:,i)  = base_patterns(:,:,cls) + 0.05*randn(H,W);
    val_labels(i)    = cls;
end
fprintf('Train: %d samples | Val: %d samples\n\n', n_train, n_val);

% Initialise SP weights
pca_coeff  = zeros(potential_radius, potential_radius);
w_perm     = initialize_permanence(pca_coeff, potential_radius, overlap_dimension, false);

% SP hyperparameters (from dissertation optimised values)
base_density     = 0.3085;
syn_inc          = 0.0975;
syn_dec          = 0.0040;
sample_counter   = 0;
syn_thresh       = 0.2;
decay_scaling    = 1/5797;
endurance_rate   = 1/179340;
entropy_thresh   = 0.05;
sparsity_thresh  = 1.0;
val_acc_history  = [];

% Initialise TM
tm_state = struct();

fprintf('--- Training SP + TM ---\n');
n_epochs  = 5;
epoch_results = zeros(n_epochs, 4); % acc, sparsity, entropy, anomaly

for epoch = 1:n_epochs
    cfg.EPOCH_START_FLAG = true;

    [w_perm, sample_counter, val_acc_history, base_density, ...
     syn_inc, syn_dec, ~, ~, ~, ~, ~, sparsity, entropy, ~] = ...
        train_spatial_pooler(train_data, train_labels, base_density, ...
        syn_inc, syn_dec, sample_counter, syn_thresh, false, w_perm, ...
        decay_scaling, endurance_rate, pca_coeff, potential_radius, ...
        overlap_dimension, val_data, val_labels, val_acc_history, ...
        entropy_thresh, sparsity_thresh, 'synthetic');

    % Run TM on training sequence for this epoch
    anomaly_sum = 0;
    for i = 1:n_train
        active_cols = rand(overlap_dimension) > (1 - base_density);
        [~, ~, ~, tm_state] = temporal_memory(active_cols, tm_state, true, ...
                                               sample_counter + i);
        anomaly_sum = anomaly_sum + tm_state.anomaly_score;
    end
    avg_anomaly = anomaly_sum / n_train;

    val_acc = 0;
    if ~isempty(val_acc_history)
        val_acc = val_acc_history(end);
    end

    epoch_results(epoch, :) = [val_acc, sparsity, entropy, avg_anomaly];

    fprintf('Epoch %d/%d | Acc=%.1f%% | Sparsity=%.2f%% | Entropy=%.4f | Anomaly=%.3f\n', ...
            epoch, n_epochs, val_acc, sparsity, entropy, avg_anomaly);
end

fprintf('\n--- Results Summary ---\n');
fprintf('%-8s %-10s %-12s %-10s %-10s\n', 'Epoch', 'Accuracy', 'Sparsity', 'Entropy', 'Anomaly');
for e = 1:n_epochs
    fprintf('%-8d %-10.1f %-12.2f %-10.4f %-10.3f\n', e, ...
            epoch_results(e,1), epoch_results(e,2), ...
            epoch_results(e,3), epoch_results(e,4));
end

fprintf('\n=== Pipeline complete ===\n');
