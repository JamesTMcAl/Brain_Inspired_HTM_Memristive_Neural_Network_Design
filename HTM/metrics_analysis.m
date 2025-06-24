function metrics_analysis(dataset, F)

%run from run all to get correct file name if running old training results
%change cfg.SubsetTraining in sp config to the correct dataset amount used
%METRICS_ANALYSIS  Visualise and summarise results saved by new_sp.m

%% Locate the newest results file
cfg = sp_config.instance();

assert(~isempty(F), 'No runs found – training_results_*.mat not present');
[~, idx] = max([F.datenum]);
R = load(F(idx).name);

% Number of epochs 
if isfield(R,'sparsity_per_epoch')
    epochs = numel(R.sparsity_per_epoch);
elseif isfield(R,'entropy_per_epoch')
    epochs = numel(R.entropy_per_epoch);
else
    epochs = NaN;
end

stop_ep = fielddefault(R,'stop_epoch', epochs);

%% 2×2 Main dashboard 
figure('Name','Training Metrics', ...
       'Units','normalized','Position',[.1 .1 .8 .8]);

% Panel 1 : Validation accuracy 
subplot(2,2,1);
if isfield(R,'val_accuracy_history') && ~isempty(R.val_accuracy_history)
    plot(1:numel(R.val_accuracy_history), R.val_accuracy_history,'-o','LineWidth',1.5); hold on;
    idxPlot = min(stop_ep, numel(R.val_accuracy_history));
    plot(idxPlot, R.val_accuracy_history(idxPlot),'rs','MarkerSize',8,'MarkerFaceColor','r');
    text(idxPlot, R.val_accuracy_history(idxPlot), '  Early stop','VerticalAlignment','bottom');
    xlabel('Epoch'); ylabel('Validation Accuracy (%)');
    title(['Validation Accuracy - ', upper(dataset), ' (', num2str(cfg.SubsetTraining), ') Dataset']); grid on; ylim([0,100]);
else
    text(0.5,0.5,'No validation data','HorizontalAlignment','center'); axis off;
end

% Sparsity per epoch
subplot(2,2,2);
if isfield(R,'sparsity_per_epoch') && ~isempty(R.sparsity_per_epoch)
    plot(1:epochs, R.sparsity_per_epoch,'-o','LineWidth',1.5);
    xlabel('Epoch'); ylabel('Sparsity (%)'); title(['SDR Sparsity - ', upper(dataset), ' (', num2str(cfg.SubsetTraining), ') Dataset']); grid on;
else
    text(0.5,0.5,'No sparsity data','HorizontalAlignment','center'); axis off;
end

% Entropy per epoch 
subplot(2,2,3);
if isfield(R,'entropy_per_epoch') && ~isempty(R.entropy_per_epoch)
    plot(1:epochs, R.entropy_per_epoch,'-o','LineWidth',1.5);
    xlabel('Epoch'); ylabel('Entropy (bits)'); title(['Entropy per Epoch - ', upper(dataset), ' (', num2str(cfg.SubsetTraining), ') Dataset']); grid on;
else
    text(0.5,0.5,'No entropy data','HorizontalAlignment','center'); axis off;
end

% Energy per epoch 
subplot(2,2,4);
if isfield(R,'energy_history') && ~isempty(R.energy_history)
    plot(1:numel(R.energy_history), R.energy_history,'-o','LineWidth',1.5);
    xlabel('Epoch'); ylabel('Energy / update'); title(['Energy per Epoch - ', upper(dataset), ' (', num2str(cfg.SubsetTraining), ') Dataset']); grid on;
else
    text(0.5,0.5,'No energy data','HorizontalAlignment','center'); axis off;
end

%% MNIST-specific plots
if strcmpi(dataset, 'mnist')
    if isfield(R,'write_cycle_history') && ~isempty(R.write_cycle_history)
        figure('Name','Write‑Cycle Distribution');
        histogram(R.write_cycle_history);
        xlabel('Write‑cycle count'); ylabel('Frequency'); title('Memristor Write‑Cycle Distribution');
    end

    if isfield(R,'sample_input') && isfield(R,'sample_sdr')
        figure('Name','SDR Example');
        subplot(1,2,1); imagesc(R.sample_input); colormap gray; axis image off; title('Original Input');
        subplot(1,2,2); imagesc(R.sample_sdr);   colormap gray; axis image off; title('Binary SDR');
    end
end

%% General plots
if isfield(R,'val_accuracy_history')
    energy_this_run = fielddefault(R,'energy_history',0);
    totalEnergyThis = sum(energy_this_run);
    cats = categorical({'This SP','Liu et al. (2022)','Baseline CNN'});
    accs = [R.val_accuracy_history(end), 74.5, 82.3];
    ener = [totalEnergyThis,             1.2e6, 3.4e6];

    figure('Name','Benchmark Comparison');
    yyaxis left; bar(cats, accs); ylabel('Accuracy (%)');
    yyaxis right; bar(cats, ener); ylabel('Total Energy (J)');
    title(['Accuracy vs. Lifetime Energy - ', upper(dataset), ' (', num2str(cfg.SubsetTraining), ') Dataset']);
    legend({'Accuracy','Energy'},'Location','northoutside');
end

if isfield(R,'inhibition_radius_history') && ~isempty(R.inhibition_radius_history)
    figure('Name','Inhibition Radius per Epoch');
    plot(1:numel(R.inhibition_radius_history), R.inhibition_radius_history,'-o','LineWidth',1.5);
    xlabel('Epoch'); ylabel('Radius'); title(['Inhibition Radius per Epoch - ', upper(dataset), ' (', num2str(cfg.SubsetTraining), ') Dataset']); grid on;
end

if isfield(R,'syn_inc_history') && isfield(R,'syn_dec_history')
    figure('Name','LTP vs LTD Rates');
    plot(1:numel(R.syn_inc_history), R.syn_inc_history,'-o','LineWidth',1.5); hold on;
    plot(1:numel(R.syn_dec_history), R.syn_dec_history,'-s','LineWidth',1.5);
    xlabel('Epoch'); ylabel('Rate'); legend('LTP','LTD'); title(['LTP vs. LTD Rates - ', upper(dataset), ' (', num2str(cfg.SubsetTraining), ') Dataset']); grid on;
end

%% Summary table and console output
finalAcc    = fielddefault_nested(R,'val_accuracy_history',@(v) v(end), NaN);
avgSparsity = fielddefault_vec(R,'sparsity_per_epoch', @mean, NaN);
avgEntropy  = fielddefault_vec(R,'entropy_per_epoch',  @mean, NaN);
totalEnergy = fielddefault_vec(R,'energy_history',     @sum, NaN);
totalWrites = fielddefault_vec(R,'write_cycle_history',@sum, NaN);

T = table(finalAcc, avgSparsity, avgEntropy, totalEnergy, totalWrites, ...
    'VariableNames',{'FinalAccuracy','AvgSparsity','AvgEntropy','TotalEnergy','TotalWrites'});

fprintf('\n=== Summary of Final Metrics ===\n');
disp(T);

if ~isnan(stop_ep) && stop_ep < epochs
    fprintf('Early stopping triggered at epoch %d (%.2f%% accuracy).\n', ...
            stop_ep, finalAcc);
end
end




function out = fielddefault(S, name, default)
%FIELDDEFAULT Return field value if it exists, else default
    if isfield(S, name)
        out = S.(name);
    else
        out = default;
    end
end
function out = fielddefault_nested(S, name, func, default)
% Return func(S.name) if field exists and is not empty, else default
    if isfield(S, name) && ~isempty(S.(name))
        out = func(S.(name));
    else
        out = default;
    end
end
function out = fielddefault_vec(S, name, func, default)
% Apply a reduction function like @mean or @sum on a vector field if it exists
    if isfield(S, name) && ~isempty(S.(name))
        out = func(S.(name));
    else
        out = default;
    end
end
