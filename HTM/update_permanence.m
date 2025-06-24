function [w_permanence, updated_params, energy, memristor_stats] = update_permanence(...
    w_permanence, active_cols, syn_inc, syn_dec, memristor_stats, sample_counter, endurance_rate, noiseLTP, noiseLTD)
%UPDATE_PERMANENCE  Apply LTP/LTD via deviceModel and track stats.
%
% Inputs:
%   w_permanence    – [4D] current synaptic weights
%   active_cols     – [4D logical] active‑column mask
%   syn_inc, syn_dec– LTP/LTD base deltas
%   memristor_stats – struct with write_cycles
%   sample_counter 
%   endurance_rate  – memristor endurance parameter
%
% Outputs:
%   w_permanence, updated_params energy, memristor_stats

    cfg = sp_config.instance();

    % Validate inputs
    validateattributes(w_permanence,{'numeric'},{'nonsparse','nonempty'});
    assert(ndims(w_permanence)==4, 'w_permanence must be 4‑D');
    

    validateattributes(active_cols,  {'logical'},{'size',size(w_permanence)},  mfilename,'active_cols',2);
    validateattributes(syn_inc,      {'numeric'},{'size',size(w_permanence)},  mfilename,'syn_inc',3);
    validateattributes(syn_dec,      {'numeric'},{'size',size(w_permanence)},  mfilename,'syn_dec',4);
    validateattributes(endurance_rate,{'numeric'},{'scalar','>=',0},     mfilename,'endurance_rate',7);

    dm = deviceModel(cfg.SYN_NOISE_LEVEL, endurance_rate);
    energy = 0;

    % touch synapses on active columns

    % Potentiation (LTP): input=1 → increase permanence
    potMask = (active_cols) & (syn_inc > 0);
    if any(potMask(:))
        deltaLTP = zeros(size(w_permanence), 'like', w_permanence);
        deltaLTP(potMask) = syn_inc(potMask);
        [w_permanence, eLTP, memristor_stats] = dm.applyLTP(...
            w_permanence, deltaLTP, memristor_stats, noiseLTP);
        energy = energy + eLTP;
    end

    % Depression (LTD): input=0 → decrease permanence
    depMask = (active_cols) & (syn_dec > 0);
    if any(depMask(:))
        deltaLTD = zeros(size(w_permanence), 'like', w_permanence);
        deltaLTD(depMask) = syn_dec(depMask);
        [w_permanence, eLTD, memristor_stats] = dm.applyLTD(...
            w_permanence, deltaLTD, memristor_stats, noiseLTD);
        energy = energy + eLTD;
    end

    updated_params = [];
end
