classdef deviceModel
    properties
        noiseLevel       % Scalar in [0,1]
        enduranceRate    % Positive scalar
    end
    methods
        function obj = deviceModel(noiseLevel, enduranceRate)
            % Constructor: validate inputs
            validateattributes(noiseLevel,    {'numeric'},{'scalar','>=',0,'<=',1}, mfilename, 'noiseLevel', 1);
            validateattributes(enduranceRate, {'numeric'},{'scalar','>=',0},        mfilename, 'enduranceRate',2);
            obj.noiseLevel    = noiseLevel;
            perturbed = enduranceRate .* (1 + 0.1*randn());
            obj.enduranceRate = max(perturbed, 1e5);        
        end

        function [w_new, energy, stats] = applyLTP(obj, w, delta, stats,noiseLTP)
            % LTP (potentiation) update: delta >= 0
            validateattributes(w,     {'numeric'},{'nonempty'},       mfilename, 'w',     2);
            validateattributes(delta, {'numeric'},{'size',size(w)},   mfilename, 'delta', 3);
            if ~isfield(stats,'write_cycles')
                stats.write_cycles = zeros(size(w));
            end

            % noise 
            noise = noiseLTP .* obj.noiseLevel;
            dW    = delta + noise;
            w_new = min(max(w + dW, 0), 1);

            energy = norm(dW(:),2) / numel(dW);

            % update write‐cycle counts
            stats.write_cycles = stats.write_cycles + double(abs(dW)>0.01);
            % simulate device failure after enduranceRate writes
            failed = stats.write_cycles > obj.enduranceRate;
            w_new(failed) = 0;
        end

        function [w_new, energy, stats] = applyLTD(obj, w, delta, stats, noiseLTD)
            % LTD (depression) update: delta >= 0
            validateattributes(w,     {'numeric'},{'nonempty'},       mfilename, 'w',     2);
            validateattributes(delta, {'numeric'},{'size',size(w)},   mfilename, 'delta', 3);
            if ~isfield(stats,'write_cycles')
                stats.write_cycles = zeros(size(w));
            end

            % noise 
            noise = noiseLTD .* obj.noiseLevel;
            dW    = -delta + noise;
            w_new = min(max(w + dW, 0), 1);

            energy = norm(dW(:),2) / numel(dW);

            % update write‐cycle counts
            stats.write_cycles = stats.write_cycles + double(abs(dW)>0.01);
            % simulate device failure after enduranceRate writes
            failed = stats.write_cycles > obj.enduranceRate;
            w_new(failed) = 0;
        end
    end
end
