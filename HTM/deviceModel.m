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

        function [w_new, energy, stats] = applyLTP(obj, w, delta, stats, noiseLTP)
            % LTP (potentiation) delta >= 0
            validateattributes(w,     {'numeric'},{'nonempty'},       mfilename, 'w',     2);
            validateattributes(delta, {'numeric'},{'size',size(w)},   mfilename, 'delta', 3);
            if ~isfield(stats,'write_cycles') || ~isequal(size(stats.write_cycles), size(w))
                stats.write_cycles = zeros(size(w));
            end

            noise = noiseLTP .* obj.noiseLevel;
            dW    = delta + noise;

            % State-dependent plasticity: saturates near the rails (nonlinear ion drift). Mid-range updates are full-strength; near 0 or 1 they shrink.
            window_factor = 4 .* w .* (1 - w) + 0.2;   % peaks at w=0.5, floor 0.2
            dW = dW .* window_factor;

            w_new  = min(max(w + dW, 0), 1);
            energy = norm(dW(:),2) / numel(dW);

            stats.write_cycles = stats.write_cycles + double(abs(dW)>0.01);
            w_new = obj.applyWear(w_new, stats);
        end

        function [w_new, energy, stats] = applyLTD(obj, w, delta, stats, noiseLTD)
            % LTD (depression) update: delta >= 0
            validateattributes(w,     {'numeric'},{'nonempty'},       mfilename, 'w',     2);
            validateattributes(delta, {'numeric'},{'size',size(w)},   mfilename, 'delta', 3);
            if ~isfield(stats,'write_cycles') || ~isequal(size(stats.write_cycles), size(w))
                stats.write_cycles = zeros(size(w));
            end

            noise = noiseLTD .* obj.noiseLevel;
            dW    = -delta + noise;

            % State-dependent plasticity (same nonlinearity as LTP)
            window_factor = 4 .* w .* (1 - w) + 0.2;
            dW = dW .* window_factor;

            w_new  = min(max(w + dW, 0), 1);
            energy = norm(dW(:),2) / numel(dW);

            stats.write_cycles = stats.write_cycles + double(abs(dW)>0.01);
            w_new = obj.applyWear(w_new, stats);
        end
    end

    methods (Access = private)
        function w_new = applyWear(obj, w_new, stats)
            % Graded endurance degradation, cells in the final 20% of life drift toward a stuck mid conductance; exhausted cells settle there.
            stuck_level = 0.5;
            wear        = stats.write_cycles / obj.enduranceRate;
            degrading   = wear > 0.8;
            if any(degrading(:))
                pull = min(1, (wear(degrading) - 0.8) / 0.2);   % 0→1 across final 20%
                w_new(degrading) = (1 - 0.3*pull) .* w_new(degrading) ...
                                 + (0.3*pull) .* stuck_level;
            end
            dead = wear > 1.5;
            w_new(dead) = stuck_level;
        end
    end
end
