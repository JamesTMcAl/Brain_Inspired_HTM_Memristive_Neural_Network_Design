function [adj, state] = pi_controller(error, state, Kp, Ki, varargin)
% PI_CONTROLLER  PI controller with anti-windup and momentum
%
% Inputs:
%   error  : Current error signal
%   state  : Struct with 'I' (integral) and 'prevOutput' (optional)
%   Kp, Ki : Proportional/Integral gains
%   varargin : either a single struct(opts) or name/value pairs
%
% Outputs:
%   adj    : Adjustment term
%   state  : Updated state

    validateattributes(error, {'numeric'}, {'scalar','real','finite'}, mfilename,'error',1);
    validateattributes(state, {'struct'}, {},                mfilename,'state',2);
    validateattributes(Kp,    {'numeric'}, {'scalar','real','finite'}, mfilename,'Kp',3);
    validateattributes(Ki,    {'numeric'}, {'scalar','real','finite'}, mfilename,'Ki',4);


    % Parse varargin into opts
    if isempty(varargin)
        opts = struct();
    elseif numel(varargin)==1 && isstruct(varargin{1})
        opts = varargin{1};
    else
        opts = struct(varargin{:});
    end

    % Now set defaults
    if ~isfield(opts,'clampI'),   opts.clampI   = [-Inf Inf]; end
    if ~isfield(opts,'momentum'), opts.momentum = false;      end

    % Integral clamping
    new_I = state.I + error;
    state.I = min(max(new_I, opts.clampI(1)), opts.clampI(2));

    % PI adjustment
    adj = Kp * error + Ki * state.I;

    % Momentum term (optional)
    if opts.momentum && isfield(state, 'prevOutput')
        delta_adj = adj - state.prevOutput;
        delta_adj = min(max(delta_adj, -0.1), 0.1);
        cfg = sp_config.instance();
        adj = adj + cfg.MOMENTUM_GAIN * delta_adj;
    end

    % Update state
    state.prevOutput = adj;
end
