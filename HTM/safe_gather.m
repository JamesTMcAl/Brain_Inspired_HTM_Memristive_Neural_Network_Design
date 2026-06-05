function x = safe_gather(x)
% safe_gather - works in both MATLAB (GPU) and Octave (CPU)
if exist('gpuArray', 'class') && isa(x, 'gpuArray')
    x = gather(x);
end
end
