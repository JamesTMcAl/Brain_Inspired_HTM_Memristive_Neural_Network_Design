function x = safe_gpuArray(x)
% safe_gpuArray - works in both MATLAB (GPU) and Octave (CPU)
try
    if exist('gpuArray', 'builtin')
        x = gpuArray(x);
    end
catch
end
end
