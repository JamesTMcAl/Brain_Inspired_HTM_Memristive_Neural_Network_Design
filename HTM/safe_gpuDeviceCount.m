function n = safe_gpuDeviceCount()
% Returns 0 in Octave, real count in MATLAB with GPU toolbox
try
    n = gpuDeviceCount();
catch
    n = 0;
end
end
