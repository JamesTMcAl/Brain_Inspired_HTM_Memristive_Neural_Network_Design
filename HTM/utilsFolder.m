function p = utilsFolder()
%UTILSFOLDER  path to local "utils" directory
    root = fileparts(which('sp_config'));   
    p    = fullfile(root, 'utils');         
end
