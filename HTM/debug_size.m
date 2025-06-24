function debug_size(var, varname)
    sz = size(var);
    if numel(sz) == 2
        fprintf('DEBUG: %s size = [%d %d]\n', varname, sz(1), sz(2));
    elseif numel(sz) == 3
        fprintf('DEBUG: %s size = [%d %d %d]\n', varname, sz(1), sz(2), sz(3));
    elseif numel(sz) == 4
        fprintf('DEBUG: %s size = [%d %d %d %d]\n', varname, sz(1), sz(2), sz(3), sz(4));
    else
        fprintf('DEBUG: %s size = [', varname);
        fprintf('%d ', sz);
        fprintf(']\n');
    end
end
