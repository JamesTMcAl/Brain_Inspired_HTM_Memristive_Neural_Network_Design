function w_permanence = initialize_permanence(pca_coeff, potential_radius, overlap_dimension, use_gpu)
fprintf('[DEBUG] potential_radius=%d, size(pca_coeff)=%s\n', ...
    potential_radius, mat2str(size(pca_coeff)));

permanence_min = 0.4;
    permanence_max = 0.8;
    num_patches = overlap_dimension(1) * overlap_dimension(2);


   cfg = sp_config.instance();
    if cfg.USE_PCA_INIT
    if ~isequal(size(pca_coeff), [potential_radius, potential_radius])
        error('[FATAL] PCA coeff size mismatch: expected [%d,%d], got %s', ...
            potential_radius, potential_radius, mat2str(size(pca_coeff)));
    end
    w = zeros(potential_radius,potential_radius,num_patches);
        for k = 1:num_patches
            r = mod(k-1, overlap_dimension(1));
         c = floor((k-1)/overlap_dimension(1));
            w(:,:,k) = circshift(pca_coeff,[r c]);
        end

    else
         % orthogonal random patches if no PCA
         fprintf('[LOG] initialize_permanence: USE_PCA_INIT=false → using orthogonal random patches\n');

     pr2         = potential_radius^2;
    num_patches = overlap_dimension(1) * overlap_dimension(2);
    randPatches = randn(pr2, num_patches);
    if num_patches >= pr2
        Q = orth(randPatches);            % pr2×pr2
        reps        = ceil(num_patches/pr2);
        randPatches = repmat(Q, 1, reps); % pr2×(pr2*reps)
        randPatches = randPatches(:,1:num_patches);
    else
        randPatches = orth(randPatches')';  % pr2×num_patches
    end
    mn = min(randPatches(:));
    mx = max(randPatches(:));
    randPatches = (randPatches - mn) / (mx - mn + eps);

    w = reshape(randPatches, potential_radius, potential_radius, num_patches);
    end


    w_permanence = reshape(w, ...
        [potential_radius, potential_radius, overlap_dimension(1), overlap_dimension(2)]);

    w_permanence = permanence_min + (permanence_max - permanence_min) * w_permanence;
    w_permanence = w_permanence + 0.03 * randn(size(w_permanence));
    w_permanence = min(max(w_permanence, 0), 1);

    fprintf('[DEBUG] w_permanence size: %s\n', mat2str(size(w_permanence)));


    if use_gpu && ~isOctave()
        w_permanence = gpuArray(w_permanence);
    end

    fprintf('[LOG] Permanence initialized: [%.2f, %.2f] | Size: %s\n', ...
        permanence_min, permanence_max, mat2str(size(w_permanence)));

    if isempty(w_permanence)
        error('[ERROR] w_permanence failed to initialize.');
    end

    fprintf('[DEBUG] Initialized: Mean=%.4f | Min=%.4f | Max=%.4f | Size=%s\n', ...
        mean(w_permanence(:)), min(w_permanence(:)), max(w_permanence(:)), mat2str(size(w_permanence)));
end
