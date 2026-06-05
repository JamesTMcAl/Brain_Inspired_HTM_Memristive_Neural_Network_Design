function labels = classify_using_sdrs(sdrs, train_sdrs, train_labels)
% classify_using_sdrs - Octave-compatible nearest neighbour classifier
% Uses Hamming distance on SDRs (binary vectors)
n_test = size(sdrs, 1);
labels = zeros(n_test, 1);
for i = 1:n_test
    dists = sum(bsxfun(@xor, sdrs(i,:), train_sdrs), 2);
    [~, idx] = min(dists);
    labels(i) = train_labels(idx);
end
end
