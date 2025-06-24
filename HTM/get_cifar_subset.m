function [train_data, train_label, test_data, test_label] = get_cifar_subset(num_train, num_test, cifar_path)
    % Load preprocessed CIFAR-10 dataset
    cifar = load(cifar_path);

    RGB_train = cifar.train_data(:,:,:,1:num_train);     % 32×32×3×Nt
    RGB_test  = cifar.test_data (:,:,:,end-num_test+1:end);
    train_label = double(cifar.train_labels(1:num_train));
    test_label  = double(cifar.test_labels(end-num_test+1:end));

   
    lw   = [0.2989 0.5870 0.1140];                       % R,G,B weights
    train_data = squeeze( lw(1)*double(RGB_train(:,:,1,:)) + ...
                          lw(2)*double(RGB_train(:,:,2,:)) + ...
                          lw(3)*double(RGB_train(:,:,3,:)) ) / 255;       % → 32×32×Nt
    test_data  = squeeze( lw(1)*double(RGB_test (:,:,1,:)) + ...
                          lw(2)*double(RGB_test (:,:,2,:)) + ...
                          lw(3)*double(RGB_test (:,:,3,:)) ) / 255;   

    

end
