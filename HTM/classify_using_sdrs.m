function labels = classify_using_sdrs(sdrs, train_sdrs, train_labels)
  

    if exist('fitcecoc','file')
        model = fitcecoc(train_sdrs, train_labels);
    else
       warning('classify_using_sdrs:NoStatsTB',...
           'fitcecoc not found; falling back to fitclinear.');
       model = fitclinear(train_sdrs, train_labels);
    end    
    labels = predict(model, sdrs);
end
