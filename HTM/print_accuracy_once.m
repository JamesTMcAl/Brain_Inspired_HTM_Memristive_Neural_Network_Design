function print_accuracy_once(accuracy)
    % Persistent variable to track if accuracy has been printed
    persistent accuracy_printed;
    if isempty(accuracy_printed)
        fprintf('Final Classification Accuracy: %.2f%%\n', accuracy);
        accuracy_printed = true;
    else
        disp('DEBUG: Accuracy already printed in this session.');
    end
end
