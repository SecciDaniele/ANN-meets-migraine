clc
clear all
close all

load DatasetHVvsMO_depurato.mat

group = DatasetHVvsMOdepurato.Group;
% Name = DatasetHVvsMOdepurato.Name;
% Sex = DatasetHVvsMOdepurato.sex;
Var_1 = DatasetHVvsMOdepurato.VarName16;
Var_2 = DatasetHVvsMOdepurato.VarName17;
Var_3 = DatasetHVvsMOdepurato.VarName18;
Var_4 = DatasetHVvsMOdepurato.N20P25;
Var_5 = DatasetHVvsMOdepurato.P25N33;
Slope_1 = DatasetHVvsMOdepurato.Slope12;
Slope_2 = DatasetHVvsMOdepurato.Slope13;
Proxy_1 = DatasetHVvsMOdepurato.preHFOLat;
Proxy_2 = DatasetHVvsMOdepurato.postHFOLat;
Proxy_3 = DatasetHVvsMOdepurato.preHFOAmp;
Proxy_4 = DatasetHVvsMOdepurato.postHFOAmp;

% Concatenate variables into a single array
x = cat(2, Var_1, Var_2, Var_3, Var_4, Var_5, Slope_1, Slope_2, Proxy_1, Proxy_2, Proxy_3, Proxy_4);

% Find rows containing NaN values
rows_with_nan = any(isnan(x), 2);
% Name = Name(~rows_with_nan,:);
% Sex = Sex(~rows_with_nan,:);

X = x(~rows_with_nan, :);
Y = group(~rows_with_nan, :);

% X_mean = mean(X);
% X_std = std(X);
% X = (X - X_mean) ./ X_std;

% Find unique categories in the target vector
unique_categories = unique(Y);

% Convert categorical target vector to a cell array of character vectors
target_cell = cellstr(Y);

% Convert unique categories to a cell array of character vectors
unique_categories_char = cellstr(unique_categories);

% Create a map from categories to numeric labels
category_to_label = containers.Map(unique_categories_char, 1:numel(unique_categories));

% Convert categorical target vector to numeric labels
numeric_labels = cellfun(@(x) category_to_label(x), target_cell);

Y_categorical = categorical(numeric_labels);

% Perform PCA
[coeff, score, ~, ~, explained] = pca(X);

% Number of principal components to test
num_components = size(score, 2);

X_selected = score(:, 1:4);

% Initialize variables for accumulating confusion matrices and metrics
num_trials = 100; % Number of trials
num_classes = numel(unique(Y_categorical)); % Number of classes
confMat_sum_train = zeros(num_classes, num_classes);
confMat_sum_val = zeros(num_classes, num_classes);
confMat_sum_test = zeros(num_classes, num_classes);
confMat_sum_overall = zeros(num_classes, num_classes);
roc_data = zeros(176, numel(unique(Y_categorical)),num_trials,2);
auc_values = zeros(num_trials, numel(unique(Y_categorical)));

% Initialize arrays to store metrics for all datasets (train, val, test)
accuracy_vals_train = zeros(num_trials, 1);
sensitivity_vals_train = zeros(num_trials, num_classes);
specificity_vals_train = zeros(num_trials, num_classes);
f1Score_vals_train = zeros(num_trials, num_classes);

accuracy_vals_val = zeros(num_trials, 1);
sensitivity_vals_val = zeros(num_trials, num_classes);
specificity_vals_val = zeros(num_trials, num_classes);
f1Score_vals_val = zeros(num_trials, num_classes);

accuracy_vals_test = zeros(num_trials, 1);
sensitivity_vals_test = zeros(num_trials, num_classes);
specificity_vals_test = zeros(num_trials, num_classes);
f1Score_vals_test = zeros(num_trials, num_classes);

accuracy_vals_overall = zeros(num_trials, 1);
sensitivity_vals_overall = zeros(num_trials, num_classes);
specificity_vals_overall = zeros(num_trials, num_classes);
f1Score_vals_overall = zeros(num_trials, num_classes);


for j = 1:num_trials
    % Define neural network architecture
    hiddenLayerSize = [50]; % Number of neurons in the hidden layer
    net = patternnet(hiddenLayerSize);
    
    % Setup Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = 65/100;
    net.divideParam.valRatio = 20/100;
    net.divideParam.testRatio = 15/100;
    net.trainParam.showWindow = false;
    
    % Train the Network
    [net, tr] = train(net, X_selected', dummyvar(Y_categorical)');
    
    % Get indices for training, validation, and test sets
    trainInd = tr.trainInd;  % Indices of training set
    valInd = tr.valInd;      % Indices of validation set
    testInd = tr.testInd;    % Indices of test set
    
    % Test on training data
    Y_train_pred = net(X_selected(trainInd, :)');
    [~, predictedClassesTrain] = max(Y_train_pred, [], 1);
    predictedClassesTrain = categorical(predictedClassesTrain');
    confMat_train = confusionmat(Y_categorical(trainInd), predictedClassesTrain);
    
    % Test on validation data
    Y_val_pred = net(X_selected(valInd, :)');
    [~, predictedClassesVal] = max(Y_val_pred, [], 1);
    predictedClassesVal = categorical(predictedClassesVal');
    confMat_val = confusionmat(Y_categorical(valInd), predictedClassesVal);
    
    % Test on test data
    Y_test_pred = net(X_selected(testInd, :)');
    [~, predictedClassesTest] = max(Y_test_pred, [], 1);
    predictedClassesTest = categorical(predictedClassesTest');
    confMat_test = confusionmat(Y_categorical(testInd), predictedClassesTest);

    % Test on overall data
    Y_overall_pred = net(X_selected');
    [~, predictedClassesOverall] = max(Y_overall_pred, [], 1);
    predictedClassesOverall = categorical(predictedClassesOverall');
    confMat_overall = confusionmat(Y_categorical, predictedClassesOverall);
    
    % Accumulate confusion matrices
    confMat_sum_train = confMat_sum_train + confMat_train;
    confMat_sum_val = confMat_sum_val + confMat_val;
    confMat_sum_test = confMat_sum_test + confMat_test;
    confMat_sum_overall = confMat_sum_overall + confMat_overall;
    
    % Calculate metrics for training data
    accuracy_train = sum(diag(confMat_train)) / sum(confMat_train(:));
    sensitivity_train = diag(confMat_train) ./ sum(confMat_train, 2);
    specificity_train = (sum(confMat_train(:)) - sum(confMat_train, 2) - sum(confMat_train, 1)' + diag(confMat_train)) ./ (sum(confMat_train(:)) - sum(confMat_train, 2));
    precision_train = diag(confMat_train) ./ sum(confMat_train, 1)';
    f1Score_train = 2 * (precision_train .* sensitivity_train) ./ (precision_train + sensitivity_train);
    
    % Store training metrics
    accuracy_vals_train(j) = accuracy_train;
    sensitivity_vals_train(j, :) = sensitivity_train;
    specificity_vals_train(j, :) = specificity_train;
    f1Score_vals_train(j, :) = f1Score_train;
    
    % Calculate metrics for validation data
    accuracy_val = sum(diag(confMat_val)) / sum(confMat_val(:));
    sensitivity_val = diag(confMat_val) ./ sum(confMat_val, 2);
    specificity_val = (sum(confMat_val(:)) - sum(confMat_val, 2) - sum(confMat_val, 1)' + diag(confMat_val)) ./ (sum(confMat_val(:)) - sum(confMat_val, 2));
    precision_val = diag(confMat_val) ./ sum(confMat_val, 1)';
    f1Score_val = 2 * (precision_val .* sensitivity_val) ./ (precision_val + sensitivity_val);
    
    % Store validation metrics
    accuracy_vals_val(j) = accuracy_val;
    sensitivity_vals_val(j, :) = sensitivity_val;
    specificity_vals_val(j, :) = specificity_val;
    f1Score_vals_val(j, :) = f1Score_val;
    
    % Calculate metrics for test data
    accuracy_test = sum(diag(confMat_test)) / sum(confMat_test(:));
    sensitivity_test = diag(confMat_test) ./ sum(confMat_test, 2);
    specificity_test = (sum(confMat_test(:)) - sum(confMat_test, 2) - sum(confMat_test, 1)' + diag(confMat_test)) ./ (sum(confMat_test(:)) - sum(confMat_test, 2));
    precision_test = diag(confMat_test) ./ sum(confMat_test, 1)';
    f1Score_test = 2 * (precision_test .* sensitivity_test) ./ (precision_test + sensitivity_test);
    
    % Store test metrics
    accuracy_vals_test(j) = accuracy_test;
    sensitivity_vals_test(j, :) = sensitivity_test;
    specificity_vals_test(j, :) = specificity_test;
    f1Score_vals_test(j, :) = f1Score_test;

    % Calculate metrics for overall data
    accuracy_overall = sum(diag(confMat_overall)) / sum(confMat_overall(:));
    sensitivity_overall = diag(confMat_overall) ./ sum(confMat_overall, 2);
    specificity_overall = (sum(confMat_overall(:)) - sum(confMat_overall, 2) - sum(confMat_overall, 1)' + diag(confMat_overall)) ./ (sum(confMat_overall(:)) - sum(confMat_overall, 2));
    precision_overall = diag(confMat_overall) ./ sum(confMat_overall, 1)';
    f1Score_overall = 2 * (precision_overall .* sensitivity_overall) ./ (precision_overall + sensitivity_overall);
    
    % Store test metrics
    accuracy_vals_overall(j) = accuracy_overall;
    sensitivity_vals_overall(j, :) = sensitivity_overall;
    specificity_vals_overall(j, :) = specificity_overall;
    f1Score_vals_overall(j, :) = f1Score_overall;


    % ROC PLOT
    Y_categorical_numeric = double(Y_categorical);
    
    % Compute the ROC curve and AUC for each class
    for i = 1:numel(unique(Y_categorical))
        [X_ROC, Y_ROC, ~, AUC] = perfcurve(double(Y_categorical) == i, Y_overall_pred(i, :), 1);
        auc_values(j, i) = AUC;
        roc_data(:,:, j,i) = [X_ROC, Y_ROC]; % Save the ROC curve data (FPR, TPR)
    end

end

% Compute the average AUC across all trials
mean_auc = median(auc_values);
std_auc = std(auc_values);

% Display average AUC per class
for i = 1:numel(unique(Y_categorical))
    fprintf('Average AUC for class %d: %.2f (± %.2f)\n', i, mean_auc(i), std_auc(i));
end

% Plot the average ROC curve for the current class
for i = 1:numel(unique(Y_categorical))
    figure;
    roc=roc_data(:,:,:,i);
    mean_roc = median(roc, 3); % Resulting in a 176x2 matrix
    plot(mean_roc(:, 1), mean_roc(:, 2), 'LineWidth', 2);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(sprintf('Average ROC Curve for Class %d (AUC = %.2f)', i, mean_auc(1)));
    grid on;
end

% Calculate average metrics for train
mean_accuracy_train = median(accuracy_vals_train);
mean_sensitivity_train = median(sensitivity_vals_train);
mean_specificity_train = median(specificity_vals_train);
mean_f1Score_train = median(f1Score_vals_train);

% Calculate variance of the metrics for train
var_accuracy_train = std(accuracy_vals_train);
var_sensitivity_train = std(sensitivity_vals_train);
var_specificity_train = std(specificity_vals_train);
var_f1Score_train = std(f1Score_vals_train);

% Calculate average metrics for validation
mean_accuracy_validation = median(accuracy_vals_val);
mean_sensitivity_validation = median(sensitivity_vals_val);
mean_specificity_validation = median(specificity_vals_val);
mean_f1Score_validation = median(f1Score_vals_val);

% Calculate variance of the metrics for validation
var_accuracy_validation = std(accuracy_vals_val);
var_sensitivity_validation = std(sensitivity_vals_val);
var_specificity_validation = std(specificity_vals_val);
var_f1Score_validation = std(f1Score_vals_val);

% Calculate average metrics for test
mean_accuracy_test = median(accuracy_vals_test);
mean_sensitivity_test = median(sensitivity_vals_test);
mean_specificity_test = median(specificity_vals_test);
mean_f1Score_test = median(f1Score_vals_test);

% Calculate variance of the metrics for test
var_accuracy_test = std(accuracy_vals_test);
var_sensitivity_test = std(sensitivity_vals_test);
var_specificity_test = std(specificity_vals_test);
var_f1Score_test = std(f1Score_vals_test);


% Calculate average metrics for overall
mean_accuracy_overall = median(accuracy_vals_overall);
mean_sensitivity_overall = median(sensitivity_vals_overall);
mean_specificity_overall = median(specificity_vals_overall);
mean_f1Score_overall = median(f1Score_vals_overall);

% Calculate variance of the metrics for overall
var_accuracy_overall = std(accuracy_vals_overall);
var_sensitivity_overall = std(sensitivity_vals_overall);
var_specificity_overall = std(specificity_vals_overall);
var_f1Score_overall = std(f1Score_vals_overall);

% Compute average confusion matrices
confMat_avg_train = round(confMat_sum_train / num_trials);
confMat_avg_val = round(confMat_sum_val / num_trials);
confMat_avg_test = round(confMat_sum_test / num_trials);
confMat_avg_overall = round(confMat_sum_overall / num_trials);
% classLabels = categories(Y_categorical);  % Assuming Y_categorical has the class labels
classLabels = {'HV','MO'};  % Assuming Y_categorical has the class labels

% Plot average confusion matrix for the training set
figure;
chart = confusionchart(confMat_avg_train, classLabels, 'Title', 'Average Confusion Matrix - Training Set');
chart.ColumnSummary = 'column-normalized';
chart.RowSummary = 'row-normalized';

% Plot average confusion matrix for the validation set
figure;
chart = confusionchart(confMat_avg_val, classLabels, 'Title', 'Average Confusion Matrix - Validation Set');
chart.ColumnSummary = 'column-normalized';
chart.RowSummary = 'row-normalized';

% Plot average confusion matrix for the test set
figure;
chart = confusionchart(confMat_avg_test, classLabels, 'Title', 'Average Confusion Matrix - Test Set');
chart.ColumnSummary = 'column-normalized';
chart.RowSummary = 'row-normalized';

% Plot average confusion matrix for the training set
figure;
chart = confusionchart(confMat_avg_overall, classLabels, 'Title', 'Average Confusion Matrix - Whole Set');
chart.ColumnSummary = 'column-normalized';
chart.RowSummary = 'row-normalized';



