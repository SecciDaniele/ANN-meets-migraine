clc
clear all
close all

load DatasetHVvsMO_depurato.mat

group = DatasetHVvsMOdepurato.Group;
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
accuracy_vals = zeros(10, num_components);

% Holdout splitting parameters
holdoutRatio = 0.2;

% Train neural networks using different numbers of principal components
for i = 1:num_components
    X_pca = score(:, 1:i);

    % Train ten neural networks for each number of principal components
    for j = 1:100
        % Perform holdout splitting
        cv = cvpartition(Y_categorical, 'Holdout', holdoutRatio);

        X_train = X_pca(training(cv), :);
        Y_train = Y_categorical(training(cv));
        X_test = X_pca(test(cv), :);
        Y_test = Y_categorical(test(cv));

        % Train neural network
        hiddenLayerSize = [50]; % Number of neurons in the hidden layer
        net = patternnet(hiddenLayerSize);
        net.divideParam.trainRatio = 70/100;
        net.divideParam.valRatio = 30/100;
        net.divideParam.testRatio = 0/100;
        net.trainParam.showWindow = false;
        [net, ~] = train(net, X_train', dummyvar(Y_train)');

        % Test the Network
        Y_pred = net(X_test');
        [~, predictedClasses] = max(Y_pred, [], 1);
        predictedClasses = categorical(predictedClasses');

        % Calculate accuracy
        accuracy = sum(predictedClasses == Y_test) / numel(Y_test);
        accuracy_vals(j, i) = accuracy;
    end
end

% Calculate average accuracy and bounds
mean_accuracy = mean(accuracy_vals);
min_accuracy = min(accuracy_vals);
max_accuracy = max(accuracy_vals);

% Plot number of principal components vs accuracy
figure;
plot(1:num_components, mean_accuracy, 'b', 'LineWidth', 2);
hold on;
plot(1:num_components, min_accuracy, 'r--');
plot(1:num_components, max_accuracy, 'r--');
xlabel('Number of Principal Components');
ylabel('Accuracy');
title('Number of Principal Components vs Accuracy');
legend('Average Accuracy', 'Min Accuracy', 'Max Accuracy');
grid on;