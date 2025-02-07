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

X = x(~rows_with_nan, :);
Y = group(~rows_with_nan, :);

X_mean = mean(X);
X_std = std(X);
X = (X - X_mean) ./ X_std;

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

% Initialize variables for feature selection frequency
features = size(X, 2);
feature_frequency = zeros(1, features);

% Perform forward feature selection 100 times
for iter = 1:2
    iter
    selected_features = [];
    remaining_features = 1:features;
    best_accuracy = zeros(1, features);
    accuracy_vals = zeros(10, features);

    for i = 1:features
        best_feature = [];
        for feature = remaining_features
            trial_features = [selected_features, feature];
            X_trial = X(:, trial_features);

            % Train ten neural networks for each feature extraction
            for j = 1:10
                % Split data into training, validation, and testing sets
                cv = cvpartition(Y_categorical, 'Holdout', 0.3);
                X_train = X_trial(training(cv), :);
                Y_train = Y_categorical(training(cv));
                X_test = X_trial(test(cv), :);
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

            % Select the feature that results in the highest average accuracy
            if mean(accuracy_vals(:, i)) > best_accuracy(i)
                best_accuracy(i) = mean(accuracy_vals(:, i));
                best_feature = feature;
            end
        end

        if isempty(best_feature)
            break;
        end

        selected_features = [selected_features, best_feature];
        remaining_features(remaining_features == best_feature) = [];
    end

    % Update feature frequency counter
    [AA, ind_BA] = max(best_accuracy);

    % Train final model with selected features
    selected_features = selected_features(1:ind_BA);
    feature_frequency(selected_features) = feature_frequency(selected_features) + 1;
end

% Plot feature selection frequency
figure;
bar(feature_frequency);
xlabel('Feature Index');
ylabel('Selection Frequency');
title('Feature Selection Frequency over 100 Iterations');
grid on;

