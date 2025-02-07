# ANN-meets-migraine
Artificial neural networks applied to somatosensory evoked potentials for migraine classification

Authors: Gabriele Sebastianelli, Daniele Secci, Francesco Casillo, Chiara Abagnale, Cherubino Di Lorenzo, Mariano Serrao, Shuu-Jiun Wang, Fu-Jung Hsiao, Gianluca Coppola

This repository contains MATLAB scripts designed to classify migraine conditions based on somatosensory evoked potentials (SEPs) using artificial neural networks (ANNs). The script implements preprocessing, feature extraction, and classification steps to facilitate accurate identification of migraine conditions.

# Context and Purpose

The goal of this script is to support the research described in the upcoming publication (details will be shared upon acceptance). The dataset used in the analysis is available upon request, as described in the publication. This repository provides an open framework for replicating and building upon the described methodology.

# Requirements
MATLAB (R2020b or later recommended)

Deep learning toolbox.

# Dataset
The dataset required for running this script is not included in the repository. It can be obtained upon request as specified in the forthcoming publication.

# Code Overview
This repository contains four scripts. Two of them focus on feature selection or dimensionality reduction of the initial set of features (ANN_HV_vs_MO_Perform_FFS.m and ANN_HV_vs_MO_Perform_PCA.m). Of the remaining two scripts, one uses the features identified by feed-forward feature selection as the final inputs for training and testing the ANN, while the other utilizes the identified principal components to train and test the ANN (ANN_HV_vs_MO_Build_ANN_FFS.m and ANN_HV_vs_MO_Build_ANN_PCA.m).


# Code Operational View

- ANN_HV_vs_MO_Perform_FFS.m:
This script implements a Forward Feature Selection (FFS) algorithm to identify the most relevant features for training and testing an Artificial Neural Network (ANN). The code is designed to analyze a dataset of healthy versus migraine subjects, determining which features contribute the most to the classification task. Below is an operational breakdown of the script:

Key Steps:
Data Loading and Preparation: The script begins by loading the dataset DatasetHVvsMO_depurato.mat. 
Specific variables are extracted and concatenated into a feature matrix X.
Rows with missing data (NaN values) are identified and removed from both the feature matrix X and the target vector Y.

Feature Normalization: Features in X are normalized to have zero mean and unit standard deviation.

Target Vector Preprocessing: The categorical target vector Y is converted into numeric labels and subsequently into a categorical format suitable for classification.

Feature Selection with ANN: The script performs Forward Feature Selection 100 iterations (reduced to 2 in the provided example for brevity).

For each iteration: Features are selected sequentially based on their contribution to classification accuracy.
A subset of features is evaluated by training and testing a neural network.
The feature that results in the highest average classification accuracy across 10 repetitions is selected.
The process continues until no significant improvement is observed or all features are evaluated.

Neural Network Training and Testing: A pattern recognition neural network (patternnet) with a single hidden layer (50 neurons) is trained using a stratified train-test split (cvpartition with 70% training, 30% testing).
The network performance is evaluated based on classification accuracy.

Feature Selection Frequency: A frequency counter tracks how often each feature is selected across the iterations.
The final selected features are used to train a model, and their indices are recorded.

Visualization: A bar plot is generated to visualize the selection frequency of each feature across the iterations.

Outputs: Feature Selection Frequency: A bar chart shows how often each feature was selected during the FFS process, indicating the most relevant features for the ANN.

Usage: Load your dataset into the required .mat file format with appropriate variable names.
Adjust the script parameters (e.g., hidden layer size, number of iterations, etc.) as needed.
Run the script to identify critical features and visualize their selection frequency.
