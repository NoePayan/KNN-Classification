# KNN Classification

A machine learning project focused on classifying data using the K-Nearest Neighbors (KNN) algorithm.

The project covers the complete machine learning pipeline, from data exploration and preprocessing to model optimization and evaluation.

## Project Overview

The objective is to build a classification model capable of predicting the class of observations based on seven input features.

Several steps are performed to analyze the dataset, optimize the KNN model and compare its performance with other classification algorithms.

## Dataset

The dataset contains:

* 1,012 training observations
* 4,000 test observations
* 7 input features (`C1` to `C7`)
* 1 target variable (`Label`)
* 4 classes: `0`, `1`, `2` and `3`

The training dataset contains no missing values.

Because KNN is based on distances between observations, feature scaling is applied before training the models.

## Methodology

The project follows these main steps:

1. Data loading and exploration
2. Class distribution analysis
3. Outlier detection
4. Feature scaling using `StandardScaler`
5. PCA for dimensionality reduction and visualization
6. Selection of the optimal number of neighbors
7. Comparison of distance metrics
8. Hyperparameter optimization using `GridSearchCV`
9. Comparison with other classification models
10. Model evaluation
11. Generation of predictions for the test dataset

## Models

The main models evaluated are:

* K-Nearest Neighbors (KNN)
* Logistic Regression
* Random Forest
* Support Vector Machine (SVM)

## Results

The initial KNN optimization gave the following results:

Euclidean 98.22% 
Manhattan **98.72%** 
Chebyshev 97.82% 
Minkowski 98.22% 

The best configuration found during the initial optimization was:

* **Number of neighbors:** `k = 1`
* **Best distance metric:** Manhattan
* **Cross-validation accuracy:** `98.72%`

Further hyperparameter optimization and model comparison are also performed in the project.

The original datasets are not included in the repository when redistribution is not permitted.


## Technologies

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* PCA
* KNN
* GridSearchCV


## Author

**Noé PAYAN**

Engineering Student — ESILV
Data & Artificial Intelligence
