# KNN Classification
# This program allows us to:
# - load and analyze the data
# - standardize the features
# - find the best value of k
# - compare different distance metrics
# - optimize the model using GridSearchCV
# - compare KNN with other models
# - evaluate the final model
# - save the model and generate predictions

import os
import time
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    GridSearchCV,
    learning_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_curve,
    auc
)

warnings.filterwarnings("ignore")


# Parameters

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

FEATURES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
TARGET = "Label"
ID = "Id"

RANDOM_STATE = 42
N_SPLITS = 5

K_MIN = 1
K_MAX = 30

# Threshold used to detect values that are far from the mean
Z_THRESHOLD = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load the data

def load_data():
    print("\nLoading data...")

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"{TRAIN_PATH} was not found.")

    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"{TEST_PATH} was not found.")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    print(f"Train : {train.shape}")
    print(f"Test  : {test.shape}")

    # Check that the required columns are present
    for column in FEATURES:
        if column not in train.columns:
            raise ValueError(f"Column {column} is missing from the training set.")

    if TARGET not in train.columns:
        raise ValueError(f"Column {TARGET} is missing from the training set.")

    if ID not in test.columns:
        raise ValueError(f"Column {ID} is missing from the test set.")

    return train, test


# Exploratory data analysis

def exploratory_analysis(train):

    print("\nExploratory Data Analysis")

    print("\nFirst rows:")
    print(train.head())

    print("\nDescriptive statistics:")
    print(train[FEATURES].describe().T)

    # Check for missing values
    print("\nMissing values:")

    missing = train.isnull().sum()

    if missing.sum() == 0:
        print("No missing values.")
    else:
        print(missing[missing > 0])

    # Check the distribution of the classes
    print("\nClass distribution:")

    counts = train[TARGET].value_counts()

    print(counts)

    print("\nClass proportions:")
    print((counts / len(train)).round(3))

    # Plot the class distribution
    plt.figure(figsize=(6, 4))

    sns.countplot(
        data=train,
        x=TARGET
    )

    plt.title("Class Distribution")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "eda_distribution_classes.png"
        ),
        dpi=150
    )

    plt.close()

    # Plot the correlation matrix
    plt.figure(figsize=(7, 6))

    correlation = train[FEATURES].corr()

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title("Feature Correlation Matrix")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "eda_correlation.png"
        ),
        dpi=150
    )

    plt.close()

    # Boxplots for each feature
    n_columns = 4
    n_rows = int(np.ceil(len(FEATURES) / n_columns))

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4 * n_columns, 3.5 * n_rows)
    )

    axes = np.array(axes).reshape(-1)

    for i, feature in enumerate(FEATURES):

        sns.boxplot(
            data=train,
            x=TARGET,
            y=feature,
            ax=axes[i]
        )

        axes[i].set_title(feature)

    # Hide unused plots
    for i in range(len(FEATURES), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "eda_boxplots.png"
        ),
        dpi=150
    )

    plt.close()

    # Pairplot to visualize relationships between features
    if len(FEATURES) <= 8:

        pairplot = sns.pairplot(
            train[FEATURES + [TARGET]],
            hue=TARGET,
            diag_kind="kde"
        )

        pairplot.savefig(
            os.path.join(
                OUTPUT_DIR,
                "eda_pairplot.png"
            ),
            dpi=150
        )

        plt.close("all")


# Detect possible outliers

def find_outliers(train):

    print("\nLooking for possible outliers...")

    z_scores = np.abs(
        (train[FEATURES] - train[FEATURES].mean())
        / train[FEATURES].std()
    )

    outlier_mask = (z_scores > Z_THRESHOLD).any(axis=1)

    outliers = train[outlier_mask]

    print(
        f"{len(outliers)} possible outliers found "
        f"(z-score > {Z_THRESHOLD})."
    )

    return outliers


# Preprocess the data

def preprocess(train, test):

    X_train = train[FEATURES].values
    y_train = train[TARGET].values

    X_test = test[FEATURES].values
    test_ids = test[ID].values

    # KNN is based on distances between observations.
    # Standardization makes sure that all features are on a similar scale.
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        y_train,
        X_test_scaled,
        test_ids,
        scaler
    )


# PCA visualization

def plot_pca(X_train, y_train):

    print("\nCreating PCA projection...")

    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X_train)

    plt.figure(figsize=(7, 6))

    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=pd.factorize(y_train)[0],
        cmap="tab10",
        alpha=0.7
    )

    plt.xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f} %)"
    )

    plt.ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f} %)"
    )

    plt.title("PCA Projection of the Data")

    legend = plt.legend(
        *scatter.legend_elements(),
        title="Class"
    )

    plt.gca().add_artist(legend)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "pca_projection.png"
        ),
        dpi=150
    )

    plt.close()


# Find the best value of k

def find_best_k(X_train, y_train):

    print("\nSearching for the best k...")

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    k_values = range(K_MIN, K_MAX + 1)

    mean_scores = []
    std_scores = []

    for k in k_values:

        model = KNeighborsClassifier(
            n_neighbors=k
        )

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy"
        )

        mean_scores.append(scores.mean())
        std_scores.append(scores.std())

        print(
            f"k = {k:2d} | "
            f"accuracy = {scores.mean():.4f} "
            f"+/- {scores.std():.4f}"
        )

    best_k = list(k_values)[np.argmax(mean_scores)]

    print(
        f"\nBest k: {best_k} "
        f"(accuracy = {max(mean_scores):.4f})"
    )

    return (
        best_k,
        list(k_values),
        mean_scores,
        std_scores
    )


# Plot the accuracy for each k

def plot_k_curve(
    k_values,
    mean_scores,
    std_scores,
    best_k
):

    plt.figure(figsize=(8, 5))

    plt.plot(
        k_values,
        mean_scores,
        marker="o",
        label="Mean accuracy"
    )

    plt.fill_between(
        k_values,
        np.array(mean_scores) - np.array(std_scores),
        np.array(mean_scores) + np.array(std_scores),
        alpha=0.2
    )

    plt.axvline(
        best_k,
        linestyle="--",
        label=f"Best k = {best_k}"
    )

    plt.xlabel("Number of neighbors (k)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy for Different Values of k")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "k_selection_curve.png"
        ),
        dpi=150
    )

    plt.close()


# Compare different distance metrics

def compare_distances(X_train, y_train, best_k):

    print("\nComparing distance metrics...")

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    distances = [
        "euclidean",
        "manhattan",
        "chebyshev",
        "minkowski"
    ]

    results = {}

    for distance in distances:

        model = KNeighborsClassifier(
            n_neighbors=best_k,
            metric=distance
        )

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy"
        )

        results[distance] = scores.mean()

        print(
            f"{distance:<12} : "
            f"{scores.mean():.4f}"
        )

    return results


# Optimize KNN with GridSearchCV

def tune_knn(X_train, y_train, best_k):

    print("\nOptimizing KNN with GridSearchCV...")

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=3,
        random_state=RANDOM_STATE
    )

    parameters = {
        "n_neighbors": sorted(set([
            max(1, best_k - 4),
            max(1, best_k - 2),
            best_k,
            best_k + 2,
            best_k + 4
        ])),
        "weights": [
            "uniform",
            "distance"
        ],
        "metric": [
            "euclidean",
            "manhattan",
            "minkowski"
        ],
        "p": [1, 2]
    }

    grid = GridSearchCV(
        KNeighborsClassifier(),
        parameters,
        cv=cv,
        scoring="accuracy",
        n_jobs=1
    )

    start = time.time()

    grid.fit(X_train, y_train)

    print(
        f"GridSearch finished in "
        f"{time.time() - start:.1f} seconds."
    )

    print("Best parameters:")
    print(grid.best_params_)

    print(
        f"Best cross-validation accuracy: "
        f"{grid.best_score_:.4f}"
    )

    return grid


# Compare KNN with other models

def compare_models(X_train, y_train, knn):

    print("\nComparing different models...")

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    models = {
        "Optimized KNN": knn,
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE
        ),
        "SVM RBF": SVC(
            kernel="rbf",
            random_state=RANDOM_STATE
        )
    }

    results = []

    for name, model in models.items():

        accuracy = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy"
        )

        f1 = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_weighted"
        )

        results.append({
            "Model": name,
            "Mean Accuracy": accuracy.mean(),
            "Accuracy Std": accuracy.std(),
            "Mean F1-weighted": f1.mean()
        })

        print(
            f"{name:<25} | "
            f"accuracy = {accuracy.mean():.4f} "
            f"+/- {accuracy.std():.4f} | "
            f"F1 = {f1.mean():.4f}"
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "Mean Accuracy",
        ascending=False
    )

    # Plot the results
    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=results_df,
        x="Mean Accuracy",
        y="Model"
    )

    plt.xlim(0, 1)
    plt.title("Model Comparison")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "model_comparison.png"
        ),
        dpi=150
    )

    plt.close()

    return results_df


# Learning curve

def plot_learning_curve(X_train, y_train, model):

    print("\nCreating learning curve...")

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    train_sizes, train_scores, validation_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)

    validation_mean = validation_scores.mean(axis=1)
    validation_std = validation_scores.std(axis=1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        train_sizes,
        train_mean,
        "o-",
        label="Training"
    )

    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15
    )

    plt.plot(
        train_sizes,
        validation_mean,
        "o-",
        label="Validation"
    )

    plt.fill_between(
        train_sizes,
        validation_mean - validation_std,
        validation_mean + validation_std,
        alpha=0.15
    )

    plt.xlabel("Number of training samples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "learning_curve.png"
        ),
        dpi=150
    )

    plt.close()


# Evaluate the final model

def evaluate_model(model, X_train, y_train):

    print("\nEvaluating the final model...")

    y_pred = model.predict(X_train)

    accuracy = accuracy_score(
        y_train,
        y_pred
    )

    f1 = f1_score(
        y_train,
        y_pred,
        average="weighted"
    )

    print(f"\nTraining accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")

    # Classification report
    report = classification_report(
        y_train,
        y_pred
    )

    print("\nClassification report:")
    print(report)

    with open(
        os.path.join(
            OUTPUT_DIR,
            "classification_report.txt"
        ),
        "w"
    ) as file:
        file.write(report)

    # Confusion matrix
    matrix = confusion_matrix(
        y_train,
        y_pred
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "confusion_matrix.png"
        ),
        dpi=150
    )

    plt.close()

    # ROC curves
    if not hasattr(model, "predict_proba"):
        print("The model does not provide probability predictions.")
        return

    classes = np.unique(y_train)

    if len(classes) < 2:
        return

    y_binary = label_binarize(
        y_train,
        classes=classes
    )

    probabilities = model.predict_proba(
        X_train
    )

    plt.figure(figsize=(7, 6))

    for i, current_class in enumerate(classes):

        if y_binary.shape[1] == 1:

            fpr, tpr, _ = roc_curve(
                y_binary[:, 0],
                probabilities[:, 1]
            )

        else:

            fpr, tpr, _ = roc_curve(
                y_binary[:, i],
                probabilities[:, i]
            )

        roc_auc = auc(
            fpr,
            tpr
        )

        plt.plot(
            fpr,
            tpr,
            label=f"Class {current_class} (AUC = {roc_auc:.3f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        "k--",
        alpha=0.5
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "roc_curves.png"
        ),
        dpi=150
    )

    plt.close()


# Save the model

def save_model(model, scaler):

    joblib.dump(
        model,
        os.path.join(
            OUTPUT_DIR,
            "knn_model_final.joblib"
        )
    )

    joblib.dump(
        scaler,
        os.path.join(
            OUTPUT_DIR,
            "scaler.joblib"
        )
    )

    print("\nModel and scaler saved.")


# Generate predictions

def create_submission(model, X_test, test_ids):

    predictions = model.predict(
        X_test
    )

    submission = pd.DataFrame({
        ID: test_ids,
        TARGET: predictions
    })

    path = os.path.join(
        OUTPUT_DIR,
        "submission.csv"
    )

    submission.to_csv(
        path,
        index=False
    )

    print(
        f"\nSubmission file created: {path}"
    )

    print("\nPrediction distribution:")
    print(
        submission[TARGET].value_counts()
    )

    return submission


# Main program

def main():

    start_time = time.time()

    # Load the data
    train, test = load_data()

    # Explore the data
    exploratory_analysis(train)

    # Look for possible outliers
    outliers = find_outliers(train)

    if len(outliers) > 0:
        print("\nExamples of possible outliers:")
        print(outliers.head())

    # Standardize the features
    (
        X_train,
        y_train,
        X_test,
        test_ids,
        scaler
    ) = preprocess(
        train,
        test
    )

    # Visualize the data using PCA
    plot_pca(
        X_train,
        y_train
    )

    # Find the best k
    (
        best_k,
        k_values,
        mean_scores,
        std_scores
    ) = find_best_k(
        X_train,
        y_train
    )

    plot_k_curve(
        k_values,
        mean_scores,
        std_scores,
        best_k
    )

    # Compare distance metrics
    compare_distances(
        X_train,
        y_train,
        best_k
    )

    # Optimize KNN
    grid = tune_knn(
        X_train,
        y_train,
        best_k
    )

    best_model = grid.best_estimator_

    # Compare KNN with other models
    comparison = compare_models(
        X_train,
        y_train,
        best_model
    )

    comparison.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "model_comparison.csv"
        ),
        index=False
    )

    # Plot the learning curve
    plot_learning_curve(
        X_train,
        y_train,
        best_model
    )

    # Evaluate the final model
    evaluate_model(
        best_model,
        X_train,
        y_train
    )

    # Save the model
    save_model(
        best_model,
        scaler
    )

    # Generate predictions for the test set
    create_submission(
        best_model,
        X_test,
        test_ids
    )

    print(
        f"\nPipeline finished in "
        f"{time.time() - start_time:.1f} seconds."
    )


if __name__ == "__main__":
    main()

