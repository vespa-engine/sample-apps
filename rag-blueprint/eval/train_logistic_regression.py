# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    roc_auc_score,
    average_precision_score,
)
import logging
import os
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def save_coefficients_expression(model, features, intercept, scaler, output_file_path):
    """Saves the model coefficients as a linear expression that accounts for standardization."""
    try:
        # Transform coefficients to work with original (unscaled) features
        # For standardized features: z = (x - mean) / std
        # So the original expression: coef * z + intercept
        # Becomes: (coef / std) * x + (intercept - coef * mean / std)

        transformed_coefs = model.coef_[0] / scaler.scale_
        transformed_intercept = intercept - np.sum(
            model.coef_[0] * scaler.mean_ / scaler.scale_
        )

        # Create expression for original (unscaled) features
        expression_parts = [f"{transformed_intercept:.6f}"]
        for feature, coef in zip(features, transformed_coefs):
            expression_parts.append(f"{coef:+.6f}*{feature}")

        expression = "".join(expression_parts)

        # Also save scaling parameters for reference
        scaling_info = {
            "feature_means": dict(zip(features, scaler.mean_)),
            "feature_stds": dict(zip(features, scaler.scale_)),
            "original_coefficients": dict(zip(features, model.coef_[0])),
            "original_intercept": float(intercept),
            "transformed_coefficients": dict(zip(features, transformed_coefs)),
            "transformed_intercept": float(transformed_intercept),
        }

        # Save the expression
        with open(output_file_path, "w") as f:
            f.write("# Linear expression for original (unscaled) features:\n")
            f.write(expression + "\n\n")
            f.write("# Scaling and coefficient information:\n")
            f.write(json.dumps(scaling_info, indent=2))

        logging.info(f"Model coefficients expression saved to {output_file_path}")
        logging.info(
            "Expression works with original feature values (no scaling needed)"
        )

    except Exception as e:
        logging.error(f"Error saving coefficients expression: {e}")


def perform_cross_validation(file_path, output_coef_file):
    """Loads data, applies standardization, and performs 5-fold stratified cross-validation."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error(f"Input file '{file_path}' not found.")
        raise FileNotFoundError(f"Input file '{file_path}' not found.")

    # Define and drop irrelevant columns
    columns_to_drop = [
        "doc_id",
        "query_id",
        "relevance_score",
    ]
    df = df.drop(columns=columns_to_drop)
    # Convert target variable to binary (0/1)
    df["relevance_label"] = df["relevance_label"].astype(int)

    # Define features (X) and target (y)
    X = df.drop(columns=["relevance_label"])
    features = X.columns.tolist()  # Store feature names for later use
    y = df["relevance_label"]

    # Display feature statistics before scaling
    print("\nFeature Statistics (before scaling):")
    print("-" * 60)
    print(f"{'Feature':<35} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10}")
    print("-" * 60)
    for feature in features:
        stats = X[feature].describe()
        print(
            f"{feature:<35} | {stats['mean']:<10.4f} | {stats['std']:<10.4f} | {stats['min']:<10.4f} | {stats['max']:<10.4f}"
        )
    print("-" * 60)

    # Initialize StandardScaler
    scaler = StandardScaler()

    # --- Stratified K-Fold Cross-Validation ---
    N_SPLITS = 5
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    model = LogisticRegression(random_state=42)

    # Lists to store metrics for each fold
    accuracies, precisions, recalls, f1_scores = [], [], [], []
    log_losses, roc_aucs, avg_precisions = [], [], []

    logging.info(
        f"Performing {N_SPLITS}-Fold Stratified Cross-Validation with standardization...\n"
    )

    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        # Split the data for the current fold
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Fit scaler on training data and transform both train and test
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train the model on scaled data
        model.fit(X_train_scaled, y_train)

        # Make predictions on scaled test data
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Calculate and store metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
        log_losses.append(log_loss(y_test, y_pred_proba))
        roc_aucs.append(roc_auc_score(y_test, y_pred_proba))
        avg_precisions.append(average_precision_score(y_test, y_pred_proba))

        logging.info(
            f"Fold {fold}: Acc = {accuracies[-1]:.4f}, F1 = {f1_scores[-1]:.4f}, ROC AUC = {roc_aucs[-1]:.4f}, Avg Prec = {avg_precisions[-1]:.4f}"
        )

    # --- Output Results ---
    print("\n" + "-" * 60)
    print(f"{'Cross-Validation Results (5-Fold, Standardized)':^60}")
    print("-" * 60)
    print(f"{'Metric':<18} | {'Mean':<18} | {'Std Dev':<18}")
    print("-" * 60)
    print(
        f"{'Accuracy':<18} | {np.mean(accuracies):<18.4f} | {np.std(accuracies):<18.4f}"
    )
    print(
        f"{'Precision':<18} | {np.mean(precisions):<18.4f} | {np.std(precisions):<18.4f}"
    )
    print(f"{'Recall':<18} | {np.mean(recalls):<18.4f} | {np.std(recalls):<18.4f}")
    print(
        f"{'F1-Score':<18} | {np.mean(f1_scores):<18.4f} | {np.std(f1_scores):<18.4f}"
    )
    print(
        f"{'Log Loss':<18} | {np.mean(log_losses):<18.4f} | {np.std(log_losses):<18.4f}"
    )
    print(f"{'ROC AUC':<18} | {np.mean(roc_aucs):<18.4f} | {np.std(roc_aucs):<18.4f}")
    print(
        f"{'Avg Precision':<18} | {np.mean(avg_precisions):<18.4f} | {np.std(avg_precisions):<18.4f}"
    )
    print("-" * 60)

    # --- Model Coefficients ---
    # Retrain on full standardized data to get final coefficients
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)

    print("\nModel Coefficients (on standardized features):")
    print("-" * 50)
    for feature, coef in zip(features, model.coef_[0]):
        print(f"{feature:<35}: {coef:.6f}")
    print(f"{'Intercept':<35}: {model.intercept_[0]:.6f}")
    print("-" * 50)

    # Show transformed coefficients for original features
    transformed_coefs = model.coef_[0] / scaler.scale_
    transformed_intercept = model.intercept_[0] - np.sum(
        model.coef_[0] * scaler.mean_ / scaler.scale_
    )

    print("\nTransformed Coefficients (for original unscaled features):")
    print("-" * 50)
    for feature, coef in zip(features, transformed_coefs):
        print(f"{feature:<35}: {coef:.6f}")
    print(f"{'Intercept':<35}: {transformed_intercept:.6f}")
    print("-" * 50)

    # Save coefficients expression to file
    if output_coef_file:
        save_coefficients_expression(
            model, features, model.intercept_[0], scaler, output_coef_file
        )


if __name__ == "__main__":
    # Set up argument Parser
    parser = argparse.ArgumentParser(
        description="Perform 5-fold stratified cross-validation on a logistic regression model with standardization and save coefficients."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="output/Vespa-training-data_matchfeatures-firstphase_20250619_095907.csv",
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output_coef_file",
        type=str,
        default=None,  # Default to None, will be constructed if not provided
        help="Path to save the model coefficients expression .txt file. If not provided, it defaults to '[input_file_basename]_coefficients.txt'.",
    )
    args = parser.parse_args()

    output_coef_file_path = args.output_coef_file
    if output_coef_file_path is None:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        output_coef_file_path = f"{base_name}_logreg_coefficients.txt"

    # Run the cross-validation process
    perform_cross_validation(args.input_file, output_coef_file_path)
