# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def strip_feature_prefix(feature_name: str) -> str:
    """Strip 'rank_' or 'match_' prefix from feature names."""
    stripped = re.sub(r"^(rank_|match_)", "", feature_name)
    return stripped


def save_feature_importance(importance_frames, output_file_path):
    """Save feature importance to CSV file."""
    try:
        imp_all = pd.concat(importance_frames, axis=0)
        imp_mean = (
            imp_all.groupby("feature")["gain"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        imp_mean.to_csv(output_file_path, index=False)
        logging.info(f"Feature importance saved to {output_file_path}")

        logging.info("Mean feature importance (gain):")
        for _, row in imp_mean.head(10).iterrows():
            logging.info(f"  {row['feature']}: {row['gain']:.4f}")

        return imp_mean
    except Exception as e:
        logging.error(f"Error saving feature importance: {e}")
        return None


def perform_cross_validation(
    file_path,
    target_col,
    drop_cols,
    folds,
    seed,
    max_rounds,
    early_stop,
    learning_rate,
    output_model_file,
    output_importance_file,
):
    """Load data and perform stratified cross-validation with LightGBM."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    except FileNotFoundError:
        logging.error(f"Input file '{file_path}' not found.")
        raise FileNotFoundError(f"Input file '{file_path}' not found.")

    # Set random seed
    np.random.seed(seed)

    # --- Data Cleaning ---
    # Drop columns where every value is identical (not informative)
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        logging.info(f"Dropping {len(constant_cols)} constant columns")

    # Drop user-specified identifier columns
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    if cols_to_drop:
        logging.info(f"Dropping ID columns: {cols_to_drop}")

    # Assemble feature list
    feature_cols = df.columns.difference(
        constant_cols + cols_to_drop + [target_col]
    ).tolist()

    # Apply strip_feature_prefix to all feature columns
    stripped_feature_mapping = {}
    for original_col in feature_cols:
        stripped_name = strip_feature_prefix(original_col)
        stripped_feature_mapping[original_col] = stripped_name

    # Rename columns in dataframe to use stripped names
    df_renamed_cols = {}
    for col in df.columns:
        if col in feature_cols:
            df_renamed_cols[col] = stripped_feature_mapping[col]
        else:
            df_renamed_cols[col] = col

    df = df.rename(columns=df_renamed_cols)
    feature_cols = [stripped_feature_mapping[col] for col in feature_cols]

    # --- Handle Categorical Variables ---
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c in feature_cols]
    label_encoders = {}

    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        label_encoders[c] = le

    # LightGBM expects categorical feature indices
    categorical_feature_idx = [feature_cols.index(c) for c in cat_cols]

    # --- Prepare X, y ---
    X = df[feature_cols]
    y = df[target_col].astype(int)

    # Store original feature names (now stripped) for later use
    original_feature_names = X.columns.tolist()
    # Rename columns of X to feature_i to avoid issues with special characters
    X.columns = [f"feature_{i}" for i in range(len(X.columns))]
    feature_name_mapping = dict(zip(X.columns, original_feature_names))

    # --- Stratified K-Fold Cross-Validation ---
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    oof_pred = np.zeros(len(df))
    models = []
    best_iterations = []
    importance_frames = []

    # Lists to store metrics for each fold
    fold_aucs = []
    fold_accs = []

    logging.info(f"Performing {folds}-Fold Stratified Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logging.info(f"Training Fold {fold}/{folds}")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        lgb_train = lgb.Dataset(
            X_train,
            y_train,
            categorical_feature=categorical_feature_idx,
            free_raw_data=False,
        )
        lgb_val = lgb.Dataset(
            X_val,
            y_val,
            categorical_feature=categorical_feature_idx,
            reference=lgb_train,
            free_raw_data=False,
        )

        params = dict(
            objective="binary",
            boosting_type="gbdt",
            learning_rate=learning_rate,
            num_leaves=10,
            max_depth=3,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            metric="auc",
            seed=seed,
            verbose=-1,
        )

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stop),
            lgb.log_evaluation(period=100),
        ]

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=max_rounds,
            valid_sets=[lgb_train, lgb_val],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        best_iterations.append(model.best_iteration)
        models.append(model)

        oof_pred[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)

        auc = roc_auc_score(y_val, oof_pred[val_idx])
        acc = accuracy_score(y_val, (oof_pred[val_idx] > 0.5).astype(int))

        fold_aucs.append(auc)
        fold_accs.append(acc)

        logging.info(f"Fold {fold}: AUC = {auc:.4f}, ACC = {acc:.4f}")

        # Store feature importance for this fold
        imp_df = pd.DataFrame(
            {
                "feature": original_feature_names,
                "gain": model.feature_importance(importance_type="gain"),
                "fold": fold,
            }
        )
        importance_frames.append(imp_df)

    # --- Output Results ---
    overall_auc = roc_auc_score(y, oof_pred)
    overall_acc = accuracy_score(y, (oof_pred > 0.5).astype(int))

    print("\n" + "-" * 60)
    print(f"{'Cross-Validation Results ({}-Fold)':^60}".format(folds))
    print("-" * 60)
    print(f"{'Metric':<18} | {'Mean':<18} | {'Std Dev':<18}")
    print("-" * 60)
    print(
        f"{'Accuracy':<18} | {np.mean(fold_accs):<18.4f} | {np.std(fold_accs):<18.4f}"
    )
    print(f"{'ROC AUC':<18} | {np.mean(fold_aucs):<18.4f} | {np.std(fold_aucs):<18.4f}")
    print("-" * 60)
    print(f"Overall CV AUC: {overall_auc:.4f} • ACC: {overall_acc:.4f}")
    print("-" * 60)

    # --- Feature Importance ---
    imp_mean = save_feature_importance(importance_frames, output_importance_file)
    if imp_mean is not None:
        # Select only non-zero importance features
        final_features = imp_mean[imp_mean["gain"] > 0]["feature"].tolist()
        logging.info(
            f"Selected {len(final_features)} features with non-zero importance"
        )

        # --- Train Final Model ---
        final_boost_rounds = int(np.mean(best_iterations))
        logging.info(
            f"Training final model on all data for {final_boost_rounds} rounds"
        )

        # Create X_final with only the selected features
        final_feature_cols = []
        for orig_name in final_features:
            for renamed_col, mapped_orig in feature_name_mapping.items():
                if mapped_orig == orig_name:
                    final_feature_cols.append(renamed_col)
                    break

        X_final = X[final_feature_cols]
        full_dataset = lgb.Dataset(
            X_final, y, categorical_feature=categorical_feature_idx
        )
        final_model = lgb.train(
            params, full_dataset, num_boost_round=final_boost_rounds
        )

        # Export model
        model_json = final_model.dump_model()

        # Replace feature names in the model JSON with original names
        model_json_str = json.dumps(model_json)
        for renamed_feature, original_feature in feature_name_mapping.items():
            escaped_renamed = re.escape(renamed_feature)
            model_json_str = re.sub(
                rf'"{escaped_renamed}"', f'"{original_feature}"', model_json_str
            )

        model_json = json.loads(model_json_str)

        try:
            out_path = Path(output_model_file)
            with out_path.open("w") as f:
                json.dump(model_json, f)
            logging.info(f"Model exported to {out_path.resolve()}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")

    logging.info("Training completed successfully!")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Perform stratified cross-validation with LightGBM binary classifier and save model."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="relevance_label",
        help="Name of the target column (default: %(default)s)",
    )
    parser.add_argument(
        "--drop_cols",
        nargs="+",
        default=["query_id", "doc_id", "relevance_score"],
        help="Columns to drop as identifiers (default: %(default)s)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: %(default)s)"
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=1000,
        help="Max boosting rounds (default: %(default)s)",
    )
    parser.add_argument(
        "--early_stop",
        type=int,
        default=50,
        help="Early-stopping patience (default: %(default)s)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.05,
        help="Learning rate for LightGBM (default: %(default)s)",
    )
    parser.add_argument(
        "--output_model_file",
        type=str,
        default=None,
        help="Path to save the model JSON file. If not provided, defaults to '[input_file_basename]_lightgbm_model.json'.",
    )
    parser.add_argument(
        "--output_importance_file",
        type=str,
        default=None,
        help="Path to save the feature importance CSV file. If not provided, defaults to '[input_file_basename]_feature_importance.csv'.",
    )

    args = parser.parse_args()

    # Set default output file paths if not provided
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]

    output_model_file = args.output_model_file
    if output_model_file is None:
        output_model_file = f"{base_name}_lightgbm_model.json"

    output_importance_file = args.output_importance_file
    if output_importance_file is None:
        output_importance_file = f"{base_name}_feature_importance.csv"

    # Run the cross-validation process
    perform_cross_validation(
        file_path=args.input_file,
        target_col=args.target,
        drop_cols=args.drop_cols,
        folds=args.folds,
        seed=args.seed,
        max_rounds=args.max_rounds,
        early_stop=args.early_stop,
        learning_rate=args.learning_rate,
        output_model_file=output_model_file,
        output_importance_file=output_importance_file,
    )
