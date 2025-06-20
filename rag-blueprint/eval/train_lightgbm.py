# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import re

# ------------------------------------------------------------------
# 1. CLI arguments
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="LightGBM binary classifier trainer")
parser.add_argument(
    "--data",
    required=True,
    help="Path to CSV file",
)
parser.add_argument(
    "--target",
    default="relevance_label",
    help="Name of the target column (default: %(default)s)",
)
parser.add_argument(
    "--drop-cols",
    nargs="+",
    default=[
        "query_id",
        "doc_id",
        "relevance_score",
    ],  # `relevance_score` is the random score
    help="Columns to drop as identifiers (default: %(default)s)",
)
parser.add_argument("--folds", type=int, default=5, help="K for K-fold CV")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--max-rounds", type=int, default=1000, help="Max boosting rounds")
parser.add_argument(
    "--early-stop", type=int, default=50, help="Early-stopping patience"
)
parser.add_argument(
    "--learning-rate", type=float, default=0.05, help="Learning rate for LightGBM"
)
parser.add_argument(
    "--out-json",
    default="lightgbm_model.json",
    help="Filename to save the final model JSON",
)
args = parser.parse_args()

np.random.seed(args.seed)

# ------------------------------------------------------------------
# 2. Load data
# ------------------------------------------------------------------
df = pd.read_csv(args.data)
print(
    f"[{datetime.now():%H:%M:%S}] Loaded {df.shape[0]:,} rows × {df.shape[1]:,} columns"
)

# ------------------------------------------------------------------
# 3. Basic cleaning / feature pruning
# ------------------------------------------------------------------
# a) Specify explicitly which columns to use
use_cols = []
# use_cols = [
#     "rank_nativeProximity",
#     "match_max_chunk_text_scores",
#     "rank_firstPhase",
#     "rank_nativeFieldMatch",
#     "rank_nativeRank",
#     "match_avg_top_3_chunk_text_scores",
#     "match_avg_top_3_chunk_sim_scores",
#     "rank_elementCompleteness(chunks).completeness",
#     "match_closeness(chunk_embeddings)",
#     "rank_elementCompleteness(chunks).queryCompleteness",
#     "rank_bm25(chunks)",
#     "match_bm25(chunks)",
#     "rank_elementSimilarity(chunks)",
#     "match_modified_freshness",
#     "match_max_chunk_sim_scores",
# ]

# a) Drop columns where every value is identical (not informative)
constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
if constant_cols:
    print(f"   ✂ Dropping {len(constant_cols)} constant columns")

# b) Drop user-specified identifier columns
cols_to_drop = [c for c in args.drop_cols if c in df.columns]
if cols_to_drop:
    print(f"   ✂ Dropping ID columns: {cols_to_drop}")


def strip_feature_prefix(feature_name: str) -> str:
    """Strip 'rank_' or 'match_' prefix from feature names."""
    stripped = re.sub(r"^(rank_|match_)", "", feature_name)
    return stripped


# c) Assemble feature list
if not use_cols:
    feature_cols = df.columns.difference(
        constant_cols + cols_to_drop + [args.target]
    ).tolist()
else:
    feature_cols = [c for c in use_cols if c in df.columns]

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

# Update feature_cols to use stripped names
feature_cols = [stripped_feature_mapping[col] for col in feature_cols]

# ------------------------------------------------------------------
# 4. Handle categorical variables
# ------------------------------------------------------------------
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
cat_cols = [c for c in cat_cols if c in feature_cols]  # only keep relevant ones
label_encoders = {}  # keep encoders to transform test data if needed

for c in cat_cols:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c].astype(str))
    label_encoders[c] = le

# LightGBM expects categorical feature *indices*
categorical_feature_idx = [feature_cols.index(c) for c in cat_cols]

# ------------------------------------------------------------------
# 5. Prepare X, y
# ------------------------------------------------------------------
X = df[feature_cols]
y = df[args.target].astype(int)  # ensure 0/1 integers

# Store original feature names (now stripped) for later use
original_feature_names = X.columns.tolist()
# Rename columns of X to feature_i to avoid issues with special characters
X.columns = [f"feature_{i}" for i in range(len(X.columns))]
# Store the mapping for later use (now using stripped names)
feature_name_mapping = dict(zip(X.columns, original_feature_names))

# ------------------------------------------------------------------
# 6. Stratified K-fold cross-validation
# ------------------------------------------------------------------
skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

oof_pred = np.zeros(len(df))
models = []
best_iterations = []

print("\n🚀 Starting cross-validation")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n── Fold {fold}/{args.folds} ──────────────────────────────────")
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
        learning_rate=args.learning_rate,
        num_leaves=10,
        max_depth=3,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        metric="auc",
        seed=args.seed,
        verbose=-1,
    )
    callbacks = [
        lgb.early_stopping(stopping_rounds=args.early_stop),
        lgb.log_evaluation(period=100),
    ]
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=args.max_rounds,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    best_iterations.append(model.best_iteration)
    models.append(model)

    oof_pred[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)

    auc = roc_auc_score(y_val, oof_pred[val_idx])
    acc = accuracy_score(y_val, (oof_pred[val_idx] > 0.5).astype(int))
    print(f"   Fold-{fold} AUC: {auc:.4f} • ACC: {acc:.4f}")

# Overall CV performance
overall_auc = roc_auc_score(y, oof_pred)
overall_acc = accuracy_score(y, (oof_pred > 0.5).astype(int))
print(f"\n🏁 Overall CV AUC: {overall_auc:.4f} • ACC: {overall_acc:.4f}")

# ------------------------------------------------------------------
# 7. Aggregate feature importance
# ------------------------------------------------------------------
importance_frames = []
for i, m in enumerate(models, 1):
    imp_df = pd.DataFrame(
        {
            "feature": original_feature_names,  # Use original_feature_names instead of feature_cols
            "gain": m.feature_importance(importance_type="gain"),
            "fold": i,
        }
    )
    importance_frames.append(imp_df)

imp_all = pd.concat(importance_frames, axis=0)
imp_mean = (
    imp_all.groupby("feature")["gain"].mean().sort_values(ascending=False).reset_index()
)

print("\n🔑 Mean feature importance (gain):")
print(imp_mean.to_string(index=False))
# Save feature importance to CSV
imp_mean.to_csv("feature_importance.csv", index=False)
print("✅ Feature importance saved to feature_importance.csv")

# Select only non-zero importance features
final_features = imp_mean[imp_mean["gain"] > 0]["feature"].tolist()
print(f"\n📊 Selected {len(final_features)} features with non-zero importance:")
print("\n".join(final_features))

# final_features already contains the original feature names, no need to map
print(f"\nFinal features to use: {final_features}\n")

# ------------------------------------------------------------------
# 8. Train final model on full data & export
# ------------------------------------------------------------------
final_boost_rounds = int(np.mean(best_iterations))
print(f"\n📦 Training final model on all data for {final_boost_rounds} rounds")

# Create X_final with only the selected features
# Need to map final_features back to the renamed column names in X
final_feature_cols = []
for orig_name in final_features:
    # Find the corresponding renamed column in X
    for renamed_col, mapped_orig in feature_name_mapping.items():
        if mapped_orig == orig_name:
            final_feature_cols.append(renamed_col)
            break

X_final = X[final_feature_cols]
full_dataset = lgb.Dataset(X_final, y, categorical_feature=categorical_feature_idx)
final_model = lgb.train(params, full_dataset, num_boost_round=final_boost_rounds)

model_json = final_model.dump_model()

# Replace feature names in the model JSON with original names
model_json_str = json.dumps(model_json)
for renamed_feature, original_feature in feature_name_mapping.items():
    escaped_renamed = re.escape(renamed_feature)
    model_json_str = re.sub(
        rf'"{escaped_renamed}"', f'"{original_feature}"', model_json_str
    )

model_json = json.loads(model_json_str)

out_path = Path(args.out_json)
with out_path.open("w") as f:
    json.dump(model_json, f)
print(f"✅ Model exported to {out_path.resolve()}")

# ------------------------------------------------------------------
# 9. (Optional) Save OOF predictions for stacking / inspection
# ------------------------------------------------------------------
# df["oof_pred"] = oof_pred
# df.to_csv("train_oof_preds.csv", index=False)

print("\n🎉 All done!")
