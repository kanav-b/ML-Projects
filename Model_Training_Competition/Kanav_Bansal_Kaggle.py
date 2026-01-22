#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fast PRS baseline aiming for AUC >= 0.71

- Version-proof OneHotEncoder wrapper (sklearn 1.2+)
- Removes ID from *all* feature paths to avoid ColumnTransformer errors
- Sparse-aware pipeline + SelectKBest for speed
- SGDClassifier (logistic loss) with early stopping
- Tiny hyperparam sweep (k x alpha) on a single stratified split
- Outputs submission.csv with: id,breast_cancer
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# --- debug/runtime verification ---
import sys, sklearn
print(">>> sklearn:", sklearn.__version__)
print(">>> python :", sys.executable)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42
TARGET = "breast_cancer"
ID_CANDIDATES = ["id", "ID", "Id"]

# ---------- Utilities ----------

def make_ohe():
    """Return a OneHotEncoder that keeps outputs sparse across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)          # sklearn >= 1.4
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)   # sklearn 1.2–1.3

def infer_columns(df: pd.DataFrame, target_col=None):
    """Treat non-numeric columns (except target) as categorical (ancestry)."""
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in num:
        num.remove(target_col)
    cat = [c for c in df.columns if c not in num and c != target_col]

    # If everything is numeric, try common ancestry names (may be int-coded)
    if not cat:
        guess = [c for c in df.columns if c.lower() in ("ancestry", "race", "population", "ancestry_group")]
        if guess:
            cat = guess
            num = [c for c in num if c not in guess]
    return num, cat

def build_preprocessor(numeric_cols, categorical_cols, k):
    k_eff = min(k, len(numeric_cols)) if numeric_cols else 0
    prs_steps = []
    if numeric_cols:
        prs_steps = [
            ("scale", StandardScaler(with_mean=False)),
            ("kbest", SelectKBest(score_func=f_classif, k=k_eff if k_eff > 0 else "all")),
        ]
    prs_pipe = Pipeline(steps=prs_steps) if prs_steps else "drop"

    transformers = []
    if numeric_cols:
        transformers.append(("prs", prs_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("anc", make_ohe(), categorical_cols))
    if not transformers:
        raise ValueError("No features detected. Check your input columns.")
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0)

def sgd_logreg(alpha, class_weight="balanced"):
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        fit_intercept=True,
        max_iter=2000,
        tol=1e-3,
        early_stopping=True,
        n_iter_no_change=5,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
        class_weight=class_weight,
    )

def tiny_sweep(X_tr, y_tr, X_val, y_val, numeric_cols, categorical_cols, k_list, alpha_list):
    best = {"auc": -1, "alpha": None, "k": None, "prep": None}
    for k in k_list:
        prep = build_preprocessor(numeric_cols, categorical_cols, k)
        for alpha in alpha_list:
            clf = sgd_logreg(alpha)
            pipe = Pipeline([("prep", prep), ("clf", clf)])
            pipe.fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, proba)
            print(f"[trial] k={k:4d}, alpha={alpha:.1e} -> AUC={auc:.5f}")
            if auc > best["auc"]:
                best = {"auc": auc, "alpha": alpha, "k": k, "prep": prep}
    print(f"[best] k={best['k']}, alpha={best['alpha']:.1e}, AUC={best['auc']:.5f}")
    return best

# ---------- Main ----------

def main(args):
    train_path = Path(args.train)
    test_path  = Path(args.test)
    out_path   = Path(args.out)

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    assert TARGET in df_train.columns, f"Missing target '{TARGET}' in train.csv"

    # --- Ensure test has an id to write out ---
    id_col_test = next((c for c in ID_CANDIDATES if c in df_test.columns), None)
    if id_col_test is None:
        df_test = df_test.copy()
        df_test["id"] = np.arange(len(df_test))
        id_col_test = "id"

    # --- Ensure ID is NOT a feature anywhere ---
    # Drop id from train (so it can't leak into numeric/categorical features)
    id_cols_train = [c for c in ID_CANDIDATES if c in df_train.columns]
    if id_cols_train:
        df_train = df_train.drop(columns=id_cols_train)

    # Build X/y
    X = df_train.drop(columns=[TARGET])
    y = df_train[TARGET].astype(float)

    # Just in case: remove id from X if present
    X = X.drop(columns=[c for c in ID_CANDIDATES if c in X.columns], errors="ignore")

    # Split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=args.val_size, random_state=RANDOM_STATE, stratify=y
    )

    # Feature inference AFTER removing id
    numeric_cols, categorical_cols = infer_columns(X)

    # Sweep k and alpha
    k_base = min(max(args.k, 50), max(50, len(numeric_cols)))
    k_list = sorted(set([
        max(200, k_base // 2),
        k_base,
        min(len(numeric_cols), max(400, int(k_base * 1.5))),
    ]))
    alpha_list = [1e-4, 3e-4, 1e-3] if args.alphas is None else [float(a) for a in args.alphas.split(",")]

    best = tiny_sweep(X_tr, y_tr, X_val, y_val, numeric_cols, categorical_cols, k_list, alpha_list)

    # Refit on ALL training data with best settings
    final_clf = sgd_logreg(best["alpha"])
    final_pipe = Pipeline([("prep", best["prep"]), ("clf", final_clf)])
    final_pipe.fit(X, y)

    # Final validation check
    val_pred = final_pipe.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred)
    print(f"[final check] Holdout AUC={val_auc:.5f}")

    # Predict on test (drop test id ONLY at transform time)
    X_test = df_test.drop(columns=[id_col_test], errors="ignore")
    test_proba = final_pipe.predict_proba(X_test)[:, 1]
    test_proba = np.clip(test_proba, 0.0, 1.0)

    # Write submission
    submission = pd.DataFrame({"id": df_test[id_col_test].values, "breast_cancer": test_proba})
    submission = submission.sort_values("id").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(f"[done] wrote {out_path.resolve()}")
    print(submission.head())

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=str, default="train.csv", help="Path to train CSV")
    p.add_argument("--test", type=str, default="test.csv", help="Path to test CSV")
    p.add_argument("--out", type=str, default="submission.csv", help="Output submission CSV path")
    p.add_argument("--k", type=int, default=1500, help="Top-K numeric PRS features to keep (default: 1500)")
    p.add_argument("--val-size", type=float, default=0.2, help="Validation fraction (default: 0.2)")
    p.add_argument("--alphas", type=str, default=None, help="Comma-separated SGD alpha values (e.g., '2e-4,3e-4,5e-4')")
    args = p.parse_args()
    main(args)