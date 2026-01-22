#!/usr/bin/env python3
# Part 3: NMF on Dogs SNP dataset
# Output:
#   - dogs_ancestry_summary.tsv
#
# Requirements from prompt:
#  - n_components=5
#  - Normalize rows of W to sum to 1
#  - Average normalized W per clade
#  - Sort rows alphabetically by clade
#  - Label/order ancestry columns by prevalence across clades (each clade weighted equally)
#  - Round to two decimals, exact header format

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

def main():
    X = np.load("dogs_X.npy")              # (1355, 784), non-negative SNP features
    
    try:
        clades = np.load("dogs_clades.npy")
    except ValueError:
        # dogs_clades.npy contains strings (object array) → allow pickle for trusted class data
        clades = np.load("dogs_clades.npy", allow_pickle=True)

    # Ensure 1-D string array
    clades = np.asarray(clades).reshape(-1).astype(str)

    # Factorize
    nmf = NMF(n_components=5, init="nndsvda", random_state=0, max_iter=1000)
    W = nmf.fit_transform(X)   # (1355, 5)
    H = nmf.components_        # (5, 784)  # not directly needed for summary

    # Normalize rows of W to sum to 1 (avoid divide-by-zero)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W_norm = W / row_sums

    # Average normalized W within each clade (each dog contributes equally within its clade)
    df = pd.DataFrame(W_norm, columns=[f"comp{i+1}" for i in range(5)])
    df["clade"] = clades

    per_clade = df.groupby("clade", as_index=False).mean(numeric_only=True)

    # Determine prevalence across clades: average each component across clades (equal weight per clade)
    comp_cols = [c for c in per_clade.columns if c.startswith("comp")]
    comp_means_across_clades = per_clade[comp_cols].mean(axis=0)  # (5,)
    # Order components by prevalence descending
    ordered = list(comp_means_across_clades.sort_values(ascending=False).index)

    # Rename to ancestry1..ancestry5 in that order
    rename_map = {ordered[i]: f"ancestry{i+1}" for i in range(5)}
    per_clade = per_clade.rename(columns=rename_map)

    # Reorder ancestry columns in ancestry1..ancestry5 sequence
    ancestry_cols = [f"ancestry{i+1}" for i in range(5)]
    # Some columns may not yet be renamed if ordered mapping used original comp names:
    # ensure all exist after renaming
    missing = [c for c in ancestry_cols if c not in per_clade.columns]
    if missing:
        # Should not happen, but guard anyway
        raise RuntimeError(f"Missing ancestry columns after renaming: {missing}")

    # Keep only clade + ancestry columns
    per_clade = per_clade[["clade"] + ancestry_cols]

    # Sort rows alphabetically by clade name
    per_clade = per_clade.sort_values("clade", kind="mergesort").reset_index(drop=True)

    # Round to two decimals — IMPORTANT: formatting for TSV
    # We will format as strings with two decimals so the TSV exactly matches expectations.
    out = per_clade.copy()
    for c in ancestry_cols:
        out[c] = out[c].round(2)

    # Save as TSV with exact header
    out.to_csv("dogs_ancestry_summary.tsv", sep="\t", index=False)

if __name__ == "__main__":
    main()