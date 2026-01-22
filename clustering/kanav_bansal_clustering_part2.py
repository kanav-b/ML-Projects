#!/usr/bin/env python3
"""
Hierarchical Clustering on Dogs Dataset (Part 2)

- Uses Ward linkage with scipy.cluster.hierarchy
- Saves a truncated dendrogram with 30 leaves (default) and leaf labels = cluster size
- Computes majority-vote clustering error on the k terminal nodes
- Prints EXACTLY: "k=<k>, ERROR=<number>"

Usage:
  python kanav_bansal_clustering_part2.py
  python kanav_bansal_clustering_part2.py --show
  python kanav_bansal_clustering_part2.py --k 30 --x_path dogs_X.npy --y_path dogs_clades.npy
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


# ----------------------------- CLI ----------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Hierarchical clustering on Dogs dataset (Ward linkage).")
    p.add_argument("--x_path", type=str, default="dogs_X.npy", help="path to dogs_X.npy")
    p.add_argument("--y_path", type=str, default="dogs_clades.npy", help="path to dogs_clades.npy")
    p.add_argument("--k", type=int, default=30, help="number of clusters / leaves to cut to (default: 30)")
    p.add_argument("--show", action="store_true", help="show dendrogram window after saving")
    return p.parse_args()


# ----------------------------- Error ----------------------------- #

def clustering_error_majority(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> int:
    """
    Same definition as Part 1:
      For each cluster (1..k): take majority true label among its members,
      misclassified = cluster_size - majority_count. Sum over clusters.
    y_true must be non-negative ints (we encode strings before calling).
    """
    total = 0
    for c in range(1, k + 1):
        members = (y_pred == c)
        if not np.any(members):
            continue
        yc = y_true[members]
        binc = np.bincount(yc)         # works because yc are ints starting at 0
        majority = int(binc.argmax())
        miscls = len(yc) - int(binc[majority])
        total += miscls
    return int(total)


# ----------------------------- Dendrogram helpers ----------------------------- #

def _node_sizes_lookup(Z: np.ndarray, n: int):
    """
    Build a dict mapping node_id -> subtree size (number of original observations).
    For SciPy linkage:
      - Original leaves have ids 0..n-1 (size 1).
      - Merged nodes have ids n..n+Z.shape[0]-1; size is Z[i,3].
    """
    sizes = {i: 1 for i in range(n)}
    for i, row in enumerate(Z):
        sizes[n + i] = int(row[3])
    return sizes


def plot_truncated_dendrogram(Z: np.ndarray, k: int, out_path: Path, show: bool = False):
    """
    Plot a dendrogram truncated to the last p=k leaves.
    Each visible leaf label is the cluster size (number of samples under that node).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = Z.shape[0] + 1
    sizes = _node_sizes_lookup(Z, n)

    def leaf_label_func(node_id: int) -> str:
        return str(int(sizes[int(node_id)]))

    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=k,
        leaf_label_func=leaf_label_func,
        show_leaf_counts=False,   # we supply our own labels
        no_labels=False,
        color_threshold=None,
        ax=ax,
    )
    ax.set_title(f"Dogs Dendrogram (Ward-linkage) — truncated to {k} leaves")
    ax.set_xlabel("Leaf label = cluster size")
    ax.set_ylabel("Distance")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
    if show:
        plt.show()
    plt.close(fig)


# ----------------------------- Main ----------------------------- #

def main():
    args = parse_args()
    k = int(args.k)

    # Load data
    X = np.load(args.x_path, allow_pickle=True)
    y_raw = np.load(args.y_path, allow_pickle=True)

    # Ensure numeric arrays
    X = np.asarray(X, dtype=np.float64)

    # Encode string clade labels -> integer IDs (0..C-1)
    unique_labels, y_enc = np.unique(y_raw, return_inverse=True)
    y = y_enc.astype(np.int64, copy=False)

    # Ward-linkage hierarchical clustering
    Z = linkage(X, method="ward")

    # Save truncated dendrogram with k leaves (leaf labels = cluster sizes)
    out_png = Path("outputs") / "Dogs_dendrogram_truncated.png"
    plot_truncated_dendrogram(Z, k=k, out_path=out_png, show=args.show)

    # Assign samples to k clusters and compute error
    labels = fcluster(Z, t=k, criterion="maxclust")  # cluster labels in 1..k (not necessarily used order)
    err = clustering_error_majority(y_true=y, y_pred=labels, k=k)

    # Print EXACT required line
    print(f"k={k}, ERROR={err}")


if __name__ == "__main__":
    main()