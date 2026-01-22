#!/usr/bin/env python3
"""
K-Means on MNIST subset (from scratch, no sklearn/scipy).

Usage:
  python Kanav_Bansal_clustering_part1.py -k 10
  python Kanav_Bansal_clustering_part1.py -k 11
  python Kanav_Bansal_clustering_part1.py -k 10 --show
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from time import time


def parse_args():
    p = argparse.ArgumentParser(description="K-Means on MNIST subset (from scratch).")
    p.add_argument("-k", type=int, required=True, help="number of clusters")
    p.add_argument("--x_path", type=str, default="MNIST_X_subset.npy",
                   help="path to MNIST_X_subset.npy")
    p.add_argument("--y_path", type=str, default="MNIST_y_subset.npy",
                   help="path to MNIST_y_subset.npy")
    p.add_argument("--max_iters", type=int, default=300, help="maximum K-Means iterations")
    p.add_argument("--tol", type=float, default=1e-4, help="convergence tolerance")
    p.add_argument("--seed", type=int, default=None, help="random seed")
    p.add_argument("--no_progress", action="store_true", help="suppress per-iteration prints")
    p.add_argument("--show", action="store_true", help="show centroid figure after saving")
    return p.parse_args()


def init_centroids_random(X, k, rng):
    n = X.shape[0]
    if k > n:
        raise ValueError(f"k={k} cannot exceed number of samples n={n}")
    idx = rng.choice(n, size=k, replace=False)
    return X[idx].astype(np.float64, copy=True)


def assign_clusters(X, centroids):
    X_sq = np.einsum("ij,ij->i", X, X)[:, None]
    C_sq = np.einsum("ij,ij->i", centroids, centroids)
    distances = X_sq + C_sq[None, :] - 2.0 * (X @ centroids.T)
    return np.argmin(distances, axis=1)


def update_centroids(X, labels, k, rng):
    d = X.shape[1]
    newC = np.zeros((k, d), dtype=np.float64)
    for c in range(k):
        members = (labels == c)
        if np.any(members):
            newC[c] = X[members].mean(axis=0)
        else:
            newC[c] = X[rng.integers(0, X.shape[0])]
    return newC


def kmeans(X, k, max_iters=300, tol=1e-4, seed=None, verbose=True):
    rng = np.random.default_rng(seed)
    centroids = init_centroids_random(X, k, rng)
    for it in range(1, max_iters + 1):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k, rng)
        shifts = np.linalg.norm(new_centroids - centroids, axis=1)
        max_shift = float(shifts.max())
        if verbose:
            print(f"[iter {it}] max centroid shift = {max_shift:.6f}")
        centroids = new_centroids
        if max_shift < tol:
            break
    return centroids, labels


def clustering_error(labels_true, labels_pred, k):
    total_err = 0
    for c in range(k):
        members = (labels_pred == c)
        if not np.any(members):
            continue
        y_c = labels_true[members]
        binc = np.bincount(y_c, minlength=10)
        majority = int(binc.argmax())
        miscls = len(y_c) - int(binc[majority])
        total_err += miscls
    return int(total_err)


def save_centroids_image(centroids, k, out_path, show=False):
    side = 28
    imgs = centroids.reshape(k, side, side)

    fig_w = max(6, min(20, int(1.8 * k)))
    fig, axes = plt.subplots(1, k, figsize=(fig_w, 2.2), squeeze=False)
    for i in range(k):
        ax = axes[0, i]
        ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
    plt.tight_layout(pad=0.1)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()

    # Load data
    X = np.load(args.x_path, allow_pickle=True)
    y = np.load(args.y_path, allow_pickle=True)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).reshape(-1)

    t0 = time()
    centroids, labels = kmeans(
        X, args.k,
        max_iters=args.max_iters,
        tol=args.tol,
        seed=args.seed,
        verbose=not args.no_progress
    )
    t1 = time()

    err = clustering_error(y, labels, args.k)

    out_png = f"outputs/centroids_k{args.k}.png"
    save_centroids_image(centroids, args.k, out_png, show=args.show)

    print(f"k={args.k}, ERROR={err}")
    print(f"[done in {t1 - t0:.2f}s] saved {out_png}")


if __name__ == "__main__":
    main()
    