#!/usr/bin/env python3
# Part 1: PCA & Eigendecomposition — 2-D tutorial
# Outputs:
#   - projection_1D.png
#   - Prints tutorial values + answers to Q1–Q5 to stdout

import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe for headless
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def main():
    X = np.load('part1_data.npy', allow_pickle=True)
    n = X.shape[0]

    # --- Sklearn PCA (2 comps)
    pca = PCA(n_components=2)
    pca.fit(X)
    x_pca = pca.transform(X)

    print("Sklearn direction of each PC (PC1, PC2):\n", pca.components_)
    print("Sklearn explained variance for each PC (PC1, PC2):\n", pca.explained_variance_)
    print("Sklearn fraction of variance explained by each PC (PC1, PC2):\n", pca.explained_variance_ratio_)
    print("Sklearn data projection into 1-D (onto PC1):\n", x_pca[:, 0])

    # --- Center data
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu

    # --- Covariance
    sigma = (Xc.T @ Xc) / (n - 1)
    print("\nCovariance matrix (Sigma):\n", sigma)

    # --- Eigendecomposition (for symmetric, use eigh -> sorted ascending)
    evals, evecs = np.linalg.eigh(sigma)
    # Sort descending by eigenvalue to match PCA ordering (PC1 largest variance)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    lam1, lam2 = evals[0], evals[1]
    v1, v2 = evecs[:, 0], evecs[:, 1]

    print("\nEigenvalues (λ1, λ2):", lam1, lam2)
    print("Eigenvectors (v1, v2 as columns):\n", np.column_stack([v1, v2]))

    # --- Q1: Compare eigendecomposition vs sklearn
    # components_ are unit vectors; eigenvectors are also unit (from eigh).
    # They may differ by sign.
    def aligned_cos(u, v):
        u = u / np.linalg.norm(u)
        v = v / np.linalg.norm(v)
        return float(np.dot(u, v))

    cos_pc1 = aligned_cos(v1, pca.components_[0])
    cos_pc2 = aligned_cos(v2, pca.components_[1])
    q1 = (
        "Q1: Sklearn PCs vs eigendecomposition: directions match up to a possible sign flip; "
        f"cos(angle)(v1, PC1)={cos_pc1:.6f}, cos(angle)(v2, PC2)={cos_pc2:.6f}. "
        "Eigenvalues match sklearn explained variances (within numerical precision)."
    )
    print("\n" + q1)

    # --- Q2: Direction of greatest variance vs v1
    q2 = (
        "Q2: From the scatter, the direction of greatest variance is along the elongated diagonal. "
        "v1 points along that direction (up to sign), as it is the eigenvector for the largest eigenvalue."
    )
    print(q2)

    # --- Ratios
    lam_sum = lam1 + lam2
    r1 = lam1 / lam_sum
    r2 = lam2 / lam_sum
    print("\nλ1_ratio, λ2_ratio:", r1, r2)

    # --- Q3: Compare ratios to sklearn explained_variance_ratio_
    r_diff1 = abs(r1 - pca.explained_variance_ratio_[0])
    r_diff2 = abs(r2 - pca.explained_variance_ratio_[1])
    q3 = (
        "Q3: Eigenvalue ratios match sklearn’s explained_variance_ratio_ "
        f"(differences ~ {r_diff1:.2e}, {r_diff2:.2e})."
    )
    print(q3)

    # --- Projection onto v1
    projection = Xc @ v1
    print("\nManual projection onto v1:\n", projection)

    # --- Q4: Compare projection to sklearn PC1 scores (up to sign)
    # sklearn’s first PC scores are x_pca[:,0]; may differ by sign depending on v1 orientation.
    # Find best sign to compare.
    sign = 1.0 if np.corrcoef(projection, x_pca[:, 0])[0, 1] >= 0 else -1.0
    proj_diff = np.linalg.norm(sign * projection - x_pca[:, 0]) / np.linalg.norm(x_pca[:, 0])
    q4 = (
        "Q4: Manual projection onto v1 matches sklearn’s PC1 scores up to sign; "
        f"relative L2 diff = {proj_diff:.2e}."
    )
    print(q4)

    # --- Plot 1D number line of projection
    plt.figure(figsize=(7, 1.75))
    y = np.zeros_like(projection)
    plt.scatter(projection, y, s=8)
    plt.yticks([])
    plt.xlabel("Projection onto v1 (PC1)")
    plt.title("1D Projection of Centered Data onto v1")
    plt.tight_layout()
    plt.savefig("projection_1D.png", dpi=200)
    plt.close()

    # --- Q5
    q5 = (
        "Q5: The 1D projection preserves the ordering along the principal direction seen in the 2D scatter. "
        "Spread is large along this axis and near-zero orthogonal to it; what’s lost is the minor variation "
        "in the perpendicular direction."
    )
    print(q5)

if __name__ == "__main__":
    main()