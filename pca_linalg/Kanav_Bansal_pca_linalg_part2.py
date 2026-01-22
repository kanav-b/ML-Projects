#!/usr/bin/env python3
# Part 2: PCA on MNIST subset (6000 images, 784 features)
# Outputs:
#   - MNIST_PCA_2D.png
#   - MNIST_reconstructed_2PC.png
#   - MNIST_original.png
#   - MNIST_reconstructed_1_from_coord.png

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def imshow_28x28(vec, path, vmin=0.0, vmax=1.0, title=None):
    plt.figure(figsize=(3,3))
    plt.imshow(vec.reshape(28,28), cmap="gray", vmin=vmin, vmax=vmax)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def main():
    X = np.load("MNIST_X_subset.npy")         # (6000, 784), values typically 0..1
    y = np.load("MNIST_y_subset.npy")         # (6000,), labels 0..9

    # Fit PCA to 2D for visualization + inverse_transform for reconstruction
    pca2 = PCA(n_components=2, random_state=0)
    X2 = pca2.fit_transform(X)

    # --- 2D scatter colored by label
    plt.figure(figsize=(7,6))
    for d in range(10):
        mask = (y == d)
        plt.scatter(X2[mask,0], X2[mask,1], s=6, label=str(d), alpha=0.7)
    plt.legend(markerscale=2, title="Digit")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("MNIST (subset) PCA to 2D")
    plt.tight_layout()
    plt.savefig("MNIST_PCA_2D.png", dpi=250)
    plt.close()

    # --- Reconstruct the first example using only 2 PCs
    x0 = X[0:1, :]                 # shape (1, 784)
    x0_2 = pca2.transform(x0)      # (1,2)
    x0_rec = pca2.inverse_transform(x0_2)  # (1,784)

    # Save original and reconstructed
    imshow_28x28(X[0], "MNIST_original.png", title="Original")
    imshow_28x28(x0_rec[0], "MNIST_reconstructed_2PC.png", title="Reconstructed (2 PCs)")

    # --- Generate a new "1" from a selected 2D coordinate
    # Strategy: choose a point near the centroid of the '1' cluster but not exactly a data point.
    one_coords = X2[y == 1]
    centroid = one_coords.mean(axis=0)
    # Nudge a bit so it's not exactly a data point (manual choice)
    chosen_2d = centroid + np.array([0.5, -0.5])  # adjust freely if you want a different style
    # Map this 2D point back to 784-D using inverse_transform
    gen_vec = pca2.inverse_transform(chosen_2d.reshape(1, -1))
    imshow_28x28(gen_vec[0], "MNIST_reconstructed_1_from_coord.png",
                 title="Generated from chosen 2D coord (≈'1')")

if __name__ == "__main__":
    main()