import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

BASE_DIR = Path("/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/assets")

# Pick ONE snapshot iteration to plot (no iteration differentiation)
ITER_TO_PLOT = 0  # will look for ..._01000.csv

USE_PCA_3D = True

def pca_3d(H):
    # Center
    Hc = H - H.mean(axis=0, keepdims=True)
    # Covariance
    C = (Hc.T @ Hc) / max(1, (Hc.shape[0] - 1))
    # Eigen decomposition (symmetric)
    vals, vecs = np.linalg.eigh(C)
    # Take top 3 eigenvectors
    order = np.argsort(vals)[::-1]
    W = vecs[:, order[:3]]       # (latent_dim, 3)
    Z = Hc @ W                   # (n, 3)
    return Z

def load_latent_digit(digit, iteration):
    fname = BASE_DIR / f"nb_{digit}" / f"H_latent_iter_{digit}_{iteration:05d}.csv"
    if not fname.exists():
        print(f"[WARN] missing: {fname}")
        return None
    H = np.loadtxt(fname, delimiter=",")
    if H.ndim == 1:
        H = H[None, :]
    return H

def main():
    H_list = []
    labels_list = []

    for d in range(10):
        H = load_latent_digit(d, ITER_TO_PLOT)
        if H is None:
            continue
        H_list.append(H)
        labels_list.append(np.full((H.shape[0],), d, dtype=np.int32))

    if len(H_list) == 0:
        raise RuntimeError("No CSV files were loaded. Check ITER_TO_PLOT and paths.")

    H_all = np.vstack(H_list)                 # (N_total, latent_dim)
    y_all = np.concatenate(labels_list)       # (N_total,)

    # Project to 3D
    if USE_PCA_3D:
        P = pca_3d(H_all)                     # (N_total, 3)
        axis_names = ("PC1", "PC2", "PC3")
    else:
        P = H_all[:, :3]
        axis_names = ("h1", "h2", "h3")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.cm.get_cmap("tab10", 10)

    for d in range(10):
        mask = (y_all == d)
        if not np.any(mask):
            continue
        ax.scatter(
            P[mask, 0], P[mask, 1], P[mask, 2],
            s=10, alpha=0.6, color=cmap(d), label=str(d)
        )

    ax.set_title(f"Latent space (all digits) at iter {ITER_TO_PLOT:05d} - PCA to 3D" if USE_PCA_3D
                 else f"Latent space (all digits) at iter {ITER_TO_PLOT:05d} - first 3 dims")
    ax.set_xlabel(axis_names[0])
    ax.set_ylabel(axis_names[1])
    ax.set_zlabel(axis_names[2])
    ax.legend(title="Digit")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()