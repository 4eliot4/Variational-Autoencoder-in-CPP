# plot_csv_iterations.py
# Plot points from CSV files: one color per iteration.
# Parameters: START_ITER, END_ITER, STEP, DIMS or PCA.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ======================
# USER PARAMETERS
# ======================

BASE_DIR   = Path("/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/assets")

# Your CSV naming pattern (adjust if needed):
# Example: H_latent_iter_00000.csv
FNAME_FMT  = "H_latent_iter{it:05d}.csv"

START_ITER = 0
END_ITER   = 1000
STEP       = 50

# Choose 3D projection:
USE_PCA_3D = True          # True: PCA->3D per iteration; False: use raw dims
DIMS       = (0, 1, 2)     # only used if USE_PCA_3D=False

# Optional: limit number of points per iteration
MAX_POINTS = None          # e.g. 300, or None for all

POINT_SIZE = 6
ALPHA      = 0.55

# ======================
# HELPERS
# ======================

def load_latent_csv(iteration: int):
    fname = BASE_DIR / FNAME_FMT.format(it=iteration)
    if not fname.exists():
        print(f"[WARN] missing: {fname}")
        return None
    H = np.loadtxt(fname, delimiter=",")
    if H.ndim == 1:
        H = H[None, :]
    return H

def pca_3d(H: np.ndarray):
    # Center
    mu = H.mean(axis=0, keepdims=True)
    Hc = H - mu
    # Covariance
    denom = max(1, Hc.shape[0] - 1)
    C = (Hc.T @ Hc) / denom
    # Eigen decomposition (symmetric)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    W = vecs[:, order[:3]]
    return Hc @ W  # (n,3)

# ======================
# MAIN
# ======================

def main():
    iters = list(range(START_ITER, END_ITER + 1, STEP))
    if len(iters) == 0:
        raise RuntimeError("Empty iteration range: check START/END/STEP")

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.cm.get_cmap("viridis", len(iters))

    loaded = 0
    for k, it in enumerate(iters):
        H = load_latent_csv(it)
        if H is None:
            continue

        if MAX_POINTS is not None and H.shape[0] > MAX_POINTS:
            H = H[:MAX_POINTS, :]

        if USE_PCA_3D:
            P = pca_3d(H)
            x, y, z = P[:, 0], P[:, 1], P[:, 2]
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
        else:
            if max(DIMS) >= H.shape[1]:
                raise ValueError(f"DIMS={DIMS} but latent dim is {H.shape[1]}")
            x, y, z = H[:, DIMS[0]], H[:, DIMS[1]], H[:, DIMS[2]]
            ax.set_xlabel(f"h{DIMS[0]+1}")
            ax.set_ylabel(f"h{DIMS[1]+1}")
            ax.set_zlabel(f"h{DIMS[2]+1}")

        ax.scatter(
            x, y, z,
            s=POINT_SIZE,
            alpha=ALPHA,
            color=cmap(k),
            label=f"it {it}"
        )
        loaded += 1

    if loaded == 0:
        raise RuntimeError("No CSVs loaded — check BASE_DIR and FNAME_FMT")

    title = "Latent points — color=iteration (PCA->3D)" if USE_PCA_3D else "Latent points — color=iteration (raw dims)"
    ax.set_title(title)

    # Legend: if too many iterations, it will be crowded.
    # You can comment this out or only label every N iterations.
    ax.legend(title="Iteration", fontsize=8)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()