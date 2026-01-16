import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============================================================
# USER CONFIG
# ============================================================

BASE_DIR = Path("/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/assets")

START_ITER = 0
END_ITER   = 1000
STEP       = 50          # must match your CSV naming
USE_PCA_3D = True

# visual tuning
POINT_SIZE = 6
ALPHA_BASE = 0.35        # transparency for all points

# ============================================================
# UTILITIES
# ============================================================

def shade_color(rgb, factor):
    """
    Darken/lighten a base RGB color.
    factor in [0,1]: 0=white, 1=original color
    """
    return tuple(factor * c + (1.0 - factor) for c in rgb)

def pca_fit_3d(H):
    """Fit PCA(3) on H and return (mean, W)."""
    mu = H.mean(axis=0, keepdims=True)
    Hc = H - mu
    C = (Hc.T @ Hc) / max(1, (Hc.shape[0] - 1))
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    W = vecs[:, order[:3]]
    return mu, W

def pca_project(H, mu, W):
    return (H - mu) @ W

def load_latent_digit_iter(digit, iteration):
    """
    Load CSV:
      assets/nb_X/H_latent_iter_X_YYYYY.csv
    """
    fname = BASE_DIR / f"nb_{digit}" / f"H_latent_iter_{digit}_{iteration:05d}.csv"
    if not fname.exists():
        return None
    H = np.loadtxt(fname, delimiter=",")
    if H.ndim == 1:
        H = H[None, :]
    return H

# ============================================================
# MAIN
# ============================================================

def main():
    iters = list(range(START_ITER, END_ITER + 1, STEP))
    n_iters = len(iters)

    H_all = []
    y_all = []
    t_all = []

    # --------------------------------------------------------
    # 1) LOAD ALL LATENTS (ALL DIGITS, ALL ITERATIONS)
    # --------------------------------------------------------
    missing = 0
    for k, it in enumerate(iters):
        for d in range(10):
            H = load_latent_digit_iter(d, it)
            if H is None:
                missing += 1
                continue
            H_all.append(H)
            y_all.append(np.full((H.shape[0],), d))
            t_all.append(np.full((H.shape[0],), k))

    if len(H_all) == 0:
        raise RuntimeError("No CSVs loaded — check paths and iteration range")

    if missing > 0:
        print(f"[INFO] Missing CSV files skipped: {missing}")

    H_all = np.vstack(H_all)
    y_all = np.concatenate(y_all)
    t_all = np.concatenate(t_all)

    # --------------------------------------------------------
    # 2) PROJECT TO 3D (ONE PCA BASIS)
    # --------------------------------------------------------
    if USE_PCA_3D:
        mu, W = pca_fit_3d(H_all)
        P = pca_project(H_all, mu, W)
        labels = ("PC1", "PC2", "PC3")
        title = "Latent space — PCA(3) — color=digit, shade=iteration"
    else:
        P = H_all[:, :3]
        labels = ("h1", "h2", "h3")
        title = "Latent space — raw latent dims — color=digit, shade=iteration"

    # --------------------------------------------------------
    # 3) PLOT (ALL ITERATIONS ON SAME AXES)
    # --------------------------------------------------------
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    base_cmap = plt.cm.get_cmap("tab10", 10)

    for d in range(10):
        base_color = base_cmap(d)[:3]
        mask_d = (y_all == d)

        for k in range(n_iters):
            mask = mask_d & (t_all == k)
            if not np.any(mask):
                continue

            # shade encodes training time
            shade = 0.25 + 0.75 * (k / (n_iters - 1))
            color = shade_color(base_color, shade)

            ax.scatter(
                P[mask, 0],
                P[mask, 1],
                P[mask, 2],
                s=POINT_SIZE,
                alpha=ALPHA_BASE,
                color=color,
                label=str(d) if k == n_iters - 1 else None
            )

    ax.set_title(title)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.legend(title="Digit")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()