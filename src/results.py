import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D)

# ======= USER PARAMETERS =======

# Folder where your CSVs are saved
BASE_DIR = Path("/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/assets")

# First iteration to plot
START_ITER = 35000          # e.g. 0 or 500 or 2500

# Step between iterations (this is your "modulo", e.g. 500, 1000, 2500)
STEP = 5000             # change to 500, 1000, ... as you like

# How many different iterations to show
NUM_SNAPSHOTS = 1       # e.g. 6 snapshots: 0, 2500, 5000, 7500, 10000, 12500

# Latent dims to plot:
#   - (0, 1)  -> 2D plot of h1 vs h2
#   - (0, 1, 2) -> 3D plot of h1 vs h2 vs h3
DIMS = (0, 1, 2)

# Optional: limit max points per snapshot to avoid huge scatter
MAX_POINTS_PER_SNAPSHOT = 300

# ================================


def load_latent_csv(iteration):
    """Load H_latent_iter_XXXXX.csv for a given iteration."""
    fname = BASE_DIR / f"H_latent_iter_{iteration:05d}.csv"
    if not fname.exists():
        print(f"[WARN] File not found: {fname}")
        return None
    H = np.loadtxt(fname, delimiter=",")
    if H.ndim == 1:
        # Single row corner case: shape (D,) -> make it (1, D)
        H = H[None, :]
    print(f"Loaded {fname.name}, shape={H.shape}")
    return H


def main():
    iters = [START_ITER + k * STEP for k in range(NUM_SNAPSHOTS)]

    # Choose 2D vs 3D depending on how many dims we asked for
    if len(DIMS) == 2:
        fig, ax = plt.subplots()
    elif len(DIMS) == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        raise ValueError("DIMS must have length 2 or 3.")

    # Colormap for different iterations
    cmap = plt.cm.get_cmap("viridis", len(iters))

    for idx, it in enumerate(iters):
        H = load_latent_csv(it)
        if H is None:
            continue

        # Optionally subsample points
        if H.shape[0] > MAX_POINTS_PER_SNAPSHOT:
            H = H[:MAX_POINTS_PER_SNAPSHOT, :]

        if len(DIMS) == 2:
            x = H[:, DIMS[0]]
            y = H[:, DIMS[1]]
            ax.scatter(x, y,
                       s=5,
                       alpha=0.7,
                       color=cmap(idx),
                       label=f"iter {it}")
        else:  # 3D
            x = H[:, DIMS[0]]
            y = H[:, DIMS[1]]
            z = H[:, DIMS[2]]
            ax.scatter(x, y, z,
                       s=5,
                       alpha=0.7,
                       color=cmap(idx),
                       label=f"iter {it}")

    # Axes labels
    ax.set_xlabel(f"h{DIMS[0] + 1}")
    ax.set_ylabel(f"h{DIMS[1] + 1}")
    if len(DIMS) == 3:
        ax.set_zlabel(f"h{DIMS[2] + 1}")

    ax.legend()
    plt.title("Latent space snapshots at different iterations")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()