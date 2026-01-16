import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------
# USER PARAMETERS
# -------------------------
BASE_DIR = Path("/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/assets")

MNIST_DIR = Path("/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/MNIST")
TRAIN_IMAGES = MNIST_DIR / "train-images.idx3-ubyte"
TRAIN_LABELS = MNIST_DIR / "train-labels.idx1-ubyte"

START_ITER = 5000
STEP = 5000
NUM_SNAPSHOTS = 8

# How many samples per digit to show (keep small for readability)
N_PER_DIGIT = 50

# If True: PCA to 3D. If False: use first 3 latent dims (0,1,2).
USE_PCA_3D = True

# -------------------------
# IDX HELPERS
# -------------------------
def read_u32_be(f):
    return int.from_bytes(f.read(4), byteorder="big", signed=False)

def load_idx3_images(path):
    with open(path, "rb") as f:
        magic = read_u32_be(f)
        n = read_u32_be(f)
        rows = read_u32_be(f)
        cols = read_u32_be(f)
        assert magic == 2051
        data = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8)
    X = data.reshape(n, rows * cols).astype(np.float32) / 255.0
    return X

def load_idx1_labels(path):
    with open(path, "rb") as f:
        magic = read_u32_be(f)
        n = read_u32_be(f)
        assert magic == 2049
        y = np.frombuffer(f.read(n), dtype=np.uint8)
    return y

# -------------------------
# LATENT LOADER
# -------------------------
def load_latent_csv(iteration):
    fname = BASE_DIR / f"H_latent_iter_{iteration:05d}.csv"
    if not fname.exists():
        print(f"[WARN] missing {fname}")
        return None
    H = np.loadtxt(fname, delimiter=",")
    if H.ndim == 1:
        H = H[None, :]
    return H

# -------------------------
# PCA to 3D (no sklearn)
# -------------------------
def pca_3d(H):
    # Center
    Hc = H - H.mean(axis=0, keepdims=True)
    # Covariance
    C = (Hc.T @ Hc) / (Hc.shape[0] - 1)
    # Eigen decomposition (symmetric -> eigh)
    vals, vecs = np.linalg.eigh(C)
    # Take top 3 eigenvectors
    order = np.argsort(vals)[::-1]
    W = vecs[:, order[:3]]  # (latent_dim, 3)
    Z = Hc @ W              # (n, 3)
    return Z

# -------------------------
# MAIN
# -------------------------
def main():
    # Load MNIST labels (for coloring)
    X = load_idx3_images(TRAIN_IMAGES)
    y = load_idx1_labels(TRAIN_LABELS)

    # Pick fixed indices: N_PER_DIGIT per digit, deterministic
    chosen_idx = []
    for d in range(10):
        idx_d = np.where(y == d)[0][:N_PER_DIGIT]
        chosen_idx.append(idx_d)
    chosen_idx = np.concatenate(chosen_idx)
    chosen_labels = y[chosen_idx]

    iters = [START_ITER + k * STEP for k in range(NUM_SNAPSHOTS)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # One color per digit
    cmap = plt.cm.get_cmap("tab10", 10)

    for it in iters:
        H = load_latent_csv(it)
        if H is None:
            continue

        # IMPORTANT:
        # This assumes H rows correspond to the SAME chosen_idx order.
        # So H must have at least len(chosen_idx) rows and match that ordering.
        if H.shape[0] < len(chosen_idx):
            print(f"[WARN] iteration {it}: H has {H.shape[0]} rows but need {len(chosen_idx)}")
            continue

        Hsel = H[:len(chosen_idx), :]  # assumes same ordering as your fixed eval batch

        if USE_PCA_3D:
            P = pca_3d(Hsel)
        else:
            P = Hsel[:, :3]

        # Plot each digit separately (so legend is clean)
        for d in range(10):
            mask = (chosen_labels == d)
            ax.scatter(P[mask, 0], P[mask, 1], P[mask, 2],
                       s=8, alpha=0.35, color=cmap(d),
                       label=f"{d}" if it == iters[0] else None)

        ax.set_title(f"Latent space (iter {it})")
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.set_zlabel("z3")
        plt.pause(0.6)  # optional: animate through iterations
        ax.cla()        # clear for next iter (comment out if you want overlays)

    # If you want one static plot (not animation), remove cla/pause and just overlay.
    plt.show()

if __name__ == "__main__":
    main()