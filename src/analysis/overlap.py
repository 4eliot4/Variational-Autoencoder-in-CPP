
import numpy as np
from sklearn.neighbors import NearestNeighbors

Z = np.loadtxt("latent_means.csv", delimiter=",")
labels = np.loadtxt(
    "latent_labels.csv", delimiter=",", dtype=int
).reshape(-1)

# Nearest neighbors in the ORIGINAL 32D space
nn = NearestNeighbors(n_neighbors=11)
nn.fit(Z)

distances, indices = nn.kneighbors(Z)

# Exclude each point itself (first neighbor).
neighbor_labels = labels[indices[:, 1:]]

same_digit = neighbor_labels == labels[:, None]

print(
    "Fraction of 10 nearest neighbors with same label:",
    same_digit.mean()
)

for digit in range(10):
    mask = labels == digit
    print(
        f"Digit {digit}: "
        f"{same_digit[mask].mean():.1%} same-label neighbors"
    )
