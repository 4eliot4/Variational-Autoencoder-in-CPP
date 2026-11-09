
import numpy as np

Z = np.loadtxt("latent_means.csv", delimiter=",")
labels = np.loadtxt(
    "latent_labels.csv", delimiter=",", dtype=int
).reshape(-1)

assert Z.shape[0] == labels.size
assert np.isfinite(Z).all()

# One centroid per digit, computed in original 32D
centroids = np.array([
    Z[labels == digit].mean(axis=0)
    for digit in range(10)
])

print("Centroid dimensions:", centroids.shape)

# Whitespace-separated: easy to read in C++
np.savetxt("digit_centroids.txt", centroids, fmt="%.9g")
