
import numpy as np
import pandas as pd
import plotly.express as px
import umap

Z = np.loadtxt("latent_means.csv", delimiter=",")
labels = np.loadtxt(
    "latent_labels.csv", delimiter=",", dtype=int
).reshape(-1)

assert Z.shape[0] == labels.size
assert np.isfinite(Z).all()

# Project 32D latent means into 3D
reducer = umap.UMAP(
    n_components=3,
    n_neighbors=30,
    min_dist=0.0,
    metric="euclidean",
    random_state=42
)

Z3 = reducer.fit_transform(Z)

df = pd.DataFrame({
    "x": Z3[:, 0],
    "y": Z3[:, 1],
    "z": Z3[:, 2],
    "Digit": labels.astype(str),
    "Index": np.arange(len(labels))
})

fig = px.scatter_3d(
    df,
    x="x", y="y", z="z",
    color="Digit",
    hover_data=["Index"],
    opacity=0.65,
    category_orders={
        "Digit": [str(i) for i in range(10)]
    },
    title="CNN-VAE — 3D UMAP Latent Space"
)

fig.update_traces(marker=dict(size=2.5))
fig.update_layout(
    template="plotly_dark",
    height=800
)

fig.show()

df.to_csv("latent_umap_3d.csv", index=False)
