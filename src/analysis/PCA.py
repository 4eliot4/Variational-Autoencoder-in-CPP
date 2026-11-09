
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

# 1. Load the latent means and MNIST labels
Z = np.loadtxt("latent_means.csv", delimiter=",")
labels = np.loadtxt(
    "latent_labels.csv", delimiter=",", dtype=int
)

assert Z.ndim == 2
assert Z.shape[0] == len(labels)
assert np.isfinite(Z).all()

print("Latent vectors:", Z.shape)

# 2. PCA: 32D -> 3D
pca = PCA(n_components=3)
Z3 = pca.fit_transform(Z)

print("Explained variance:", pca.explained_variance_ratio_)
print(
    "Total explained variance:",
    pca.explained_variance_ratio_.sum()
)

# 3. Prepare the visualization
df = pd.DataFrame({
    "PC1": Z3[:, 0],
    "PC2": Z3[:, 1],
    "PC3": Z3[:, 2],
    "Digit": labels.astype(str),
    "Index": np.arange(len(labels)),
})

# 4. Interactive 3D scatterplot
fig = px.scatter_3d(
    df,
    x="PC1",
    y="PC2",
    z="PC3",
    color="Digit",
    hover_data=["Index"],
    title="CNN-VAE — MNIST Latent Space",
    opacity=0.65,
    category_orders={
        "Digit": [str(i) for i in range(10)]
    }
)

fig.update_traces(marker=dict(size=2.5))

fig.update_layout(
    template="plotly_dark",
    width=1100,
    height=800,
    scene=dict(
        xaxis_title="PC1",
        yaxis_title="PC2",
        zaxis_title="PC3",
    ),
    legend_title_text="Digit",
)

# 5. Save coordinates for future React Three Fiber use
df.to_csv("latent_3d.csv", index=False)

# 6. Save interactive HTML and open the plot
fig.write_html("latent_3d.html")
fig.show()
