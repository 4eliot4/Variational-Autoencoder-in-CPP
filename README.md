# LatentSketch — VAE from Scratch in C++

A beginner-friendly yet rigorous implementation of a **Variational Autoencoder (VAE)** built **from scratch in C++ (CPU)**. The goal is to learn the math and the engineering by implementing the critical pieces yourself (tensors, tiny autograd, losses, optimizer), reach **working reconstructions + latent-space interpolation** on tiny synthetic images, and keep the codebase extendable to conv layers or audio spectrograms later.

> **Why this repo exists**
>
> * Many tutorials jump straight to big frameworks. Here you’ll actually implement the building blocks.
> * Scope is intentionally small (MLP + 16×16 grayscale shapes) so you get results fast and learn deeply.
> * Clean milestones + tests keep you from dead-ends and rewrites.

---

## v1 Targets (what “working” means)

* Trainable **MLP-VAE** on 16×16 synthetic shapes (circles/triangles/squares).
* **Reconstructions** look like inputs.
* **Non-zero KL** after warm-up (latent is used).
* **Latent interpolation** produces smooth morphs; export GIF/MP4.
* **2D latent map** (decode a grid over z ∈ [−3,3]²) to visualize the learned manifold.

---

## Roadmap (Milestones with Definitions of Done)

### Milestone A — Core Numerics & Tiny Autograd

**Goal:** forward/backward for basic ops; no networks yet.

* [ ] `Tensor` (data, grad, requires_grad, grad_fn)
* [ ] Ops (each returns new tensor + tape node): `matmul`, `add`, `mul`, `tanh`, `sigmoid`
* [ ] Losses: **MSE**, **BCE-with-logits** (stable)
* [ ] Optimizer: **Adam** (β1=0.9, β2=0.999, ε=1e-8)
* [ ] **Finite-difference grad checks** for each op & loss (rel. error < 1e−3)

**Done when:** All grad checks pass; a toy step runs without NaNs.

---

### Milestone B — Plain Autoencoder (AE)

**Goal:** prove the training loop, data pipeline, logging.

* [ ] `SyntheticShapes` generator (16×16 grayscale; values in [0,1])
* [ ] Modules: `Linear`, activations, `Sequential`, `Module::parameters()`
* [ ] Model: `256 → 128 → 2 → 128 → 256` (tanh or ReLU)
* [ ] Loss: **BCE-with-logits** (apply `sigmoid` only when saving PNGs)
* [ ] Logging: recon grids (PNG), latent CSV (2D bottleneck), scalar losses CSV

**Done when:** Reconstructions are recognizable; 2D latent scatter shows clustering by shape; loss decreases.

---

### Milestone C — Variational Autoencoder (VAE)

**Goal:** add probabilistic latent + stabilize training.

* [ ] Encoder heads output **μ(x)** and **log σ²(x)** (two linear layers)
* [ ] Reparameterization: `z = μ + exp(0.5·logvar) ⊙ ε`, `ε ~ N(0, I)`
* [ ] **Clamp** `logvar` to [−10, 10]; **grad clipping** (global norm ≤ 5)
* [ ] Loss (**ELBO**): `ReconBCE + β·KL_diag_gaussians(μ, logvar)`
* [ ] **β-warmup**: linearly 0 → 1 over first 20–50 epochs

**Done when:** Recon ≈ AE quality, **KL ≠ 0** after warmup, samples from `N(0, I)` decode to plausible shapes.

---

### Milestone D — Interpolation & Latent Map

**Goal:** demonstrate smooth, meaningful latent space.

* [ ] Pick two inputs → encode (use means μ₁, μ₂)
* [ ] Interpolate: `z(α) = (1−α)·μ₁ + α·μ₂`, decode frames, export GIF/MP4
* [ ] Decode a grid over z ∈ [−3,3]² → mosaic image of the manifold
* [ ] Mini report (1–2 pages): ELBO math, Recon/KL curves, visuals

**Done when:** Morphs are smooth; the manifold image shows structured transitions.

---

## Project Structure

```
/third_party/
  eigen/                # Eigen headers (linear algebra)
  stb/stb_image_write.h # PNG writer
/src/
  core/
    Tensor.h            # tensor + grad storage
    Autograd.h          # tape nodes, backward engine
    Random.h            # RNG for ε ~ N(0, I)
  nn/
    Module.h            # base class + parameters()
    Linear.h            # Linear layer
    Activations.h       # Tanh, ReLU, Sigmoid
    Losses.h            # MSE, BCE-with-logits, KL
    Optim.h             # Adam
  data/
    SyntheticShapes.h   # 16×16 circles/triangles/squares
  vae/
    VAE.h               # encoder (μ, logvar) + decoder
  app/
    train_vae.cpp       # entry point; trains; exports PNG/CSV/GIF
/assets/                # exported images / videos
/logs/                  # CSV for losses, metrics
```

---

## Build & Run

**Requirements**: C++17 compiler, CMake ≥ 3.16

```bash
# Configure & build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Train VAE (example CLI — adjust to your app)
./build/train_vae \
  --epochs 150 \
  --batch 128 \
  --lr 1e-3 \
  --latent 2 \
  --beta_warmup_epochs 30
```

> Outputs: PNG recon grids in `/assets`, latent CSV + losses CSV in `/logs`, optional GIF in `/assets`.

---

## Data: Synthetic Shapes (why and how)

* **Why**: no I/O headaches; infinite clean data; instant iterations.
* **What**: 16×16 grayscale images in [0,1]. Random size/position for circles, triangles, squares.
* **Extend**: later swap with MNIST (28×28) or spectrograms for audio.

---

## Math Primer (bare minimum you’ll implement)

### Autoencoder vs VAE

* **AE:** deterministic bottleneck `z = f(x)` → reconstruct `x̂ = g(z)`.
* **VAE:** stochastic bottleneck; encoder outputs **μ(x), log σ²(x)**; sample
  `z = μ + σ ⊙ ε`, `ε ~ N(0, I)`, `σ = exp(0.5·logvar)`.

### Loss (ELBO)

`ELBO = ReconLoss + β · KL(q(z|x) || p(z))`, with `p(z)=N(0, I)`.

* **Recon (BCE-with-logits)** for pixels in [0,1].
* **KL (diag Gaussians)** (closed form):
  [ KL = -\tfrac12 \sum (1 + \log σ^2 − μ^2 − σ^2) ]
* **β-warmup**: start β=0, ramp to 1 → avoids “posterior collapse” (model ignoring z).

---

## Autograd (what it is and rules to follow)

* During forward pass, **record** each op as a small node with a `backward()` method.
* Calling `backward(loss)` traverses nodes **in reverse order** to fill `grad`.
* **Never** overwrite tensors in place if they need gradients; every op returns a **new** tensor.
* After each step: **zero grads** and **clear the tape**.
* Write **finite-difference gradient checks** for ops & losses (rel. error < 1e−3).

---

## Training Configuration (good defaults)

* **Shapes**: 16×16 (D=256), batch=128, ≥5k samples per epoch
* **Model**: hidden 256→128, latent `k=2` (later try 8/16)
* **Optim**: Adam lr=1e−3, weight decay 1e−5
* **Stability**: gradient clip (norm ≤ 5), clamp `logvar ∈ [−10,10]`
* **Schedule**: β warmup over first 20–50 epochs; total 100–200 epochs

---

## Outputs & Visualization

* **Recon grids**: side-by-side original vs reconstructed.
* **Latent scatter (k=2)**: CSV → plot to see clusters by shape.
* **Interpolation**: encode two inputs → linear mix in z → decode frames → GIF/MP4.
* **Latent manifold**: decode a regular grid over z ∈ [−3,3]² → mosaic image.

---

## Troubleshooting (symptoms → fixes)

| Symptom                         | Likely cause                                | Fix                                              |
| ------------------------------- | ------------------------------------------- | ------------------------------------------------ |
| Grads = 0 or NaN on first batch | in-place ops; wrong backward                | return new tensors; add grad checks              |
| Loss = inf/NaN                  | sigmoid+ BCE overflow; exp(logvar) overflow | use **BCE-with-logits**; **clamp logvar**        |
| Training jumps wildly           | huge grads                                  | **gradient clipping**; lower LR                  |
| KL ≈ 0 (latent unused)          | posterior collapse                          | **β-warmup**; increase decoder capacity; tune LR |
| Recon never improves            | init/LR wrong; bug in ops                   | Xavier init; Adam 1e-3; re-check backward        |

---

## Next Steps (after v1)

* **Conv VAE**: add Conv/ConvTranspose layers (better images).
* **Audio VAE**: replace images with **spectrograms** (STFT/ISTFT) → sound morphing.
* **β-VAE study**: explore rate–distortion trade-off and disentanglement.
* **UI**: tiny **Dear ImGui** app with live z-sliders and sampling.

---

## Glossary (1-liners)

* **Neuron/Layer:** computes `y = σ(Wx + b)`.
* **Logit:** raw score before sigmoid.
* **BCE-with-logits:** stable BCE computed directly from logits.
* **KL divergence:** penalty that makes the latent look like `N(0, I)`.
* **Reparameterization:** `z = μ + σ·ε` with `ε ∼ N(0, I)` so gradients flow through μ, σ.
* **β-warmup:** gradually increase the KL weight from 0 to 1.
* **Gradient clipping:** cap gradient size to avoid exploding updates.

---

## License

Choose a permissive license (MIT/BSD-3-Clause) so others can learn from and use your code.

---

## Acknowledgements

This project is designed for learning by building. Start tiny, verify often, and extend only when v1 is solid.
