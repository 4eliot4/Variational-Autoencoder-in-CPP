# VAE from Scratch in C++

A small project to understand generative models by implementing them myself in Cpp, using Eigen for matrix operations. I started with a basic autoencoder on MNIST, then turned it into a Variational Autoencoder (VAE). I'm now working on a convolutional version, with the goal of trying diffusion models afterwards.

I'm intentionally not using PyTorch or TensorFlow for the network, gradients, or optimizer. The point is to understand what happens during training rather than just calling a library.

## Progress

- [x] Fully connected autoencoder
- [x] VAE with the reparameterization trick
- [x] Manual backpropagation and Adam optimizer
- [x] dataset reconstruction and latent sampling
- [ ] CNN-VAE (currently implementing convolutions and their gradients)
- [ ] Diffusion model (DDPM)
- [ ] Possibly latent diffusion later

## Current VAE

The first version uses a fully connected encoder and decoder on binarized **28 × 28 MNIST** images.

```text
Input image (28 × 28)
        │
     Flatten
        │ 784
     Linear
        │ 64
       ReLU
        │
    ┌───┴───┐
    │       │
  Mean μ  Log-var V
    │       │
    └───┬───┘
        │
 z = μ + exp(V/2) ⊙ ε       ε ~ N(0, I)
        │ 64
     Linear
        │ 64
       ReLU
        │
     Linear
        │ 784
     Sigmoid
        │
 Reconstructed image
```

The hidden and latent dimensions are both 64 in this version. One thing I want to change in the next version is to keep these dimensions separate so I can experiment with different latent sizes.

### Encoder and sampling

For a batch of flattened images, the encoder computes

```math
H = \operatorname{ReLU}(XW_1+b_1)
```

and predicts the mean and log-variance of a diagonal Gaussian:

```math
M = HW_\mu+b_\mu, \qquad V = HW_v+b_v.
```

Instead of sampling directly from this distribution, I use the **reparameterization trick**:

```math
E \sim \mathcal N(0,I), \qquad Z=M+\exp(V/2)\odot E.
```

This lets gradients propagate through the sampling step.

The decoder then takes `Z`, applies a linear layer, ReLU, another linear layer, and finally sigmoid to obtain pixel probabilities.

### Loss

The training objective is the negative ELBO, averaged over the batch:

```math
\mathcal L=\mathcal L_{\mathrm{rec}}+\mathcal L_{\mathrm{KL}}.
```

For the binarized images, reconstruction uses binary cross-entropy (summed over pixels, averaged over images):

```math
\mathcal L_{\mathrm{rec}}=-\frac1B\sum_{b=1}^{B}\sum_{j=1}^{784}
\left[x_{bj}\log\hat x_{bj}+(1-x_{bj})\log(1-\hat x_{bj})\right].
```

The KL term keeps the encoded distribution close to the standard normal prior:

```math
\mathcal L_{\mathrm{KL}}=\frac1{2B}\sum_{b=1}^{B}\sum_{k=1}^{d}
\left[M_{bk}^{2}+\exp(V_{bk})-1-V_{bk}\right].
```

### Manual gradients

I implemented backpropagation directly with Eigen matrices. The part I found most interesting is how the reconstruction and KL gradients combine at the latent distribution.

With sigmoid + binary cross-entropy, the gradient at the decoder output logits is

```math
G_Y=\frac{\hat X-X}{B}.
```

Backpropagating through the decoder gives $G_Z$, the reconstruction gradient with respect to the sampled latent vectors. Then:

```math
\boxed{G_M=G_Z+\frac{M}{B}}
```

```math
\boxed{G_V=\frac12 G_Z\odot\exp(V/2)\odot E+\frac{\exp(V)-1}{2B}}
```

The parameter gradients for the two encoder heads are

```math
G_{W_\mu}=H^TG_M, \qquad G_{W_v}=H^TG_V,
```

and the gradients join again at the shared hidden layer:

```math
G_H=G_MW_\mu^T+G_VW_v^T.
```

The remaining derivatives follow the usual matrix multiplication and ReLU rules. I also implemented Adam myself, including its first/second moments and bias correction.

### More maths: the complete forward and backward pass

Writing the network as matrix operations makes it easier to check dimensions and implement the gradients without automatic differentiation. Biases are broadcast across the batch.

```math
\begin{aligned}
A_1 &= XW_1+b_1, & H &= \operatorname{ReLU}(A_1),\\
M &= HW_\mu+b_\mu, & V &= HW_v+b_v,\\
Z &= M+e^{V/2}\odot E, & E_{bk} &\sim \mathcal N(0,1),\\
A_2 &= ZW_2+b_2, & H_2 &= \operatorname{ReLU}(A_2),\\
Y &= H_2W_3+b_3, & \hat X &= \sigma(Y).
\end{aligned}
```

For a batch of $B$ images, with input dimension $D=784$, hidden size $h=64$, and latent size $d=64$:

```math
\begin{aligned}
X,\hat X,Y &\in\mathbb R^{B\times D},\\
H,A_1,H_2,A_2 &\in\mathbb R^{B\times h},\\
M,V,Z,E &\in\mathbb R^{B\times d},\\
W_1 &\in\mathbb R^{D\times h},\\
W_\mu,W_v &\in\mathbb R^{h\times d},\\
W_2 &\in\mathbb R^{d\times h},\\
W_3 &\in\mathbb R^{h\times D}.
\end{aligned}
```

**Decoder backpropagation:** starting from the BCE/sigmoid simplification, the derivatives flow backwards through the two linear layers and ReLU.

```math
\begin{aligned}
G_Y &= \frac{\hat X-X}{B},\\[3pt]
G_{W_3} &= H_2^T G_Y,
& G_{b_3} &= \sum_{b=1}^B G_Y[b,:],\\[3pt]
G_{H_2} &= G_YW_3^T,\\[3pt]
G_{A_2} &= G_{H_2}\odot\mathbf 1_{A_2>0},\\[3pt]
G_{W_2} &= Z^TG_{A_2},
& G_{b_2} &= \sum_{b=1}^B G_{A_2}[b,:],\\[3pt]
G_Z &= G_{A_2}W_2^T.
\end{aligned}
```

**Through the Gaussian sample:** because $Z=M+e^{V/2}\odot E$, with $E$ fixed during backpropagation, its local derivatives are

```math
\frac{\partial Z_{bk}}{\partial M_{bk}}=1,
\qquad
\frac{\partial Z_{bk}}{\partial V_{bk}}
=\frac12e^{V_{bk}/2}E_{bk}.
```

The reconstruction gradient and KL gradient are added at the mean and log-variance:

```math
\begin{aligned}
G_M &= G_Z+\frac{M}{B},\\[5pt]
G_V &= \frac12G_Z\odot e^{V/2}\odot E
       +\frac{e^V-1}{2B}.
\end{aligned}
```

**Encoder backpropagation:** both latent heads contribute to the gradient entering the shared encoder.

```math
\begin{aligned}
G_{W_\mu} &= H^TG_M,
& G_{b_\mu} &= \sum_{b=1}^B G_M[b,:],\\[3pt]
G_{W_v} &= H^TG_V,
& G_{b_v} &= \sum_{b=1}^B G_V[b,:],\\[3pt]
G_H &= G_MW_\mu^T+G_VW_v^T,\\[3pt]
G_{A_1} &= G_H\odot\mathbf 1_{A_1>0},\\[3pt]
G_{W_1} &= X^TG_{A_1},
& G_{b_1} &= \sum_{b=1}^B G_{A_1}[b,:].
\end{aligned}
```

### Where the KL formula comes from

The encoder outputs a diagonal Gaussian, and the prior is a standard normal distribution:

```math
q_\phi(z\mid x)=\mathcal N\!\left(\mu,\operatorname{diag}(\sigma^2)\right),
\qquad p(z)=\mathcal N(0,I),
\qquad V=\log\sigma^2.
```

The evidence lower bound for one image is

```math
\log p_\theta(x)
\geq
\underbrace{\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]}_{\text{reconstruction term}}
-
\underbrace{D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p(z)\right)}_{\text{regularization}}.
```

For this diagonal Gaussian, the KL divergence can be calculated analytically:

```math
\begin{aligned}
D_{\mathrm{KL}}(q_\phi\|p)
&=\frac12\sum_{k=1}^{d}
\left(\mu_k^2+\sigma_k^2-1-\log\sigma_k^2\right)\\
&=\frac12\sum_{k=1}^{d}
\left(\mu_k^2+e^{V_k}-1-V_k\right).
\end{aligned}
```

In particular, its derivatives are simple, which is why the KL gradients can be added directly to the reconstruction gradients:

```math
\boxed{
\frac{\partial\mathcal L_{\mathrm{KL}}}{\partial M}=\frac MB,
\qquad
\frac{\partial\mathcal L_{\mathrm{KL}}}{\partial V}=\frac{e^V-1}{2B}
}
```

### Adam update

For each parameter tensor $\theta$ and its gradient $g_t$, I keep running estimates of the first and second moments:

```math
\begin{aligned}
m_t &= \beta_1m_{t-1}+(1-\beta_1)g_t,\\
v_t &= \beta_2v_{t-1}+(1-\beta_2)g_t^{\odot2},\\
\hat m_t &= \frac{m_t}{1-\beta_1^t},\\
\hat v_t &= \frac{v_t}{1-\beta_2^t},\\
\theta_t &= \theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
\end{aligned}
```

All squares, square roots, and divisions in the Adam update are elementwise. The batch loss is averaged **once**, starting from $G_Y=(\hat X-X)/B$; the reconstruction gradients should not be divided by $B$ again as they propagate through the network.

## MNIST results

Training setup for the initial VAE:

| Setting | Value |
| --- | --- |
| Batch size | 64 |
| Hidden / latent size | 64 / 64 |
| Optimizer | Adam |
| Learning rate | 0.0003 |
| Training steps | 30,000 |
| Dataset | Binarized MNIST |

At step 30,000, one training minibatch gave:

| Loss | Value |
| --- | ---: |
| Reconstruction | 61.59 |
| KL | 27.64 |
| Total | 89.23 |

The model reconstructs most digits fairly well. The outputs are still noticeably blurrier than the original images, especially around the edges. These numbers are **training minibatch losses**, not test-set benchmarks.


## What's next?

### CNN-VAE in progress

The fully connected model flattens the image, so it doesn't explicitly use the fact that nearby pixels are related. The next step is to replace the image encoder/decoder with convolutional layers.

I'm currently implementing:

- A tensor representation for batches, channels, height, and width
- Convolution with configurable filters, stride, and padding
- Gradients for convolution kernels, biases, and inputs
- Transposed convolution for the decoder

The VAE sampling and loss stay mostly the same. I'll first test the CNN-VAE on MNIST and Fashion-MNIST, then potentially move to more complex RGB images.

### Diffusion planned

After the CNN-VAE, I'd like to implement a small **Denoising Diffusion Probabilistic Model (DDPM)**. The idea is to train a network to predict noise added to an image at a given timestep, then generate images by gradually denoising random noise.

Longer term, it would be interesting to combine both approaches and try diffusion in a learned latent space.

## Tools

- C++
- Eigen (matrix operations)
- MNIST
- Standard C++ random number generation

This is mainly a learning project, and I'm keeping the implementations relatively simple for now so the math stays visible in the code.
