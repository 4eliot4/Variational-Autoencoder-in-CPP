import numpy as np
from time import perf_counter

# ---------- Settings ----------
epsilon_base = 1e-6


def rel_eps(aij):
    """Relative epsilon for stable finite differences."""
    return epsilon_base * max(1.0, abs(aij))


# ---------- Numeric gradient (central difference) ----------
def num_grad_entry(i, j, A, B):
    eps = rel_eps(A[i, j])
    Aplus = A.copy()
    Aminus = A.copy()
    Aplus[i, j] += eps
    Aminus[i, j] -= eps

    Mplus = Aplus @ B
    Mminus = Aminus @ B
    Lplus = np.sum(Mplus * Mplus)
    Lminus = np.sum(Mminus * Mminus)
    return (Lplus - Lminus) / (2.0 * eps)


def num_grad_mat(A, B):
    """Full numeric gradient (very slow for large A)."""
    G = np.empty_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            G[i, j] = num_grad_entry(i, j, A, B)
    return G


# ---------- Analytic gradient ----------
def anal_grad_mat(A, B):
    """Analytic gradient: d/dA ||A B||_F^2 = 2 * A * (B B^T)"""
    BBt = B @ B.T
    return 2.0 * (A @ BBt)


# ---------- Main ----------
def main():
    # Same shapes as your C++ code
    m, k, n = 3, 2, 3
    np.random.seed(42)
    A = np.random.uniform(-1, 1, (m, k))
    B = np.random.uniform(-1, 1, (k, n))

    # --- Loss computation ---
    t0 = perf_counter()
    M = A @ B
    Loss = np.sum(M * M)
    t1 = perf_counter()
    print(f"Loss = {Loss:.6f}  ({(t1 - t0)*1e3:.3f} ms)")

    # --- Single-entry numeric gradient ---
    t2 = perf_counter()
    g_entry = num_grad_entry(1, 1, A, B)
    t3 = perf_counter()
    print(f"Grad loss (1,1): {g_entry:.6f}  ({(t3 - t2)*1e3:.3f} ms)")

    # --- Full numeric gradient matrix ---
    t4 = perf_counter()
    grad_num = num_grad_mat(A, B)
    t5 = perf_counter()
    print(f"Grad loss Mat Numeric:\n{grad_num}\n(Time: {(t5 - t4)*1e3:.3f} ms)")

    # --- Analytic gradient matrix ---
    t6 = perf_counter()
    grad_anal = anal_grad_mat(A, B)
    t7 = perf_counter()
    print(f"Grad loss Mat Analytic:\n{grad_anal}\n(Time: {(t7 - t6)*1e3:.3f} ms)")

    # --- Relative error ---
    rel_err = np.linalg.norm(grad_num - grad_anal) / (np.linalg.norm(grad_anal) + 1e-12)
    print(f"Relative error (numeric vs analytic): {rel_err:.3e}")


if __name__ == "__main__":
    main()