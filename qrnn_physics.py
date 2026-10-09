"""
qrnn_physics.py -- one optomechanical node driven by two single photons in consecutive time frames.

Units: the Hermite-Gaussian (HG) width tau = 1.  Frame f (f=1,2) occupies [(f-1)T, fT], photon mode centred at
(f-1/2)T.  The control pulse spans both frames:
    alpha(t) = theta_0 Phi_0(t) + theta_1 Phi_1(t),   Phi_n(t) = psi_n((t-T)/s_c)/sqrt(s_c),  s_c = T/2
so Phi_0 is a broad Gaussian centred on the frame boundary and the two lobes of Phi_1 sit on the two frame centres.

Exact phonon statistics (vacuum initial state, zero-temperature baths, S = 1):
    beta_f = <K_b, xi_f>   (amplitude for photon f to end up as a phonon at t = 2T)
    P_b(2) = 2|beta_1 beta_2|^2,  P_b(1) = |beta_1|^2 + |beta_2|^2 - 4|beta_1 beta_2|^2,  P_b(0) = 1 - P_b(1) - P_b(2)
beta_f = sum_n c_{f,n} K_{f,n} where K_{f,n} = z_2(2T) of the single-photon hierarchy (Eqs. 44-45) driven by psi_n in
frame f.  Verified against a cascaded density-matrix simulation (verify_against_density_matrix).
"""
import numpy as np
from math import factorial, pi, sqrt
from numpy.polynomial.hermite import hermval


def psi(n, t):
    c = np.zeros(n + 1); c[n] = 1
    return (2 ** n * factorial(n) * sqrt(pi)) ** -0.5 * hermval(t, c) * np.exp(-t ** 2 / 2)


class Node:
    def __init__(self, T=8.0, kappa=4.0, gamma=0.0, Delta=0.0, n_modes=2, dt=0.04):
        self.T, self.kappa, self.gamma, self.Delta, self.M = T, kappa, gamma, Delta, n_modes
        self.sc = T / 2
        self.nt = int(round(2 * T / dt)); self.dt = 2 * T / self.nt
        t = np.linspace(0, 2 * T, 2 * self.nt + 1)          # half-step grid for RK4
        self.t_half = t
        # photon basis: columns (f, n) -> index f*M + n
        self.basis = np.stack([psi(n, t - (f + 0.5) * T) for f in range(2) for n in range(n_modes)], axis=-1)
        self.Phi = np.stack([psi(n, (t - T) / self.sc) / np.sqrt(self.sc) for n in range(2)], axis=-1)

    def alpha_t(self, theta):
        """theta: (..., 2) real/complex -> alpha on half-step grid (..., len(t))"""
        return np.tensordot(theta, self.Phi.T, axes=([-1], [0]))

    def kernels(self, theta):
        """K[..., f*M+n] = z_2(2T) for photon basis mode (f,n).  theta shape (R,2).  Vectorised RK4."""
        theta = np.atleast_2d(theta)
        R = theta.shape[0]; ncol = self.basis.shape[1]
        al = self.alpha_t(theta)                              # (R, 2nt+1)
        sk, k2, g2, D = np.sqrt(self.kappa), self.kappa / 2, self.gamma / 2, self.Delta
        z1 = np.zeros((R, ncol), complex); z2 = np.zeros((R, ncol), complex)
        def f(i, z1, z2):
            a = al[:, i][:, None]
            return (-(1j * D + k2) * z1 - 1j * np.conj(a) * z2 - sk * self.basis[i][None, :],
                    -1j * a * z1 - g2 * z2)
        dt = self.dt
        for s in range(self.nt):
            i0, i1, i2 = 2 * s, 2 * s + 1, 2 * s + 2
            a1, b1 = f(i0, z1, z2)
            a2, b2 = f(i1, z1 + dt / 2 * a1, z2 + dt / 2 * b1)
            a3, b3 = f(i1, z1 + dt / 2 * a2, z2 + dt / 2 * b2)
            a4, b4 = f(i2, z1 + dt * a3, z2 + dt * b3)
            z1 = z1 + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
            z2 = z2 + dt / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
        return z2                                              # (R, 2M)

    def probs(self, theta, c1, c2, K=None):
        """Exact P_b(0,1,2) for each node. c1, c2: coefficient vectors (M,) (normalised inside)."""
        if K is None:
            K = self.kernels(theta)
        c1 = np.asarray(c1, complex); c2 = np.asarray(c2, complex)
        c1 = c1 / np.linalg.norm(c1); c2 = c2 / np.linalg.norm(c2)
        M = self.M
        b1 = K[:, :M] @ c1; b2 = K[:, M:] @ c2
        x, y = abs(b1) ** 2, abs(b2) ** 2
        P2 = 2 * x * y; P1 = x + y - 4 * x * y
        P = np.stack([1 - P1 - P2, P1, P2], axis=-1)
        return np.clip(P, 0, 1) / np.clip(P, 0, 1).sum(-1, keepdims=True)


def sample(P, S, rng):
    """Shot-noise estimate of P from S repetitions (S = None or 0 -> exact)."""
    if not S:
        return P.copy()
    return np.array([rng.multinomial(S, p) / S for p in P])


# ----------------------------------------------------------------------------------------------
def verify_against_density_matrix(node, theta, c1, c2, verbose=True):
    """Cascade two virtual source cavities (one photon each, modes xi_1, xi_2) into mode a."""
    from scipy.integrate import solve_ivp
    T = node.T
    c1 = np.asarray(c1, float) / np.linalg.norm(c1); c2 = np.asarray(c2, float) / np.linalg.norm(c2)
    xi = [lambda t, c=c1: sum(c[n] * psi(n, t - 0.5 * T) for n in range(node.M)),
          lambda t, c=c2: sum(c[n] * psi(n, t - 1.5 * T) for n in range(node.M))]
    tg = np.linspace(0, 2 * T, 8001)
    cums = [np.concatenate([[0], np.cumsum(0.5 * (x(tg[1:]) ** 2 + x(tg[:-1]) ** 2) * np.diff(tg))]) for x in xi]
    g = [lambda t, x=x, cu=cu: x(t) / np.sqrt(max(1 - np.interp(t, tg, cu), 1e-12)) for x, cu in zip(xi, cums)]
    op = lambda n: np.diag(np.sqrt(np.arange(1, n)), 1)
    dims = [2, 2, 3, 3]
    def emb(o, k):
        mats = [np.eye(d) for d in dims]; mats[k] = o
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out
    c1o, c2o, a, b = (emb(op(d), k) for k, d in enumerate(dims))
    dg = lambda X: X.conj().T
    Dm = int(np.prod(dims))
    def Ds(L, r): return L @ r @ dg(L) - 0.5 * (dg(L) @ L @ r + r @ dg(L) @ L)
    th = np.asarray(theta)
    alf = lambda t: th[0] * psi(0, (t - T) / node.sc) / np.sqrt(node.sc) + th[1] * psi(1, (t - T) / node.sc) / np.sqrt(node.sc)
    def rhs(t, rv):
        r = rv.reshape(Dm, Dm)
        Ls = [g[0](t) * c1o, g[1](t) * c2o, np.sqrt(node.kappa) * a]
        A = alf(t)
        H = node.Delta * dg(a) @ a + A * a @ dg(b) + np.conj(A) * dg(a) @ b
        for i in range(3):
            for j in range(i + 1, 3):
                H = H + 0.5j * (dg(Ls[i]) @ Ls[j] - dg(Ls[j]) @ Ls[i])
        dr = -1j * (H @ r - r @ H) + Ds(sum(Ls), r) + node.gamma * Ds(b, r)
        return dr.ravel()
    psi0 = np.zeros(Dm); psi0[np.ravel_multi_index((1, 1, 0, 0), dims)] = 1
    sol = solve_ivp(rhs, (0, 2 * T), np.outer(psi0, psi0).astype(complex).ravel(), rtol=1e-9, atol=1e-11)
    r = sol.y[:, -1].reshape(Dm, Dm)
    nb = np.real(np.diag(r)).reshape(dims).sum(axis=(0, 1, 2))
    P_dm = nb
    P_fac = node.probs(np.atleast_2d(theta), c1, c2)[0]
    if verbose:
        print("density matrix P_b(0,1,2):", np.round(P_dm, 6))
        print("closed form   P_b(0,1,2):", np.round(P_fac, 6), " max diff", np.max(abs(P_dm - P_fac)))
    return P_dm, P_fac


if __name__ == "__main__":
    node = Node(gamma=0.02, Delta=0.1)
    verify_against_density_matrix(node, [2.5, -1.2], [0.8, 0.6], [0.3, -0.95])
    verify_against_density_matrix(node, [1.0, 2.0], [1, 0], [0.5, 0.5])
    # RK4 accuracy vs finer step
    fine = Node(gamma=0.02, Delta=0.1, dt=0.005)
    th = np.array([[2.5, -1.2], [1.0, 2.0]])
    print("RK4 dt=0.04 vs 0.005 kernel diff:", np.max(abs(node.kernels(th) - fine.kernels(th))))
