"""
qrnn_learn.py -- training and 5-step prediction for the two-frame optomechanical quantum RNN.

Protocol (one "step" s = one double time frame):
  1. encode the data c_s = (c^{(2s-1)}, c^{(2s)}) (two HG coefficient vectors, M = 2 modes psi_0, psi_1)
     into two single-photon pulses in frames 2s-1, 2s;
  2. each of R parallel nodes (identical hardware, different feedback weights) is driven by those photons and its
     control pulse alpha_s^r(t) = theta_{s,0}^r Phi_0(t) + theta_{s,1}^r Phi_1(t) spanning both frames;
  3. the phonon number is measured at the end of frame 2; repeating S times gives p_hat_s^r = (P0,P1,P2);
  4. control update (leaky-tanh RNN cell):
        theta_{s+1} = (1-lam) theta_s + lam * A * tanh( W_th theta_s / A + W_p p_hat_s + b )
  5. linear readout:  c_hat_{s+1} = V phi_s,   phi_s = [p_hat_s, p_hat_{s-1}, theta_{s+1}/A, 1] over all nodes.
Training: V by ridge regression (closed form); (W_th, W_p, b) by SPSA on a validation loss (hardware friendly:
two noisy loss evaluations per iteration, no gradients through the quantum dynamics).
Prediction: steps 1..50 teacher-forced with the data; steps 51..55 closed loop -- the predicted coefficients are
re-encoded as the next photons.
Run:  python qrnn_learn.py            (writes PNGs + metrics.json into ./out)
"""
import json, os, time, argparse
import numpy as np
from qrnn_physics import Node, sample, psi

# ============================================================================== configuration
CFG = dict(
    kappa=4.0, gamma=0.0, Delta=0.0, T=8.0, dt=0.08,   # node physics (HG width = 1); RK4 step
    R=6,                                         # parallel nodes
    S=1000,                                      # repetitions per double frame (shot noise)
    A=5.0, lam=0.5,                              # control amplitude scale, leak rate
    n_train=50, n_pred=5, washout=4, n_val=10,
    spsa_iters=150, spsa_a=0.3, spsa_c=0.15,
    n_eval_seeds=30, seed=1,
)


# ============================================================================== data
def make_signal(n_steps, kind="quasi"):
    """Angle of the HG superposition in each frame; c = (cos phi, sin phi).  Frames k = 1..2 n_steps."""
    k = np.arange(1, 2 * n_steps + 1)
    if kind == "quasi":
        phi = np.pi / 4 + 0.55 * np.sin(2 * np.pi * k / 22.0) + 0.25 * np.sin(2 * np.pi * k / 7.3 + 0.4)
    elif kind == "chirp":
        phi = np.pi / 4 + 0.6 * np.sin(2 * np.pi * k / 30.0 * (1 + k / 200))
    else:
        raise ValueError(kind)
    # Mode Bloch vector r = (c0^2 - c1^2, 2 c0 c1) = (cos 2phi, sin 2phi): the photon state |xi> is only defined up to a
    # global sign, and the phonon statistics are quadratic in c, so r (not c) is the natural data/target variable.
    r = np.stack([np.cos(2 * phi), np.sin(2 * phi)], -1)   # (2n, 2)
    return r.reshape(n_steps, 4), phi.reshape(n_steps, 2)   # step s -> [r^(2s-1), r^(2s)]


def r_to_c(r2):
    """Bloch vector (2,) -> HG coefficients (cos phi, sin phi)."""
    ang = 0.5 * np.arctan2(r2[1], r2[0])
    return np.array([np.cos(ang), np.sin(ang)])


def normalise(y):
    return np.concatenate([y[..., :2] / np.linalg.norm(y[..., :2], axis=-1, keepdims=True),
                           y[..., 2:] / np.linalg.norm(y[..., 2:], axis=-1, keepdims=True)], -1)


def fidelity(r_hat, r):
    """Mode overlap |<xi_hat|xi>|^2 = (1 + r_hat.r)/2 per frame, averaged over the two frames of a step."""
    a, b = normalise(r_hat), normalise(r)
    return 0.25 * (2 + (a[..., :2] * b[..., :2]).sum(-1) + (a[..., 2:] * b[..., 2:]).sum(-1))


def angle_error(r_hat, r):
    """RMS error (degrees) of the mode angle phi over the two frames: phi = arg(r)/2."""
    a, b = normalise(r_hat), normalise(r)
    d = [np.arccos(np.clip((a[..., 2*f:2*f+2] * b[..., 2*f:2*f+2]).sum(-1), -1, 1)) / 2 for f in range(2)]
    return np.degrees(np.sqrt((d[0] ** 2 + d[1] ** 2) / 2))


# ============================================================================== model
class QRNN:
    def __init__(self, cfg, node=None):
        self.cfg = cfg
        self.node = node or Node(T=cfg["T"], kappa=cfg["kappa"], gamma=cfg["gamma"], Delta=cfg["Delta"], dt=cfg["dt"])
        self.R = cfg["R"]
        self.shapes = dict(Wth=(self.R, 2, 2), Wp=(self.R, 2, 3), b=(self.R, 2))
        self.sizes = {k: int(np.prod(v)) for k, v in self.shapes.items()}
        self.n_par = sum(self.sizes.values())

    def unpack(self, w):
        out, i = {}, 0
        for k, shp in self.shapes.items():
            out[k] = w[i:i + self.sizes[k]].reshape(shp); i += self.sizes[k]
        return out

    def init_params(self, rng, scale=0.3, b_range=1.5):
        """Small random recurrent/input weights; diverse biases so that the R control pulses differ."""
        w = rng.normal(0, scale, self.n_par)
        w[-self.sizes["b"]:] = rng.uniform(-b_range, b_range, self.sizes["b"])
        return w

    def no_feedback(self, w):
        """Ablation: same biases (fixed, diverse control pulses) but W_th = W_p = 0."""
        v = np.zeros_like(w); v[-self.sizes["b"]:] = w[-self.sizes["b"]:]
        return v

    def run(self, w, data, rng, S=None, n_teacher=None, readout=None, n_steps=None, theta0=None):
        """Drive the nodes.  data: (n,4).  If readout V given, steps >= n_teacher use the model's own prediction."""
        cfg = self.cfg; A, lam = cfg["A"], cfg["lam"]
        S = cfg["S"] if S is None else S
        p = self.unpack(w)
        n_steps = n_steps or len(data)
        n_teacher = n_steps if n_teacher is None else n_teacher
        theta = A * np.tanh(p["b"]) if theta0 is None else np.array(theta0, float)
        p_prev = np.zeros((self.R, 3))
        feats, thetas, phats, inputs, preds = [], [], [], [], []
        x = data[0]
        for s in range(n_steps):
            if s < n_teacher:
                x = data[s]
            inputs.append(x); thetas.append(theta.copy())
            P = self.node.probs(theta, r_to_c(x[:2]), r_to_c(x[2:]))
            ph = sample(P, S, rng); phats.append(ph)
            pre = np.einsum("rij,rj->ri", p["Wth"], theta / A) + np.einsum("rij,rj->ri", p["Wp"], ph) + p["b"]
            theta = (1 - lam) * theta + lam * A * np.tanh(pre)
            phi = np.concatenate([ph[:, 1:].ravel(), p_prev[:, 1:].ravel(), (theta / A).ravel(), [1.0]])
            feats.append(phi); p_prev = ph
            if readout is not None:
                y = phi @ readout
                y = normalise(y)
                preds.append(y); x = y              # closed loop input for the next step
        return dict(F=np.array(feats), theta=np.array(thetas), phat=np.array(phats), x=np.array(inputs),
                    pred=np.array(preds) if preds else None)


def ridge(F, Y, reg):
    d = F.shape[1]
    return np.linalg.solve(F.T @ F + reg * np.eye(d), F.T @ Y)


REGS = 10.0 ** np.arange(-6, 1)


def fit_readout(F, data, lo, hi, reg=None, val=None):
    """Features at steps lo..hi-1 predict data at lo+1..hi.  If val=(vlo,vhi) choose reg on it."""
    X, Y = F[lo:hi], data[lo + 1:hi + 1]
    if reg is not None:
        return ridge(X, Y, reg), reg
    best = None
    for r in REGS:
        V = ridge(X, Y, r)
        e = np.mean((F[val[0]:val[1]] @ V - data[val[0] + 1:val[1] + 1]) ** 2)
        if best is None or e < best[0]:
            best = (e, r, V)
    return best[2], best[1]


def val_loss(model, w, data, rng_seed):
    """Fit V on washout..(n_train-n_val-1), one-step MSE on the last n_val training steps (all teacher-forced)."""
    cfg = model.cfg
    out = model.run(w, data[:cfg["n_train"]], np.random.default_rng(rng_seed))
    lo, hi = cfg["washout"], cfg["n_train"] - 1 - cfg["n_val"]
    X, Y = out["F"][lo:hi], data[lo + 1:hi + 1]
    Xv, Yv = out["F"][hi:cfg["n_train"] - 1], data[hi + 1:cfg["n_train"]]
    return min(np.mean((Xv @ ridge(X, Y, r) - Yv) ** 2) for r in REGS)


def spsa(model, data, w0, rng, iters, a0, c0, log=print):
    """Simultaneous-perturbation stochastic approximation with common random numbers."""
    w = w0.copy(); hist = []
    A_stab = 0.1 * iters
    best = (np.inf, w.copy())
    for k in range(iters):
        ak = a0 / (k + 1 + A_stab) ** 0.602 * (1 + A_stab) ** 0.602
        ck = c0 / (k + 1) ** 0.101
        delta = rng.choice([-1.0, 1.0], size=w.size)
        seed = int(rng.integers(1 << 30))
        Lp = val_loss(model, w + ck * delta, data, seed)
        Lm = val_loss(model, w - ck * delta, data, seed)
        g = (Lp - Lm) / (2 * ck) * delta
        g = g / max(1.0, np.linalg.norm(g) / 5.0)          # gradient clipping (as in RNN training)
        w = w - ak * g
        L = 0.5 * (Lp + Lm); hist.append(L)
        if L < best[0]:
            best = (L, w.copy())
        if k % 10 == 0 or k == iters - 1:
            log(f"  SPSA iter {k:4d}  val MSE {L:.5f}")
    return w, np.array(hist), best


# ============================================================================== baselines
def ar_baseline(data, n_train, n_pred, washout, order=2, reg=1e-4):
    X = np.array([np.concatenate([data[s - j] for j in range(order)] + [[1.0]]) for s in range(washout, n_train - 1)])
    Y = data[washout + 1:n_train]
    V = ridge(X, Y, reg)
    hist = list(data[:n_train]); preds = []
    for _ in range(n_pred):
        x = np.concatenate([hist[-1 - j] for j in range(order)] + [[1.0]])
        y = x @ V
        y = normalise(y)
        preds.append(y); hist.append(y)
    return np.array(preds)


def esn_baseline(data, n_train, n_pred, washout, n_units, rng, lam=0.5, rho=0.9):
    """Classical leaky echo-state network with the same number of hidden variables and a ridge readout."""
    Win = rng.normal(0, 1, (n_units, 5)); W = rng.normal(0, 1, (n_units, n_units))
    W *= rho / max(abs(np.linalg.eigvals(W)))
    h = np.zeros(n_units); feats = []
    def step(h, x):
        return (1 - lam) * h + lam * np.tanh(W @ h + Win @ np.concatenate([x, [1.0]]))
    for s in range(n_train):
        h = step(h, data[s]); feats.append(np.concatenate([h, [1.0]]))
    F = np.array(feats)
    _, reg = fit_readout(F, data, washout, n_train - 12, val=(n_train - 12, n_train - 1))
    V = ridge(F[washout:n_train - 1], data[washout + 1:n_train], reg)
    preds = []; x = F[-1] @ V
    for _i in range(n_pred):
        x = normalise(x)
        preds.append(x); h = step(h, x); x = np.concatenate([h, [1.0]]) @ V
    return np.array(preds)


# ============================================================================== full experiment
def evaluate(model, w, data, S, seed, n_train, n_pred, washout, reg=None):
    """Teacher-forced run on 1..n_train, fit V, then closed-loop predict n_train+1..n_train+n_pred."""
    tf = model.run(w, data[:n_train], np.random.default_rng(seed), S=S)
    nv = model.cfg["n_val"]
    if reg is None:   # choose ridge strength on the last n_val training steps, then refit on all of them
        _, reg = fit_readout(tf["F"], data, washout, n_train - 1 - nv, val=(n_train - 1 - nv, n_train - 1))
    V = ridge(tf["F"][washout:n_train - 1], data[washout + 1:n_train], reg)
    fit_train = tf["F"][washout:n_train - 1] @ V
    # closed loop: re-run the whole sequence with the same seed so steps 1..n_train are identical
    cl = model.run(w, data[:n_train], np.random.default_rng(seed), S=S, n_teacher=n_train, readout=V,
                   n_steps=n_train + n_pred - 1)
    preds = cl["pred"][n_train - 1:n_train - 1 + n_pred]
    return dict(V=V, reg=reg, fit_train=fit_train, preds=preds, run=cl, tf=tf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out"); ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--S", type=int, default=None); ap.add_argument("--signal", default="quasi")
    ap.add_argument("--R", type=int, default=None); ap.add_argument("--seeds", type=int, default=None)
    args = ap.parse_args()
    cfg = dict(CFG)
    for k, v in (("spsa_iters", args.iters), ("S", args.S), ("R", args.R), ("n_eval_seeds", args.seeds)):
        if v is not None: cfg[k] = v
    os.makedirs(args.out, exist_ok=True)
    nT, nP, wo, nS = cfg["n_train"], cfg["n_pred"], cfg["washout"], cfg["n_eval_seeds"]
    data, phi = make_signal(nT + nP, args.signal)
    truth = data[nT:]
    model = QRNN(cfg)
    rng = np.random.default_rng(cfg["seed"])
    w0 = model.init_params(rng)
    t0 = time.time()
    print(f"[train] {model.n_par} feedback parameters, R={cfg['R']} nodes, S={cfg['S']}")
    L0 = val_loss(model, w0, data, 0)
    w, hist, (Lbest, wbest) = spsa(model, data, w0, rng, cfg["spsa_iters"], cfg["spsa_a"], cfg["spsa_c"])
    L1 = val_loss(model, wbest, data, 0)
    print(f"[train] val MSE {L0:.5f} -> {L1:.5f}  ({time.time()-t0:.0f}s)")

    variants = {"QRNN trained": wbest, "QRNN untrained": w0, "QRNN no feedback": model.no_feedback(wbest)}
    preds, example = {}, {}
    for name, ww in variants.items():
        P = []
        for sd in range(nS):
            ev = evaluate(model, ww, data, cfg["S"], 1000 + sd, nT, nP, wo)
            P.append(ev["preds"])
            if sd == 0: example[name] = ev
        preds[name] = np.array(P)
    preds["persistence"] = np.repeat(data[nT - 1][None, None], nP, 1)
    preds["AR(2)"] = ar_baseline(data, nT, nP, wo)[None]
    preds["ESN (classical)"] = np.array([esn_baseline(data, nT, nP, wo, 4 * cfg["R"], np.random.default_rng(50 + i))
                                         for i in range(nS)])
    metrics = dict(config=cfg, n_feedback_params=model.n_par, val_mse_initial=float(L0), val_mse_final=float(L1),
                   train_time_s=round(time.time() - t0, 1), models={})
    for name, P in preds.items():
        E, F = angle_error(P, truth[None]), fidelity(P, truth[None])
        metrics["models"][name] = dict(angle_err_deg_mean=E.mean(0).round(3).tolist(), angle_err_deg_std=E.std(0).round(3).tolist(),
                                       fidelity_mean=F.mean(0).round(5).tolist(), fidelity_std=F.std(0).round(5).tolist(),
                                       angle_err_deg_avg=float(E.mean()), fidelity_avg=float(F.mean()))
    ex = example["QRNN trained"]
    metrics["train_one_step_angle_err_deg"] = float(angle_error(ex["fit_train"], data[wo + 1:nT]).mean())
    # shot-number sweep (trained feedback weights, readout refit for each S)
    Ssweep = []
    for S in [50, 200, 1000, 5000, 20000, 0]:
        E = [angle_error(evaluate(model, wbest, data, S, 2000 + sd, nT, nP, wo)["preds"], truth).mean()
             for sd in range(nS if S else 1)]
        Ssweep.append(dict(S=S if S else "inf", angle_err_mean=float(np.mean(E)), angle_err_std=float(np.std(E))))
        print(f"[S sweep] S={S or 'inf'}: mean 5-step angle error {np.mean(E):.2f} +- {np.std(E):.2f} deg")
    metrics["S_sweep"] = Ssweep
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    np.savez(os.path.join(args.out, "results.npz"), data=data, phi=phi, w=wbest, w0=w0, hist=hist,
             **{"pred_" + k.replace(" ", "_").replace("(", "").replace(")", ""): v for k, v in preds.items()})
    for name, m in metrics["models"].items():
        print(f"  {name:18s} angle err (deg) per horizon {m['angle_err_deg_mean']}   mean F {m['fidelity_avg']:.4f}")
    import qrnn_plots
    qrnn_plots.make_all(args.out, model, cfg, data, phi, preds, example, hist, Ssweep, metrics)


if __name__ == "__main__":
    main()
