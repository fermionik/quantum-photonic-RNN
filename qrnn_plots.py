"""Figures for the two-frame optomechanical QRNN (called from qrnn_learn.main, or re-run: python qrnn_plots.py out)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qrnn_physics import psi

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#9b9a95"
COL = {"QRNN trained": PAL[0], "QRNN no feedback": PAL[1], "QRNN untrained": PAL[6],
       "persistence": MUTED, "AR(2)": PAL[3], "ESN (classical)": PAL[2]}
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "normal", "legend.frameon": False})


def style(ax):
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2); ax.xaxis.label.set_color(INK2); ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def bloch_to_phi(r):
    return 0.5 * np.arctan2(r[..., 1], r[..., 0])


def seq_phi(R):
    """(n,4) Bloch pairs -> (2n,) mode angle per frame (unwrapped, degrees)."""
    return np.degrees(np.stack([bloch_to_phi(R[:, :2]), bloch_to_phi(R[:, 2:])], 1).ravel())


def make_all(out, model, cfg, data, phi, preds, example, hist, Ssweep, metrics):
    nT, nP, wo = cfg["n_train"], cfg["n_pred"], cfg["washout"]
    node = model.node; T = node.T
    files = []

    # ---------------------------------------------------------------- 1. encoding and control pulses
    t = np.linspace(0, 2 * T, 800)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axs[0]
    s = nT // 2
    for f in range(2):
        a = np.radians(seq_phi(data[s:s + 1])[f])
        tt = t[(t >= f * T) & (t <= (f + 1) * T)]
        ax.plot(tt, np.cos(a) * psi(0, tt - (f + .5) * T) + np.sin(a) * psi(1, tt - (f + .5) * T), color=PAL[0], lw=2,
                label=r"photon modes $\xi^{(2s-1)},\ \xi^{(2s)}$" if f == 0 else None)
    ax.axvline(T, color=MUTED, lw=1, ls="--")
    ax.set_ylim(-0.8, 1.05)
    ax.text(T / 2, 0.92, "frame 2s-1", ha="center", color=INK2); ax.text(1.5 * T, 0.92, "frame 2s", ha="center", color=INK2)
    ax.set_xlabel("t"); ax.set_title(f"Data: two single photons, HG modes $\\psi_0,\\psi_1$ (step s={s})", loc="left")
    ax.legend(loc="lower left", fontsize=9)
    ax = axs[1]
    th = example["QRNN trained"]["tf"]["theta"][s]
    ax.plot(t, psi(0, (t - T) / node.sc) / np.sqrt(node.sc), color=PAL[1], lw=1.5, ls="--", label=r"$\Phi_0$")
    ax.plot(t, psi(1, (t - T) / node.sc) / np.sqrt(node.sc), color=PAL[3], lw=1.5, ls="--", label=r"$\Phi_1$")
    for r in range(min(3, len(th))):
        al = th[r, 0] * psi(0, (t - T) / node.sc) / np.sqrt(node.sc) + th[r, 1] * psi(1, (t - T) / node.sc) / np.sqrt(node.sc)
        ax.plot(t, al, color=PAL[[0, 2, 6][r]], lw=2, label=f"$\\alpha_s(t)$, node {r+1}")
    ax.axvline(T, color=MUTED, lw=1, ls="--"); ax.set_xlabel("t")
    ax.set_title(r"Control pulses $\alpha_s^r(t)=\theta^r_{s,0}\Phi_0+\theta^r_{s,1}\Phi_1$ (weights)", loc="left")
    ax.set_ylim(None, ax.get_ylim()[1] * 1.45); ax.legend(fontsize=8, ncol=3, loc="upper left")
    for ax in axs: style(ax)
    fig.tight_layout(); files.append(os.path.join(out, "fig1_encoding_and_control.png")); fig.savefig(files[-1], dpi=140); plt.close(fig)

    # ---------------------------------------------------------------- 2. training curve
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    it = np.arange(len(hist))
    ax.plot(it, hist, color=PAL[0], lw=1, alpha=0.35, label="SPSA loss (noisy, per iteration)")
    k = max(1, min(10, len(hist) // 3)); sm = np.convolve(hist, np.ones(k) / k, mode="valid")
    ax.plot(it[k - 1:], sm, color=PAL[0], lw=2, label=f"running mean ({k})")
    ax.set_xlabel("SPSA iteration"); ax.set_ylabel("validation MSE (Bloch vector)")
    ax.set_yscale("log"); ax.set_title("Training the feedback weights $W_\\theta, W_p, b$", loc="left"); ax.legend(fontsize=9)
    style(ax); fig.tight_layout(); files.append(os.path.join(out, "fig2_training_curve.png")); fig.savefig(files[-1], dpi=140); plt.close(fig)

    # ---------------------------------------------------------------- 3. prediction vs truth
    fig, axs = plt.subplots(2, 1, figsize=(11, 6.8), gridspec_kw=dict(height_ratios=[1.3, 1]))
    frames = np.arange(1, 2 * (nT + nP) + 1)
    truth_phi = np.degrees(phi.ravel())
    ex = example["QRNN trained"]
    fit_phi = seq_phi(ex["fit_train"])
    fit_frames = np.arange(2 * (wo + 1) + 1, 2 * nT + 1)
    P = preds["QRNN trained"]
    pphi = np.array([seq_phi(p) for p in P])
    pf = np.arange(2 * nT + 1, 2 * (nT + nP) + 1)
    for ax, lo in ((axs[0], 1), (axs[1], 2 * nT - 15)):
        m = frames >= lo
        ax.axvspan(2 * nT + 0.5, 2 * (nT + nP) + 0.5, color="#f1f0ec", lw=0)
        ax.plot(frames[m], truth_phi[m], color=INK, lw=2, label="data (true mode angle $\\phi_k$)")
        mf = fit_frames >= lo
        ax.plot(fit_frames[mf], fit_phi[mf], color=PAL[0], lw=0, marker="o", ms=4, alpha=0.8,
                label="QRNN one-step prediction (training, teacher forced)")
        ax.fill_between(pf, pphi.mean(0) - pphi.std(0), pphi.mean(0) + pphi.std(0), color=PAL[1], alpha=0.25, lw=0)
        ax.plot(pf, pphi.mean(0), color=PAL[1], lw=2, marker="s", ms=5, label=f"QRNN closed-loop forecast (mean ± sd, {len(P)} shot-noise runs)")
        esn = np.array([seq_phi(p) for p in preds["ESN (classical)"]]).mean(0)
        ax.plot(pf, esn, color=PAL[2], lw=1.5, ls="--", label="classical ESN forecast")
        ax.axvline(2 * nT + 0.5, color=MUTED, lw=1, ls="--")
        ax.set_ylabel("mode angle $\\phi$ (deg)"); style(ax)
    axs[0].set_title(f"Prediction of the HG superposition: {nT} training double frames, {nP}-step closed-loop forecast (S={cfg['S']})", loc="left")
    axs[0].set_ylim(0, 118); axs[0].legend(fontsize=8, loc="upper left", ncol=2)
    axs[1].set_xlabel("time frame k  (double frame s covers k = 2s-1, 2s)"); axs[1].set_title("Zoom on the forecast window", loc="left")
    fig.tight_layout(); files.append(os.path.join(out, "fig3_prediction.png")); fig.savefig(files[-1], dpi=140); plt.close(fig)

    # ---------------------------------------------------------------- 4. performance metrics vs horizon
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    h = np.arange(1, nP + 1)
    for name, mm in metrics["models"].items():
        c = COL.get(name, INK2)
        ls = "-" if name.startswith("QRNN") else "--"
        for ax, key in ((axs[0], "angle_err_deg"), (axs[1], "fidelity")):
            mu, sd = np.array(mm[key + "_mean"]), np.array(mm[key + "_std"])
            ax.plot(h, mu, color=c, lw=2, ls=ls, marker="o", ms=6, label=name)
            if sd.any(): ax.fill_between(h, mu - sd, mu + sd, color=c, alpha=0.15, lw=0)
    axs[0].set_ylabel("RMS mode-angle error (deg)"); axs[0].set_title("Forecast error vs horizon", loc="left")
    axs[1].set_ylabel(r"mode fidelity $|\langle\hat\xi|\xi\rangle|^2$"); axs[1].set_title("Forecast fidelity vs horizon", loc="left")
    axs[0].set_yscale("log")
    for ax in axs:
        ax.set_xlabel("prediction step (double frames ahead)"); ax.set_xticks(h); style(ax)
    axs[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout(); files.append(os.path.join(out, "fig4_metrics_vs_horizon.png")); fig.savefig(files[-1], dpi=140); plt.close(fig)

    # ---------------------------------------------------------------- 5. shot-noise sweep
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    Ss = [d["S"] for d in Ssweep if d["S"] != "inf"]
    mu = np.array([d["angle_err_mean"] for d in Ssweep if d["S"] != "inf"]); sd = np.array([d["angle_err_std"] for d in Ssweep if d["S"] != "inf"])
    ax.errorbar(Ss, mu, yerr=sd, color=PAL[0], lw=2, marker="o", ms=6, capsize=3, label="QRNN trained (finite S)")
    inf = [d for d in Ssweep if d["S"] == "inf"][0]["angle_err_mean"]
    ax.axhline(inf, color=PAL[0], ls=":", lw=1.5, label="exact probabilities ($S\\to\\infty$)")
    ax.axhline(metrics["models"]["persistence"]["angle_err_deg_avg"], color=MUTED, ls="--", lw=1.5, label="persistence baseline")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("repetitions S per double frame")
    ax.set_ylabel("mean 5-step angle error (deg)"); ax.set_title("Effect of shot noise", loc="left"); ax.legend(fontsize=8, loc="center left")
    style(ax); fig.tight_layout(); files.append(os.path.join(out, "fig5_shot_noise.png")); fig.savefig(files[-1], dpi=140); plt.close(fig)

    # ---------------------------------------------------------------- 6. hidden state
    run = ex["run"]
    fig, axs = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
    steps = np.arange(1, len(run["theta"]) + 1)
    for r in range(run["theta"].shape[1]):
        axs[0].plot(steps, run["theta"][:, r, 0], color=PAL[r % 8], lw=1.8, label=f"node {r+1}")
        axs[0].plot(steps, run["theta"][:, r, 1], color=PAL[r % 8], lw=1.2, ls="--")
        axs[1].plot(steps, run["phat"][:, r, 1], color=PAL[r % 8], lw=1.8)
        axs[1].plot(steps, run["phat"][:, r, 2], color=PAL[r % 8], lw=1.2, ls=":")
    for ax in axs:
        ax.axvspan(nT + 0.5, nT + nP - 0.5, color="#f1f0ec", lw=0); style(ax)
    axs[0].set_ylabel(r"$\theta_{s,0}$ (solid), $\theta_{s,1}$ (dashed)")
    axs[0].set_title("Classical hidden state: control-pulse weights of the R nodes", loc="left")
    axs[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    axs[1].set_ylabel(r"$\hat P_b(1)$ (solid), $\hat P_b(2)$ (dotted)"); axs[1].set_xlabel("double frame s")
    axs[1].set_title("Measured phonon statistics (network output / activation)", loc="left")
    fig.tight_layout(); files.append(os.path.join(out, "fig6_hidden_state.png")); fig.savefig(files[-1], dpi=140); plt.close(fig)

    # ---------------------------------------------------------------- 7. reconstructed waveforms
    fig, axs = plt.subplots(1, nP, figsize=(13, 2.9), sharey=True)
    tl = np.linspace(0, 2 * T, 400)
    for i in range(nP):
        ax = axs[i]
        for f in range(2):
            tt = tl[(tl >= f * T) & (tl <= (f + 1) * T)]
            a_t = phi[nT + i, f]; a_p = np.radians(pphi.mean(0)[2 * i + f])
            ax.plot(tt, np.cos(a_t) * psi(0, tt - (f + .5) * T) + np.sin(a_t) * psi(1, tt - (f + .5) * T), color=INK, lw=2,
                    label="true" if f == 0 else None)
            ax.plot(tt, np.cos(a_p) * psi(0, tt - (f + .5) * T) + np.sin(a_p) * psi(1, tt - (f + .5) * T), color=PAL[1], lw=2, ls="--",
                    label="forecast" if f == 0 else None)
        ax.axvline(T, color=MUTED, lw=1, ls=":"); ax.set_title(f"s = {nT+i+1}  (+{i+1})", loc="left"); ax.set_xlabel("t"); style(ax)
    axs[0].set_ylabel(r"$\xi(t)$"); axs[0].legend(fontsize=8)
    fig.tight_layout(); files.append(os.path.join(out, "fig7_forecast_waveforms.png")); fig.savefig(files[-1], dpi=140); plt.close(fig)
    print("[plots]", *files, sep="\n  ")
    return files


def replot(out="out"):
    """Re-draw all figures from out/results.npz + out/metrics.json (only the example runs are recomputed)."""
    import json
    from qrnn_learn import QRNN, evaluate, CFG
    m = json.load(open(os.path.join(out, "metrics.json"))); cfg = dict(CFG); cfg.update(m["config"])
    d = np.load(os.path.join(out, "results.npz"))
    model = QRNN(cfg)
    names = {"QRNN trained": "pred_QRNN_trained", "QRNN untrained": "pred_QRNN_untrained",
             "QRNN no feedback": "pred_QRNN_no_feedback", "persistence": "pred_persistence",
             "AR(2)": "pred_AR2", "ESN (classical)": "pred_ESN_classical"}
    preds = {k: d[v] for k, v in names.items()}
    ws = {"QRNN trained": d["w"], "QRNN untrained": d["w0"], "QRNN no feedback": model.no_feedback(d["w"])}
    example = {k: evaluate(model, w, d["data"], cfg["S"], 1000, cfg["n_train"], cfg["n_pred"], cfg["washout"]) for k, w in ws.items()}
    return make_all(out, model, cfg, d["data"], d["phi"], preds, example, d["hist"], m["S_sweep"], m)


if __name__ == "__main__":
    import sys
    replot(sys.argv[1] if len(sys.argv) > 1 else "out")
