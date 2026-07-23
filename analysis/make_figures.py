#!/usr/bin/env python3
"""Generate result figures from the committed JSON outputs (seeds averaged).

    python -m analysis.make_figures --seeds 0 1 2 --out_dir figures

Produces, under ``figures/``:
  fig1_forgetting.png            output-KL + capability retention per condition (H1)
  fig2_behavioral.png           rhyme accuracy (test / held-out / recovery)
  fig3_decodable_vs_causal.png  Delta_newline (probe) vs handoff H (patching)
  fig4_probe_layers.png         per-layer newline family-decodability
  fig5_patching_layers.png      per-layer causal C: newline vs rhyme word
  fig6_cka_layers.png           per-layer representation similarity to base
  fig7_diversity.png            distinct-2 / self-BLEU / entropy (H5)
  fig8_mech_vs_forget.png       update concentration vs output drift (Section 17)
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONDS = ["corpus_sft", "teacher_sft", "teacher_kd", "onpolicy_kd"]
LABELS = {
    "corpus_sft": "corpus SFT", "teacher_sft": "teacher SFT",
    "teacher_kd": "teacher KD (off-policy)", "onpolicy_kd": "on-policy KD",
    "base": "base",
}
COLORS = {
    "corpus_sft": "#d1495b", "teacher_sft": "#3c6e9e",
    "teacher_kd": "#2a9d8f", "onpolicy_kd": "#6a4c93", "base": "#888888",
}


def load(results_dir, sub, tag):
    p = os.path.join(results_dir, sub, f"{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def seed_stack(results_dir, sub, cond, seeds, extract):
    """Stack an extracted array/scalar over seeds -> np.array [n_seeds, ...]."""
    out = []
    for s in seeds:
        d = load(results_dir, sub, f"{cond}_seed{s}_ckpt_100")
        if d is not None:
            out.append(np.asarray(extract(d), dtype=float))
    return np.stack(out) if out else None


def mean_range(arr):
    return arr.mean(0), arr.min(0), arr.max(0)


def bar_with_err(ax, conds, means, lo, hi, ylabel, title):
    x = np.arange(len(conds))
    yerr = np.vstack([means - lo, hi - means])
    ax.bar(x, means, color=[COLORS[c] for c in conds], yerr=yerr, capsize=4,
           edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in conds], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)


def fig_forgetting(rd, seeds, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    kl = {c: seed_stack(rd, "forgetting", c, seeds, lambda d: d["mean_output_kl"]) for c in CONDS}
    cka = {c: seed_stack(rd, "forgetting", c, seeds, lambda d: d["mean_cka"]) for c in CONDS}
    m = np.array([kl[c].mean() for c in CONDS]); lo = np.array([kl[c].min() for c in CONDS]); hi = np.array([kl[c].max() for c in CONDS])
    bar_with_err(axes[0], CONDS, m, lo, hi, "mean KL(base‖trained) on general text",
                 "Forgetting: output-distribution drift (H1)")
    m2 = np.array([cka[c].mean() for c in CONDS]); lo2 = np.array([cka[c].min() for c in CONDS]); hi2 = np.array([cka[c].max() for c in CONDS])
    bar_with_err(axes[1], CONDS, m2, lo2, hi2, "mean linear CKA to base",
                 "Representation similarity to base")
    axes[1].set_ylim(0.995, 1.0)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_behavioral(rd, seeds, out):
    splits = [("test_id", "test (in-dist.)"), ("test_heldout_family", "held-out family"),
              ("recovery", "recovery prefix")]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(CONDS)); w = 0.25
    for i, (key, lab) in enumerate(splits):
        vals = []
        for c in CONDS:
            st = seed_stack(rd, "behavioral", c, seeds,
                            lambda d: d[key]["rhyme_accuracy"] if key in d else np.nan)
            vals.append(st.mean() if st is not None else np.nan)
        ax.bar(x + (i - 1) * w, vals, w, label=lab, edgecolor="black", linewidth=0.5)
    base = load(rd, "behavioral", "base")
    if base and "test_id" in base:
        ax.axhline(base["test_id"]["rhyme_accuracy"], color="gray", ls="--",
                   label="base (test)")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[c] for c in CONDS], rotation=20, ha="right")
    ax.set_ylabel("rhyme accuracy"); ax.set_ylim(0.6, 1.0)
    ax.set_title("Behavioral rhyme accuracy by split", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_decodable_vs_causal(rd, seeds, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    dn = {c: seed_stack(rd, "probe", c, seeds, lambda d: d["delta_newline"]) for c in CONDS}
    base = load(rd, "probe", "base")
    m, lo, hi = (np.array([dn[c].mean() for c in CONDS]),
                 np.array([dn[c].min() for c in CONDS]),
                 np.array([dn[c].max() for c in CONDS]))
    bar_with_err(axes[0], CONDS, m, lo, hi, r"$\Delta_{newline}$ (family probe)",
                 "Newline is more DECODABLE\nafter teacher supervision")
    if base:
        axes[0].axhline(base["delta_newline"], color="gray", ls="--", label="base")
        axes[0].legend(fontsize=9)
    # causal: mean handoff H by layer, all conds
    for c in CONDS:
        H = seed_stack(rd, "patching", c, seeds, lambda d: d["H_mean_by_layer"])
        if H is None:
            continue
        mh, *_ = mean_range(H)
        xs = np.linspace(0, 1, len(mh))
        axes[1].plot(xs, mh, color=COLORS[c], label=LABELS[c], lw=2)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xlabel("normalized depth (layer / L)")
    axes[1].set_ylabel(r"handoff $H=C_{newline}-C_{rhyme\,word}$")
    axes[1].set_title("...but NOT more CAUSAL\n(H<0: rhyme word still dominates)",
                      fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_probe_layers(rd, seeds, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    for c in CONDS + ["base"]:
        tags = ([f"{c}_seed{s}_ckpt_100" for s in seeds] if c != "base" else ["base"])
        curves = []
        for t in tags:
            d = load(rd, "probe", t)
            if d is None:
                continue
            nl = d["n_layers"]
            curves.append([d["acc"][f"0_{L}"] for L in range(nl)])
        if not curves:
            continue
        m = np.mean(curves, 0)
        ax.plot(range(len(m)), m, color=COLORS[c], lw=2.4 if c != "base" else 1.8,
                ls="-" if c != "base" else "--", label=LABELS[c])
    ax.set_xlabel("layer"); ax.set_ylabel("family-probe accuracy at newline (i=0)")
    ax.set_title("Per-layer newline decodability of the future rhyme family",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_patching_layers(rd, seeds, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    for c in CONDS:
        Cn = seed_stack(rd, "patching", c, seeds, lambda d: d["C_newline_by_layer"])
        Cr = seed_stack(rd, "patching", c, seeds, lambda d: d["C_rhyme_word_by_layer"])
        if Cn is None:
            continue
        xs = np.linspace(0, 1, Cn.shape[1])
        ax.plot(xs, Cn.mean(0), color=COLORS[c], lw=2, label=f"{LABELS[c]} · newline")
        ax.plot(xs, Cr.mean(0), color=COLORS[c], lw=1.4, ls=":", alpha=0.8)
    ax.set_xlabel("normalized depth (layer / L)")
    ax.set_ylabel("P(adopt corrupt rhyme) when patching")
    ax.set_title("Causal patching: newline (solid) vs rhyme word (dotted)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_cka_layers(rd, seeds, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    for c in CONDS:
        st = seed_stack(rd, "forgetting", c, seeds, lambda d: d["cka_per_layer"])
        if st is None:
            continue
        ax.plot(range(st.shape[1]), st.mean(0), color=COLORS[c], lw=2, label=LABELS[c])
    ax.set_xlabel("layer"); ax.set_ylabel("linear CKA to base")
    ax.set_title("Per-layer representation drift on general text",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_diversity(rd, seeds, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    d2 = {c: seed_stack(rd, "diversity", c, seeds, lambda d: d["distinct_2"]) for c in CONDS}
    sb = {c: seed_stack(rd, "diversity", c, seeds, lambda d: d["self_bleu"]) for c in CONDS}
    m, lo, hi = (np.array([d2[c].mean() for c in CONDS]),
                 np.array([d2[c].min() for c in CONDS]), np.array([d2[c].max() for c in CONDS]))
    bar_with_err(axes[0], CONDS, m, lo, hi, "distinct-2 (higher = more diverse)",
                 "Phrasing diversity (H5)")
    m2, lo2, hi2 = (np.array([sb[c].mean() for c in CONDS]),
                    np.array([sb[c].min() for c in CONDS]), np.array([sb[c].max() for c in CONDS]))
    bar_with_err(axes[1], CONDS, m2, lo2, hi2, "self-BLEU (higher = more repetitive)",
                 "Repetition (self-BLEU)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_mech_vs_forget(rd, seeds, out):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    xs, ys = [], []
    for c in CONDS:
        for s in seeds:
            pd = load(rd, "param_drift", f"{c}_seed{s}_ckpt_100")
            fg = load(rd, "forgetting", f"{c}_seed{s}_ckpt_100")
            if pd and fg:
                xs.append(pd["gini_layer_energy"]); ys.append(fg["mean_output_kl"])
                ax.scatter(pd["gini_layer_energy"], fg["mean_output_kl"],
                           color=COLORS[c], s=70, edgecolor="black", zorder=3)
    for c in CONDS:
        ax.scatter([], [], color=COLORS[c], label=LABELS[c])
    if len(xs) >= 3:
        r = np.corrcoef(xs, ys)[0, 1]
        b, a = np.polyfit(xs, ys, 1)
        xx = np.linspace(min(xs), max(xs), 50)
        ax.plot(xx, b * xx + a, color="black", ls="--", lw=1,
                label=f"fit (r = {r:.2f})")
    ax.set_xlabel("parameter-update concentration (Gini over layers)")
    ax.set_ylabel("output-distribution KL vs base")
    ax.set_title("Update concentration predicts forgetting (Section 17)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def _per_pair_matrix(rd, tag, seeds, key):
    """(n_layers, n_pairs) corrupt-rhyme matrix, pooling pairs across seeds.

    ``key`` is "newline" or "rhyme_word". Falls back to a single un-seeded file
    (e.g. base_teacher / base12b)."""
    import glob as _glob
    paths = sorted(_glob.glob(os.path.join(rd, "patching", f"{tag}_seed*_ckpt_100.json")))
    if not paths:
        p = os.path.join(rd, "patching", f"{tag}.json")
        paths = [p] if os.path.exists(p) else []
    cols = []
    for p in paths:
        d = json.load(open(p))
        for pair in d["pairs"]:
            cols.append(pair["C"][key])
    if not cols:
        return None
    return np.array(cols).T  # (n_layers, n_pairs*seeds)


def _cluster_bootstrap(mat, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    n_layers, n_pairs = mat.shape
    idx = rng.integers(0, n_pairs, size=(n_boot, n_pairs))
    boot = mat[:, idx].mean(axis=2).T
    return mat.mean(1), np.percentile(boot, 2.5, 0), np.percentile(boot, 97.5, 0)


def fig_patching_perlayer(rd, seeds, out):
    """look-ahead-style per-layer grouped bars: rhyme word (i=-2) vs newline (i=0),
    corrupt-rhyme rate with cluster-bootstrap 95% CIs over prompt pairs."""
    panels = [
        ("base_teacher", "Gemma-3-27B  (teacher)"),
        ("base", "Gemma-3-4B  (base student)"),
        ("base12b", "Gemma-3-12B  (base student)"),
        ("teacher_sft", "Gemma-3-4B  (teacher-SFT)"),
    ]
    panels = [(t, lab) for t, lab in panels if _per_pair_matrix(rd, t, seeds, "newline") is not None]
    ncol = 2
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 4.2 * nrow))
    axes = np.array(axes).reshape(-1)
    C_RW, C_NL = "#6699cc", "#cc6677"
    for ax, (tag, label) in zip(axes, panels):
        rw = _per_pair_matrix(rd, tag, seeds, "rhyme_word")
        nl = _per_pair_matrix(rd, tag, seeds, "newline")
        m_rw, lo_rw, hi_rw = _cluster_bootstrap(rw)
        m_nl, lo_nl, hi_nl = _cluster_bootstrap(nl)
        n = len(m_rw); x = np.arange(n); w = 0.42
        ax.bar(x - w / 2, m_rw, w, color=C_RW, label="i=-2 (rhyme word)",
               yerr=np.vstack([m_rw - lo_rw, hi_rw - m_rw]),
               error_kw=dict(ecolor="black", lw=0.5, capsize=1))
        ax.bar(x + w / 2, m_nl, w, color=C_NL, label="i=0 (newline)",
               yerr=np.vstack([m_nl - lo_nl, hi_nl - m_nl]),
               error_kw=dict(ecolor="black", lw=0.5, capsize=1))
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=12); ax.set_ylabel("Corrupt rhyme rate", fontsize=12)
        ax.set_ylim(0, 1.05); ax.set_xlim(-1, n)
        ax.set_xticks(np.arange(0, n, max(1, n // 12)))
        ax.grid(axis="y", ls="--", alpha=0.4)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(loc="upper left", frameon=False, fontsize=11)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    fig.suptitle("Activation patching (corrupt→clean): where does the rhyme live causally?",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.savefig(out, dpi=180); plt.close(fig)


def fig_handoff_inheritance(rd, seeds, out):
    """Causal handoff H vs depth: 4B conditions (flat, C_nl=0) vs 27B teacher."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for c in CONDS + ["base"]:
        H = seed_stack(rd, "patching", c, seeds,
                       lambda d: np.array(d["C_newline_by_layer"]) - np.array(d["C_rhyme_word_by_layer"])) \
            if c != "base" else None
        if c == "base":
            d = load(rd, "patching", "base")
            if d is None:
                continue
            H = (np.array(d["C_newline_by_layer"]) - np.array(d["C_rhyme_word_by_layer"]))[None, :]
        if H is None:
            continue
        m = H.mean(0)
        ax.plot(np.linspace(0, 1, len(m)), m, color=COLORS[c], lw=2,
                ls="-" if c != "base" else "--", label=LABELS[c])
    teach = load(rd, "patching", "base_teacher")
    if teach:
        Ht = np.array(teach["C_newline_by_layer"]) - np.array(teach["C_rhyme_word_by_layer"])
        ax.plot(np.linspace(0, 1, len(Ht)), Ht, color="black", lw=3,
                label="27B TEACHER", zorder=5)
    ax.axhline(0, color="gray", lw=1)
    ax.annotate("teacher's causal\nrhyme-word→newline handoff\n(H>0)",
                xy=(0.52, 0.55), xytext=(0.15, 0.4), fontsize=10,
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.set_xlabel("normalized depth (layer / L)")
    ax.set_ylabel(r"handoff $H = C_{newline} - C_{rhyme\,word}$")
    ax.set_title("Mechanistic inheritance FAILS: no 4B regime acquires the\n"
                 "teacher's causal newline site (Section 16)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_matched(rd, seeds, out):
    """Output-KL at ckpt_100 vs at matched performance, per condition."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CONDS)); w = 0.38
    for j, (kind, lab, hatch) in enumerate([("ckpt_100", "at ckpt_100 (unmatched)", ""),
                                            ("matched", "at matched rhyme ≈0.86", "//")]):
        means, lo, hi = [], [], []
        for c in CONDS:
            st = seed_stack(rd, "forgetting", c, seeds,
                            lambda d: d["mean_output_kl"])  # noqa
            # override tag suffix: reload with the right kind
            vals = []
            for s in seeds:
                d = load(rd, "forgetting", f"{c}_seed{s}_{kind}")
                if d:
                    vals.append(d["mean_output_kl"])
            vals = np.array(vals)
            means.append(vals.mean()); lo.append(vals.min()); hi.append(vals.max())
        means, lo, hi = map(np.array, (means, lo, hi))
        ax.bar(x + (j - 0.5) * w, means, w, label=lab, hatch=hatch,
               color=[COLORS[c] for c in CONDS], edgecolor="black", linewidth=0.6,
               alpha=0.7 if j else 1.0,
               yerr=np.vstack([means - lo, hi - means]), capsize=3)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[c] for c in CONDS], rotation=20, ha="right")
    ax.set_ylabel("output-distribution KL vs base")
    ax.set_title("Forgetting at matched performance (H1/H4 fair test)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rd, seeds, od = args.results_dir, args.seeds, args.out_dir
    fig_forgetting(rd, seeds, f"{od}/fig1_forgetting.png")
    fig_behavioral(rd, seeds, f"{od}/fig2_behavioral.png")
    fig_decodable_vs_causal(rd, seeds, f"{od}/fig3_decodable_vs_causal.png")
    fig_probe_layers(rd, seeds, f"{od}/fig4_probe_layers.png")
    fig_patching_layers(rd, seeds, f"{od}/fig5_patching_layers.png")
    fig_cka_layers(rd, seeds, f"{od}/fig6_cka_layers.png")
    fig_diversity(rd, seeds, f"{od}/fig7_diversity.png")
    fig_mech_vs_forget(rd, seeds, f"{od}/fig8_mech_vs_forget.png")
    fig_patching_perlayer(rd, seeds, f"{od}/fig5b_patching_perlayer.png")
    if load(rd, "patching", "base_teacher"):
        fig_handoff_inheritance(rd, seeds, f"{od}/fig10_handoff_inheritance.png")
    if all(load(rd, "forgetting", f"corpus_sft_seed{s}_matched") for s in seeds):
        fig_matched(rd, seeds, f"{od}/fig9_matched.png")
        print(f"Wrote 9 figures to {od}/")
    else:
        print(f"Wrote 8 figures to {od}/ (matched not found; run scripts/run_matched.sh)")


if __name__ == "__main__":
    main()
