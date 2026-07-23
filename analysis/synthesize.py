#!/usr/bin/env python3
"""Phase 6: synthesis — connect mechanism to forgetting (outline Sections 16-17).

Aggregates the per-condition result JSONs into one table and tests the central
association: is greater forgetting (higher general-text KL, lower CKA, larger /
more diffuse parameter updates) tied to planning-site reorganization (a newline
handoff, weakened rhyme-word pathway) rather than reuse of the base pathway?

Also reports mechanistic inheritance at *normalized depth* r = layer / L
(Section 16), so student and teacher handoff depths are comparable despite
different layer counts.

Reads from results/{behavioral,diversity,forgetting,param_drift,probe,patching}/
and runs/*/history.json. Robust to missing files (prints what it has).

Usage:
  python -m analysis.synthesize --conditions corpus_sft teacher_sft teacher_kd onpolicy_kd \
      --seed 0 --out results/synthesis.json
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from opi.io_utils import read_json, write_json  # noqa: E402


def _load(path):
    return read_json(path) if os.path.exists(path) else None


def _peak_normalized(values):
    """(peak value, normalized-depth argmax) for a per-layer list."""
    if not values:
        return None, None
    n = len(values)
    amax = max(range(n), key=lambda i: values[i])
    return values[amax], amax / max(1, n - 1)


def collect(condition, seed, results_dir, runs_dir):
    ck = "ckpt_100"
    tag = f"{condition}_{ck}"
    row = {"condition": condition, "seed": seed}

    hist = _load(os.path.join(runs_dir, f"{condition}_seed{seed}", "history.json"))
    if hist and hist.get("checkpoints"):
        row["final_val_rhyme"] = hist["checkpoints"][-1]["val_rhyme"]

    beh = _load(os.path.join(results_dir, "behavioral", f"{tag}.json"))
    if beh:
        for split in ["test_id", "test_heldout_family", "recovery"]:
            if split in beh:
                row[f"rhyme_{split}"] = beh[split]["rhyme_accuracy"]

    div = _load(os.path.join(results_dir, "diversity", f"{tag}.json"))
    if div:
        row["final_word_entropy"] = div["final_word_entropy_bits"]
        row["self_bleu"] = div["self_bleu"]
        row["distinct_2"] = div["distinct_2"]

    forg = _load(os.path.join(results_dir, "forgetting", f"{tag}.json"))
    if forg:
        row["output_kl"] = forg["mean_output_kl"]
        row["mean_cka"] = forg["mean_cka"]
        row["min_cka"] = forg["min_cka"]
        row["capability_delta"] = forg["capability_delta"]

    pd = _load(os.path.join(results_dir, "param_drift", f"{tag}.json"))
    if pd:
        row["update_norm"] = pd["total_update_norm"]
        row["update_gini"] = pd["gini_layer_energy"]
        row["attn_mlp_ratio"] = pd["attn_mlp_ratio"]

    prb = _load(os.path.join(results_dir, "probe", f"{tag}.json"))
    if prb:
        row["delta_newline"] = prb["delta_newline"]
        row["probe_peak_newline"] = prb["peak_acc_by_offset"].get("0")

    pat = _load(os.path.join(results_dir, "patching", f"{tag}.json"))
    if pat:
        row["handoff"] = pat["handoff"]["handoff"]
        row["peak_newline_C"] = pat["peak_newline_C"]
        row["peak_rhyme_word_C"] = pat["peak_rhyme_word_C"]
        peak_C, norm_depth = _peak_normalized(pat["C_newline_by_layer"])
        row["newline_C_norm_depth"] = norm_depth
    return row


def correlate(rows, x_key, y_key):
    """Pearson r across conditions (illustrative with few points)."""
    pts = [(r[x_key], r[y_key]) for r in rows if r.get(x_key) is not None and r.get(y_key) is not None]
    if len(pts) < 3:
        return None
    xs, ys = zip(*pts)
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in pts)
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx and vy else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conditions", nargs="+",
                    default=["corpus_sft", "teacher_sft", "teacher_kd", "onpolicy_kd"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--teacher_patching", default="results/patching/base_teacher.json",
                    help="Optional teacher patching result for inheritance depth.")
    ap.add_argument("--out", default="results/synthesis.json")
    args = ap.parse_args()

    rows = [collect(c, args.seed, args.results_dir, args.runs_dir) for c in args.conditions]

    # Central Section-17 associations (across conditions).
    associations = {
        "handoff_vs_output_kl": correlate(
            rows, "peak_newline_C", "output_kl"),
        "newlineC_vs_mean_cka": correlate(
            rows, "peak_newline_C", "mean_cka"),
        "rhymeword_reuse_vs_cka": correlate(
            rows, "peak_rhyme_word_C", "mean_cka"),
        "update_gini_vs_output_kl": correlate(
            rows, "update_gini", "output_kl"),
    }

    # Inheritance: normalized handoff depth vs teacher (Section 16).
    inheritance = {}
    teach = _load(args.teacher_patching)
    if teach:
        _, t_depth = _peak_normalized(teach["C_newline_by_layer"])
        inheritance["teacher_newline_C_norm_depth"] = t_depth
        for r in rows:
            if r.get("newline_C_norm_depth") is not None and t_depth is not None:
                inheritance[f"{r['condition']}_depth_gap"] = abs(r["newline_C_norm_depth"] - t_depth)

    out = {"seed": args.seed, "rows": rows,
           "associations": associations, "inheritance": inheritance}
    write_json(args.out, out)

    # console table
    cols = ["condition", "rhyme_test_id", "rhyme_recovery", "final_word_entropy",
            "output_kl", "mean_cka", "capability_delta",
            "delta_newline", "peak_newline_C", "peak_rhyme_word_C", "handoff"]
    print("\n" + "  ".join(f"{c:>16}" for c in cols))
    for r in rows:
        print("  ".join(
            f"{(r.get(c) if not isinstance(r.get(c), float) else round(r.get(c),3)):>16}"
            if r.get(c) is not None else f"{'-':>16}" for c in cols))
    print("\nAssociations (Pearson r across conditions):")
    for k, v in associations.items():
        print(f"  {k:32s} {v if v is None else round(v,3)}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
