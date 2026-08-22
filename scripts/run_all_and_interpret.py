import argparse
import os
import subprocess
import sys
import pandas as pd
import numpy as np

def run_cmd(cmd):
    print(f"[RUN] {' '.join(cmd)}")
    p = subprocess.run(cmd, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

def safe_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def summarize(df, tag):
    g = df.groupby("method", as_index=False).mean(numeric_only=True)
    g["source"] = tag
    return g

def rank_methods(df):
    agg = df.groupby("method", as_index=False).agg({
        "translation_error": "mean",
        "rotation_error_deg": "mean",
        "final_residual": "mean",
        "lambda_min_H": "mean",
        "kappa_H": "mean",
        "failure": "mean",
    })

    def norm_col(arr, higher_better=False):
        arr = np.array(arr, dtype=float)
        mn, mx = arr.min(), arr.max()
        if abs(mx - mn) < 1e-12:
            z = np.zeros_like(arr)
        else:
            z = (arr - mn) / (mx - mn)
        return 1 - z if higher_better else z

    score = (
        0.30 * norm_col(agg["translation_error"], higher_better=False) +
        0.25 * norm_col(agg["rotation_error_deg"], higher_better=False) +
        0.15 * norm_col(agg["final_residual"], higher_better=False) +
        0.10 * norm_col(agg["kappa_H"], higher_better=False) +
        0.10 * norm_col(agg["failure"], higher_better=False) +
        0.10 * norm_col(agg["lambda_min_H"], higher_better=True)
    )
    agg["score_lower_better"] = score
    agg = agg.sort_values("score_lower_better", ascending=True).reset_index(drop=True)
    agg["rank"] = np.arange(1, len(agg)+1)
    return agg

def winloss_vs_baseline(df, baseline="conf_unc"):
    rows = []
    scenarios = sorted(df["scenario"].unique())
    methods = sorted(df["method"].unique())
    for m in methods:
        if m == baseline:
            continue
        wt = wr = 0
        for sc in scenarios:
            a = df[(df["scenario"]==sc)&(df["method"]==m)].iloc[0]
            b = df[(df["scenario"]==sc)&(df["method"]==baseline)].iloc[0]
            if a["translation_error"] < b["translation_error"]:
                wt += 1
            if a["rotation_error_deg"] < b["rotation_error_deg"]:
                wr += 1
        rows.append({
            "method": m,
            "wins_translation_vs_conf_unc": wt,
            "wins_rotation_vs_conf_unc": wr,
            "num_scenarios": len(scenarios)
        })
    return pd.DataFrame(rows)

def relative_improvement_vs_baseline(df, baseline="conf_unc"):
    rows = []
    scenarios = sorted(df["scenario"].unique())
    methods = sorted(df["method"].unique())
    for m in methods:
        if m == baseline:
            continue
        imp_t, imp_r = [], []
        for sc in scenarios:
            a = df[(df["scenario"]==sc)&(df["method"]==m)].iloc[0]
            b = df[(df["scenario"]==sc)&(df["method"]==baseline)].iloc[0]
            imp_t.append((b["translation_error"] - a["translation_error"]) / max(b["translation_error"], 1e-12))
            imp_r.append((b["rotation_error_deg"] - a["rotation_error_deg"]) / max(b["rotation_error_deg"], 1e-12))
        rows.append({
            "method": m,
            "mean_rel_impr_translation_vs_conf_unc": float(np.mean(imp_t)),
            "mean_rel_impr_rotation_vs_conf_unc": float(np.mean(imp_r)),
        })
    return pd.DataFrame(rows)

def extract_key_findings(df_v1, df_v2):
    # Focus on IG methods
    ig_methods = [m for m in df_v2["method"].unique() if "ig_" in m]
    baseline = "conf_unc"

    def mean_metric(df, method, col):
        d = df[df["method"]==method][col]
        return float(d.mean()) if len(d) else np.nan

    findings = []
    for m in ig_methods:
        t1 = mean_metric(df_v1, m, "translation_error") if m in df_v1["method"].values else np.nan
        t2 = mean_metric(df_v2, m, "translation_error")
        r1 = mean_metric(df_v1, m, "rotation_error_deg") if m in df_v1["method"].values else np.nan
        r2 = mean_metric(df_v2, m, "rotation_error_deg")
        b2t = mean_metric(df_v2, baseline, "translation_error")
        b2r = mean_metric(df_v2, baseline, "rotation_error_deg")
        findings.append({
            "method": m,
            "v2_translation_error": t2,
            "v2_rotation_error_deg": r2,
            "v2_vs_conf_unc_rel_impr_t": (b2t - t2) / max(b2t, 1e-12),
            "v2_vs_conf_unc_rel_impr_r": (b2r - r2) / max(b2r, 1e-12),
            "v1_translation_error_if_available": t1,
            "v1_rotation_error_deg_if_available": r1,
        })
    return pd.DataFrame(findings)

def final_verdict(df_v2, relimp_v2, ranking_v2):
    # Simple rule-based verdict
    ig_rows = relimp_v2[relimp_v2["method"].str.contains("ig_", na=False)]
    if len(ig_rows)==0:
        return "INCONCLUSIVE: no IG method found in V2."

    good_t = (ig_rows["mean_rel_impr_translation_vs_conf_unc"] > 0.03).any()
    good_r = (ig_rows["mean_rel_impr_rotation_vs_conf_unc"] > 0.03).any()

    top_method = ranking_v2.iloc[0]["method"]
    ig_top = "ig_" in str(top_method)

    if good_t and good_r and ig_top:
        return "JUSTIFIED: IG-based weighting shows consistent practical gains and top overall ranking."
    if (good_t or good_r):
        return "PROMISING BUT PARTIAL: IG helps on some metrics/scenarios; needs tuning and more trials."
    return "NOT YET JUSTIFIED: IG does not beat conf_unc consistently in this run."

def write_report(out_path, args, v1_df, v2_df, rank1, rank2, wl1, wl2, ri1, ri2, findings, verdict):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Final Combined Report: Experiment V1 + V2\n\n")
        f.write("## Run Config\n\n")
        f.write(f"- seed: {args.seed}\n")
        f.write(f"- num_points: {args.num_points}\n")
        f.write(f"- noise_px: {args.noise_px}\n")
        f.write(f"- out_v1: {args.out_v1}\n")
        f.write(f"- out_v2: {args.out_v2}\n\n")

        f.write("## V1 Summary (mean by method)\n\n")
        f.write(v1_df.groupby("method", as_index=False).mean(numeric_only=True).to_markdown(index=False))
        f.write("\n\n## V2 Summary (mean by method)\n\n")
        f.write(v2_df.groupby("method", as_index=False).mean(numeric_only=True).to_markdown(index=False))

        f.write("\n\n## V1 Ranking\n\n")
        f.write(rank1.to_markdown(index=False))
        f.write("\n\n## V2 Ranking\n\n")
        f.write(rank2.to_markdown(index=False))

        f.write("\n\n## Win/Loss vs conf_unc (V1)\n\n")
        f.write(wl1.to_markdown(index=False))
        f.write("\n\n## Win/Loss vs conf_unc (V2)\n\n")
        f.write(wl2.to_markdown(index=False))

        f.write("\n\n## Relative Improvement vs conf_unc (V1)\n\n")
        f.write(ri1.to_markdown(index=False))
        f.write("\n\n## Relative Improvement vs conf_unc (V2)\n\n")
        f.write(ri2.to_markdown(index=False))

        f.write("\n\n## IG-focused Cross-Version Findings\n\n")
        f.write(findings.to_markdown(index=False))

        f.write("\n\n## Final Verdict\n\n")
        f.write(f"**{verdict}**\n")

        f.write("\n\n## Interpretation Notes\n\n")
        f.write("- در سناریوهای degenerate (pure_rotation/forward_motion/low_parallax) بهبود IG مهم‌تر است.\n")
        f.write("- اگر conditioning بهتر شود (`lambda_min ↑`, `kappa ↓`) ولی pose بهتر نشود، tuning لازم است.\n")
        f.write("- V2 به دلیل robust Huber معمولاً در outlier_contamination پایدارتر از V1 است.\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-points", type=int, default=140)
    ap.add_argument("--noise-px", type=float, default=1.0)
    ap.add_argument("--out-v1", type=str, default="results_v1")
    ap.add_argument("--out-v2", type=str, default="results_v2")
    ap.add_argument("--out-final", type=str, default="results_final")
    args = ap.parse_args()

    os.makedirs(args.out_final, exist_ok=True)

    # Run V1
    run_cmd([
        sys.executable, "scripts/run_experiments.py",
        "--seed", str(args.seed),
        "--num-points", str(args.num_points),
        "--noise-px", str(args.noise_px),
        "--output-dir", args.out_v1
    ])

    # Run V2
    run_cmd([
        sys.executable, "scripts/run_experiments_v2.py",
        "--seed", str(args.seed),
        "--num-points", str(args.num_points),
        "--noise-px", str(args.noise_px),
        "--output-dir", args.out_v2
    ])

    # Load summaries
    v1_summary_path = os.path.join(args.out_v1, "summary.csv")
    v2_summary_path = os.path.join(args.out_v2, "summary.csv")

    v1_df = safe_read_csv(v1_summary_path)
    v2_df = safe_read_csv(v2_summary_path)

    rank1 = rank_methods(v1_df)
    rank2 = rank_methods(v2_df)
    wl1 = winloss_vs_baseline(v1_df, baseline="conf_unc")
    wl2 = winloss_vs_baseline(v2_df, baseline="conf_unc")
    ri1 = relative_improvement_vs_baseline(v1_df, baseline="conf_unc")
    ri2 = relative_improvement_vs_baseline(v2_df, baseline="conf_unc")

    findings = extract_key_findings(v1_df, v2_df)
    verdict = final_verdict(v2_df, ri2, rank2)

    # Save csv artifacts
    rank1.to_csv(os.path.join(args.out_final, "ranking_v1.csv"), index=False)
    rank2.to_csv(os.path.join(args.out_final, "ranking_v2.csv"), index=False)
    wl1.to_csv(os.path.join(args.out_final, "winloss_v1_vs_conf_unc.csv"), index=False)
    wl2.to_csv(os.path.join(args.out_final, "winloss_v2_vs_conf_unc.csv"), index=False)
    ri1.to_csv(os.path.join(args.out_final, "relative_impr_v1_vs_conf_unc.csv"), index=False)
    ri2.to_csv(os.path.join(args.out_final, "relative_impr_v2_vs_conf_unc.csv"), index=False)
    findings.to_csv(os.path.join(args.out_final, "ig_cross_version_findings.csv"), index=False)

    # Final report
    report_path = os.path.join(args.out_final, "final_combined_report.md")
    write_report(report_path, args, v1_df, v2_df, rank1, rank2, wl1, wl2, ri1, ri2, findings, verdict)

    print("\n[DONE] Final combined artifacts:")
    print(" -", report_path)
    print(" -", os.path.join(args.out_final, "ranking_v1.csv"))
    print(" -", os.path.join(args.out_final, "ranking_v2.csv"))
    print(" -", os.path.join(args.out_final, "ig_cross_version_findings.csv"))

if __name__ == "__main__":
    main()
