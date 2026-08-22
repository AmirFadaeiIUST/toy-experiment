import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Math / SE3
# =========================
def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)

def so3_exp(w):
    th = np.linalg.norm(w)
    I = np.eye(3)
    if th < 1e-12:
        return I + skew(w)
    K = skew(w / th)
    return I + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)

def se3_exp(xi):
    v, w = xi[:3], xi[3:]
    R = so3_exp(w)
    th = np.linalg.norm(w)
    I = np.eye(3)
    if th < 1e-12:
        V = I + 0.5 * skew(w)
    else:
        K = skew(w / th)
        V = I + ((1 - np.cos(th)) / th) * K + ((th - np.sin(th)) / th) * (K @ K)
    t = V @ v
    return R, t

def compose(R1, t1, R2, t2):
    return R1 @ R2, R1 @ t2 + t1

def apply_inc(R, t, dxi):
    dR, dt = se3_exp(dxi)
    return compose(dR, dt, R, t)

def rot_err_deg(R_est, R_gt):
    dR = R_est @ R_gt.T
    c = np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

# =========================
# Camera / geometry
# =========================
def make_K():
    return np.array([[500.0, 0.0, 320.0],
                     [0.0, 500.0, 240.0],
                     [0.0, 0.0, 1.0]], dtype=float)

def transform(R, t, X):
    return (R @ X.T).T + t[None, :]

def project(K, Xc):
    z = np.where(np.abs(Xc[:, 2:3]) < 1e-9, 1e-9, Xc[:, 2:3])
    xn = Xc[:, :2] / z
    uv = np.empty((Xc.shape[0], 2), dtype=float)
    uv[:, 0] = K[0, 0] * xn[:, 0] + K[0, 2]
    uv[:, 1] = K[1, 1] * xn[:, 1] + K[1, 2]
    return uv

def jacobian_2x6_analytic(K, Xc):
    fx, fy = K[0, 0], K[1, 1]
    x, y, z = Xc
    z2 = max(z * z, 1e-12)

    Jp = np.array([[fx / z, 0.0, -fx * x / z2],
                   [0.0, fy / z, -fy * y / z2]], dtype=float)

    dXdv = np.eye(3)
    dXdw = np.array([[0.0, z, -y],
                     [-z, 0.0, x],
                     [y, -x, 0.0]], dtype=float)

    return Jp @ np.hstack([dXdv, dXdw])

def residual_and_jac(K, R, t, X, uv_obs):
    Xc = transform(R, t, X)
    uv_hat = project(K, Xc)
    r = (uv_hat - uv_obs).reshape(-1)
    n = X.shape[0]
    J = np.zeros((2 * n, 6), dtype=float)
    for i in range(n):
        J[2 * i:2 * i + 2, :] = jacobian_2x6_analytic(K, Xc[i])
    return r, J, Xc

def residual_single(K, R, t, X, uv_obs, i):
    Xc = transform(R, t, X[i:i+1])[0]
    uv_hat = project(K, Xc[None, :])[0]
    return uv_hat - uv_obs[i]

def jacobian_2x6_numeric(K, R, t, X, uv_obs, i, eps=1e-6):
    J = np.zeros((2, 6), dtype=float)
    for k in range(6):
        d = np.zeros(6, dtype=float)
        d[k] = eps
        Rp, tp = apply_inc(R, t, d)
        Rm, tm = apply_inc(R, t, -d)
        rp = residual_single(K, Rp, tp, X, uv_obs, i)
        rm = residual_single(K, Rm, tm, X, uv_obs, i)
        J[:, k] = (rp - rm) / (2 * eps)
    return J

# =========================
# Weighting / information
# =========================
def base_weights(C, sigma, mode, eps=1e-9):
    if mode == "conf":
        w = C.copy()
    elif mode == "unc":
        w = 1.0 / (sigma**2 + eps)
    else:
        w = C / (sigma**2 + eps)  # conf_unc and ig variants
    return np.clip(w, eps, 1e9)

def compute_Hi_list(J, w):
    out = []
    for i in range(len(w)):
        Ji = J[2 * i:2 * i + 2, :]
        out.append(Ji.T @ (w[i] * np.eye(2)) @ Ji)
    return out

def safe_logdet(A, reg=1e-9):
    Ar = A + reg * np.eye(A.shape[0])
    sign, ld = np.linalg.slogdet(Ar)
    if sign <= 0:
        eig = np.linalg.eigvalsh(Ar)
        shift = max(0.0, -float(np.min(eig)) + reg)
        sign, ld = np.linalg.slogdet(Ar + shift * np.eye(A.shape[0]))
    return float(ld)

def information_gain_per_point(Hi, lam=1e-6):
    H = np.sum(Hi, axis=0)
    ld_full = safe_logdet(H + lam * np.eye(6))
    ig = np.zeros(len(Hi), dtype=float)
    for i, h in enumerate(Hi):
        ld_wo = safe_logdet(H - h + lam * np.eye(6))
        ig[i] = max(0.0, ld_full - ld_wo)
    return ig, H

def ig_gate(ig, mode, eps=1e-9, T=0.5):
    ig = np.nan_to_num(ig, nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "conf_unc_ig_ratio":
        g = ig / (np.mean(ig) + eps)
    elif mode == "conf_unc_ig_softmax":
        z = ig / max(T, eps)
        z = z - np.max(z)
        p = np.exp(z)
        p = p / (np.sum(p) + eps)
        g = p * len(ig)
    else:
        g = np.ones_like(ig)
    return np.clip(g, eps, 1e6)

def info_metrics(H):
    Hs = 0.5 * (H + H.T)
    e = np.linalg.eigvalsh(Hs)
    lam_min = float(np.min(e))
    lam_max = float(np.max(e))
    kappa = float(lam_max / max(lam_min, 1e-12))
    return lam_min, kappa

# =========================
# Robust GN (Huber IRLS)
# =========================
def huber_weight_from_norm(norm_r, delta=2.0):
    # weight for each 2D correspondence norm
    # sqrt-weight used in LS:
    # w_h = 1 (small), delta/|r| (large)
    return np.where(norm_r <= delta, 1.0, delta / np.maximum(norm_r, 1e-12))

def solve_gn_irls_huber(K, X, uv_obs, w_corr, R0, t0, max_iter=25, lam=1e-6, huber_delta=2.0):
    R, t = R0.copy(), t0.copy()
    prev_cost = np.inf
    fail = False

    for _ in range(max_iter):
        r, J, _ = residual_and_jac(K, R, t, X, uv_obs)
        n = X.shape[0]

        r2 = r.reshape(n, 2)
        nr = np.linalg.norm(r2, axis=1)
        wh = huber_weight_from_norm(nr, delta=huber_delta)

        w_total = np.clip(w_corr * wh, 1e-12, 1e12)
        W = np.repeat(w_total, 2)
        sw = np.sqrt(W)

        rw = r * sw
        Jw = J * sw[:, None]

        H = Jw.T @ Jw + lam * np.eye(6)
        g = Jw.T @ rw

        try:
            dxi = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            fail = True
            break

        Rn, tn = apply_inc(R, t, dxi)
        rn, _, _ = residual_and_jac(K, Rn, tn, X, uv_obs)

        rn2 = rn.reshape(n, 2)
        nrn = np.linalg.norm(rn2, axis=1)
        whn = huber_weight_from_norm(nrn, delta=huber_delta)
        w_total_n = np.clip(w_corr * whn, 1e-12, 1e12)
        sn = np.sqrt(np.repeat(w_total_n, 2))

        cost = 0.5 * float(np.sum((rn * sn) ** 2))

        if cost > prev_cost + 1e-8:
            dxi *= 0.5
            Rn, tn = apply_inc(R, t, dxi)
            rn, _, _ = residual_and_jac(K, Rn, tn, X, uv_obs)
            rn2 = rn.reshape(n, 2)
            nrn = np.linalg.norm(rn2, axis=1)
            whn = huber_weight_from_norm(nrn, delta=huber_delta)
            w_total_n = np.clip(w_corr * whn, 1e-12, 1e12)
            sn = np.sqrt(np.repeat(w_total_n, 2))
            cost = 0.5 * float(np.sum((rn * sn) ** 2))

        R, t = Rn, tn
        prev_cost = cost

        if np.linalg.norm(dxi) < 1e-8:
            break
        if (not np.isfinite(cost)) or cost > 1e12:
            fail = True
            break

    return R, t, float(prev_cost), fail

# =========================
# Scenarios
# =========================
SCENARIOS = [
    "normal_3d",
    "planar_scene",
    "forward_motion",
    "pure_rotation",
    "low_parallax",
    "redundant_correspondences",
    "outlier_contamination",
]
METHODS = ["conf", "unc", "conf_unc", "conf_unc_ig_ratio", "conf_unc_ig_softmax"]

def sample_normal(rng, n):
    X = np.empty((n, 3), dtype=float)
    X[:, 0] = rng.uniform(-2, 2, n)
    X[:, 1] = rng.uniform(-1.5, 1.5, n)
    X[:, 2] = rng.uniform(4, 12, n)
    return X

def sample_planar(rng, n):
    X = np.empty((n, 3), dtype=float)
    X[:, 0] = rng.uniform(-3, 3, n)
    X[:, 1] = rng.uniform(-2, 2, n)
    X[:, 2] = 8.0 + rng.normal(0, 0.05, n)
    return X

def pose_for(name):
    if name == "normal_3d":
        w, t = np.array([0.02, -0.01, 0.015]), np.array([0.25, -0.05, 0.1])
    elif name == "planar_scene":
        w, t = np.array([0.01, 0.0, 0.02]), np.array([0.2, 0.0, 0.05])
    elif name == "forward_motion":
        w, t = np.array([0.01, 0.0, 0.0]), np.array([0.02, 0.0, 0.5])
    elif name == "pure_rotation":
        w, t = np.array([0.08, -0.03, 0.05]), np.array([0.0, 0.0, 0.0])
    elif name == "low_parallax":
        w, t = np.array([0.005, 0.0, 0.005]), np.array([0.005, 0.0, 0.04])
    elif name == "redundant_correspondences":
        w, t = np.array([0.01, 0.0, 0.0]), np.array([0.25, 0.0, 0.08])
    elif name == "outlier_contamination":
        w, t = np.array([0.02, -0.01, 0.02]), np.array([0.2, -0.03, 0.12])
    else:
        raise ValueError(name)
    return so3_exp(w), t

def gen_scene(name, seed, n, noise_px):
    rng = np.random.default_rng(seed)
    K = make_K()

    X = sample_planar(rng, n) if name == "planar_scene" else sample_normal(rng, n)

    if name == "redundant_correspondences":
        base = sample_normal(rng, max(10, n // 6))
        reps = int(np.ceil(n / len(base)))
        X = np.vstack([base + rng.normal(0, 0.01, base.shape) for _ in range(reps)])[:n]

    if name == "low_parallax":
        X[:, 2] = rng.uniform(15, 30, n)

    R_gt, t_gt = pose_for(name)
    X2 = transform(R_gt, t_gt, X)
    uv2 = project(K, X2)
    uv_obs = uv2 + rng.normal(0, noise_px, uv2.shape)

    C = np.clip(rng.normal(0.85, 0.08, n), 0.05, 1.0)
    sigma = np.clip(rng.lognormal(np.log(max(noise_px, 0.3)), 0.25, n), 0.05, 10.0)

    if name == "outlier_contamination":
        m = max(1, n // 8)
        idx = rng.choice(n, size=m, replace=False)
        uv_obs[idx] += rng.normal(0, 25.0, (m, 2))
        C[idx] = np.clip(C[idx] * 0.5, 0.01, 1.0)
        sigma[idx] = np.clip(sigma[idx] * 2.0, 0.05, 20.0)

    return {
        "name": name,
        "K": K,
        "X": X,
        "uv_obs": uv_obs,
        "R_gt": R_gt,
        "t_gt": t_gt,
        "C": C,
        "sigma": sigma,
    }

# =========================
# Diagnostics and evaluation
# =========================
def jacobian_check(scene, num_samples=10, seed=0):
    rng = np.random.default_rng(seed)
    K, X, uv = scene["K"], scene["X"], scene["uv_obs"]
    R = so3_exp(np.array([0.01, -0.005, 0.008]))
    t = np.array([0.03, -0.01, 0.02])

    idxs = rng.choice(len(X), size=min(num_samples, len(X)), replace=False)
    rows = []
    for i in idxs:
        _, _, Xc_all = residual_and_jac(K, R, t, X, uv)
        Ja = jacobian_2x6_analytic(K, Xc_all[i])
        Jn = jacobian_2x6_numeric(K, R, t, X, uv, i, eps=1e-6)
        err = np.linalg.norm(Ja - Jn) / (np.linalg.norm(Jn) + 1e-12)
        rows.append({"point_id": int(i), "rel_jac_error": float(err)})
    return pd.DataFrame(rows)

def run_method(scene, method, seed, use_huber=True, huber_delta=2.0):
    rng = np.random.default_rng(seed + 1000)
    R0 = so3_exp(np.array([0.02, -0.01, 0.015]) + rng.normal(0, 0.01, 3))
    t0 = np.array([0.05, -0.02, 0.03]) + rng.normal(0, 0.02, 3)

    _, J0, _ = residual_and_jac(scene["K"], R0, t0, scene["X"], scene["uv_obs"])

    base_mode = method if method in ["conf", "unc", "conf_unc"] else "conf_unc"
    wb = base_weights(scene["C"], scene["sigma"], base_mode)

    Hi = compute_Hi_list(J0, wb)
    ig, H = information_gain_per_point(Hi)
    lam_min, kappa = info_metrics(H)
    trace_hi = np.array([np.trace(h) for h in Hi])

    if "ig_" in method:
        g = ig_gate(ig, method)
        w = np.clip(wb * g, 1e-9, 1e9)
    else:
        w = wb

    if use_huber:
        R_est, t_est, final_cost, fail = solve_gn_irls_huber(
            scene["K"], scene["X"], scene["uv_obs"], w, R0, t0, max_iter=25, lam=1e-6, huber_delta=huber_delta
        )
    else:
        R_est, t_est, final_cost, fail = solve_gn_irls_huber(
            scene["K"], scene["X"], scene["uv_obs"], w, R0, t0, max_iter=25, lam=1e-6, huber_delta=1e9
        )

    t_err = float(np.linalg.norm(t_est - scene["t_gt"]))
    r_err = rot_err_deg(R_est, scene["R_gt"])

    degenerate = 1 if (lam_min < 1e-3 or kappa > 5e3) else 0
    if (not np.isfinite(kappa)) or (kappa > 1e10) or (not np.isfinite(final_cost)):
        fail = True

    return {
        "scenario": scene["name"],
        "method": method,
        "translation_error": t_err,
        "rotation_error_deg": r_err,
        "final_residual": final_cost,
        "lambda_min_H": lam_min,
        "kappa_H": kappa,
        "failure": int(fail),
        "degenerate_flag": int(degenerate),
    }, ig, scene["C"], scene["sigma"], trace_hi

def ablation_equal_conf_sigma(scene):
    R0 = so3_exp(np.array([0.01, 0.0, 0.01]))
    t0 = np.array([0.02, 0.0, 0.02])
    _, J, _ = residual_and_jac(scene["K"], R0, t0, scene["X"], scene["uv_obs"])
    wb = base_weights(scene["C"], scene["sigma"], "conf_unc")
    Hi = compute_Hi_list(J, wb)
    ig, _ = information_gain_per_point(Hi)
    tr = np.array([np.trace(h) for h in Hi])

    C, sigma = scene["C"], scene["sigma"]
    best = None
    n = len(C)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(C[i] - C[j]) < 0.03 and abs(sigma[i] - sigma[j]) < 0.1:
                ratio = (ig[i] + 1e-9) / (ig[j] + 1e-9)
                ratio = max(ratio, 1.0 / ratio)
                if best is None or ratio > best[0]:
                    best = (ratio, i, j)

    if best is None:
        i, j = int(np.argmax(ig)), int(np.argmin(ig))
        ratio = (ig[i] + 1e-9) / (ig[j] + 1e-9)
    else:
        ratio, i, j = best

    df = pd.DataFrame([
        {"id": i, "conf": C[i], "sigma": sigma[i], "trace_Hi": tr[i], "IG": ig[i]},
        {"id": j, "conf": C[j], "sigma": sigma[j], "trace_Hi": tr[j], "IG": ig[j]},
    ])
    df["IG_ratio_maxmin"] = max(ratio, 1.0 / max(ratio, 1e-12))
    return df

def make_plots(summary_df, point_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for metric in ["translation_error", "rotation_error_deg", "lambda_min_H", "kappa_H", "failure"]:
        plt.figure(figsize=(11, 4))
        piv = summary_df.pivot(index="scenario", columns="method", values=metric)
        piv.plot(kind="bar", ax=plt.gca())
        plt.title(metric)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{metric}.png"), dpi=150)
        plt.close()

    # scatter point-level (for one representative scenario/method: pure_rotation + conf_unc)
    sub = point_df[(point_df["scenario"] == "pure_rotation") & (point_df["method"] == "conf_unc")].copy()
    if len(sub) > 0:
        plt.figure(figsize=(6, 5))
        plt.scatter(sub["conf"], sub["ig"], s=12, alpha=0.7)
        plt.xlabel("confidence")
        plt.ylabel("IG")
        plt.title("pure_rotation: confidence vs IG")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "scatter_conf_vs_ig_pure_rotation.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.scatter(sub["trace_hi"], sub["ig"], s=12, alpha=0.7)
        plt.xlabel("trace(H_i)")
        plt.ylabel("IG")
        plt.title("pure_rotation: trace(H_i) vs IG")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "scatter_traceHi_vs_ig_pure_rotation.png"), dpi=150)
        plt.close()

def winloss_vs_baseline(summary_df, baseline="conf_unc"):
    rows = []
    for method in summary_df["method"].unique():
        if method == baseline:
            continue
        wins_t = 0
        wins_r = 0
        for sc in summary_df["scenario"].unique():
            a = summary_df[(summary_df["scenario"] == sc) & (summary_df["method"] == method)].iloc[0]
            b = summary_df[(summary_df["scenario"] == sc) & (summary_df["method"] == baseline)].iloc[0]
            if a["translation_error"] < b["translation_error"]:
                wins_t += 1
            if a["rotation_error_deg"] < b["rotation_error_deg"]:
                wins_r += 1
        rows.append({
            "method": method,
            "wins_translation_over_conf_unc": wins_t,
            "wins_rotation_over_conf_unc": wins_r,
            "num_scenarios": summary_df["scenario"].nunique()
        })
    return pd.DataFrame(rows)

def build_method_ranking(summary_df):
    agg = summary_df.groupby("method", as_index=False).agg({
        "translation_error": "mean",
        "rotation_error_deg": "mean",
        "final_residual": "mean",
        "lambda_min_H": "mean",
        "kappa_H": "mean",
        "failure": "mean",
    })

    # lower better: trans, rot, residual, kappa, failure
    # higher better: lambda_min
    # normalized ranking score
    def norm_col(s, higher_better=False):
        a = s.values.astype(float)
        mn, mx = np.min(a), np.max(a)
        if abs(mx - mn) < 1e-12:
            z = np.zeros_like(a)
        else:
            z = (a - mn) / (mx - mn)
        return 1.0 - z if higher_better else z

    score = (
        0.30 * norm_col(agg["translation_error"], higher_better=False) +
        0.25 * norm_col(agg["rotation_error_deg"], higher_better=False) +
        0.15 * norm_col(agg["final_residual"], higher_better=False) +
        0.10 * norm_col(agg["kappa_H"], higher_better=False) +
        0.10 * norm_col(agg["failure"], higher_better=False) +
        0.10 * norm_col(agg["lambda_min_H"], higher_better=True)
    )
    agg["composite_score_lower_is_better"] = score
    agg = agg.sort_values("composite_score_lower_is_better", ascending=True).reset_index(drop=True)
    agg["rank"] = np.arange(1, len(agg) + 1)
    return agg

def degeneracy_detection_table(summary_df):
    # Here degenerate_flag from info matrix; failure from optimizer
    # We report for each method precision-like and recall-like proxy
    rows = []
    for m in summary_df["method"].unique():
        d = summary_df[summary_df["method"] == m]
        tp = int(np.sum((d["degenerate_flag"] == 1) & (d["failure"] == 1)))
        fp = int(np.sum((d["degenerate_flag"] == 1) & (d["failure"] == 0)))
        fn = int(np.sum((d["degenerate_flag"] == 0) & (d["failure"] == 1)))
        tn = int(np.sum((d["degenerate_flag"] == 0) & (d["failure"] == 0)))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        rows.append({
            "method": m,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision_deg_to_fail": precision,
            "recall_deg_to_fail": recall
        })
    return pd.DataFrame(rows)

def relative_improvement_vs_baseline(summary_df, baseline="conf_unc"):
    rows = []
    scenarios = summary_df["scenario"].unique()
    methods = summary_df["method"].unique()

    for m in methods:
        if m == baseline:
            continue
        imp_t, imp_r = [], []
        for sc in scenarios:
            a = summary_df[(summary_df["scenario"] == sc) & (summary_df["method"] == m)].iloc[0]
            b = summary_df[(summary_df["scenario"] == sc) & (summary_df["method"] == baseline)].iloc[0]
            imp_t.append((b["translation_error"] - a["translation_error"]) / max(b["translation_error"], 1e-12))
            imp_r.append((b["rotation_error_deg"] - a["rotation_error_deg"]) / max(b["rotation_error_deg"], 1e-12))
        rows.append({
            "method": m,
            "mean_rel_impr_translation_vs_conf_unc": float(np.mean(imp_t)),
            "mean_rel_impr_rotation_vs_conf_unc": float(np.mean(imp_r)),
        })
    return pd.DataFrame(rows)

def write_evaluation_report(out_path, summary_df, ranking_df, winloss_df, relimp_df, deg_df, jac_df, ab_df):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Evaluation Report (V2)\n\n")

        f.write("## 1) Summary by scenario/method\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## 2) Method ranking (composite)\n\n")
        f.write(ranking_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## 3) Win/Loss vs conf_unc baseline\n\n")
        f.write(winloss_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## 4) Relative improvement vs conf_unc\n\n")
        f.write(relimp_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## 5) Degeneracy detection proxy\n\n")
        f.write(deg_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## 6) Jacobian analytical vs numerical check\n\n")
        f.write(jac_df.to_markdown(index=False))
        f.write("\n\n")
        f.write(f"- Mean relative Jacobian error: **{jac_df['rel_jac_error'].mean():.4e}**\n")
        f.write(f"- Max relative Jacobian error: **{jac_df['rel_jac_error'].max():.4e}**\n\n")

        f.write("## 7) Key ablation: C≈, sigma≈ but IG differs\n\n")
        f.write(ab_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## 8) Interpretation guide\n\n")
        f.write("- اگر IG-based روش‌ها در `kappa` کمتر و `lambda_min` بیشتر باشند، conditioning بهتر شده.\n")
        f.write("- اگر همزمان خطای pose هم بهتر شود، utility هندسی به بهبود عملی تبدیل شده.\n")
        f.write("- در pure_rotation / forward_motion / low_parallax بهبود مهم‌تر از normal_3d است.\n")
        f.write("- اگر فقط residual بهتر شد ولی pose بهتر نشد، احتمال overfitting به noise/outlier وجود دارد.\n")

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-points", type=int, default=140)
    ap.add_argument("--noise-px", type=float, default=1.0)
    ap.add_argument("--output-dir", type=str, default="results_v2")
    ap.add_argument("--huber-delta", type=float, default=2.0)
    ap.add_argument("--no-huber", action="store_true", help="Disable Huber robust weighting")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)

    use_huber = not args.no_huber
    all_rows = []
    point_rows = []

    print("[INFO] Running scenarios...")
    for sidx, sname in enumerate(SCENARIOS):
        scene = gen_scene(sname, args.seed + 10 * sidx, args.num_points, args.noise_px)

        for midx, method in enumerate(METHODS):
            row, ig, C, sigma, tr = run_method(
                scene, method, args.seed + 100 * sidx + midx, use_huber=use_huber, huber_delta=args.huber_delta
            )
            all_rows.append(row)

            # store point-level diagnostics
            for i in range(len(ig)):
                point_rows.append({
                    "scenario": sname,
                    "method": method,
                    "point_id": i,
                    "conf": float(C[i]),
                    "sigma": float(sigma[i]),
                    "trace_hi": float(tr[i]),
                    "ig": float(ig[i]),
                })

    summary_df = pd.DataFrame(all_rows)
    point_df = pd.DataFrame(point_rows)

    # core outputs
    summary_csv = os.path.join(args.output_dir, "summary.csv")
    summary_md = os.path.join(args.output_dir, "summary.md")
    summary_df.to_csv(summary_csv, index=False)

    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Summary Results\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n## Mean by method\n\n")
        f.write(summary_df.groupby("method", as_index=False).mean(numeric_only=True).to_markdown(index=False))

    point_df.to_csv(os.path.join(args.output_dir, "point_diagnostics.csv"), index=False)

    # Jacobian check (representative scenario)
    scene_j = gen_scene("normal_3d", args.seed + 777, args.num_points, args.noise_px)
    jac_df = jacobian_check(scene_j, num_samples=12, seed=args.seed + 55)
    jac_df.to_csv(os.path.join(args.output_dir, "jacobian_check.csv"), index=False)

    # key ablation
    scene_ab = gen_scene("pure_rotation", args.seed + 999, args.num_points, args.noise_px)
    ab_df = ablation_equal_conf_sigma(scene_ab)
    ab_df.to_csv(os.path.join(args.output_dir, "ablation_equal_conf_sigma.csv"), index=False)
    with open(os.path.join(args.output_dir, "ablation_equal_conf_sigma.md"), "w", encoding="utf-8") as f:
        f.write("# Ablation: C≈ and sigma≈ but IG differs\n\n")
        f.write(ab_df.to_markdown(index=False))

    # evaluation tables
    ranking_df = build_method_ranking(summary_df)
    winloss_df = winloss_vs_baseline(summary_df, baseline="conf_unc")
    relimp_df = relative_improvement_vs_baseline(summary_df, baseline="conf_unc")
    deg_df = degeneracy_detection_table(summary_df)

    ranking_df.to_csv(os.path.join(args.output_dir, "method_ranking.csv"), index=False)
    winloss_df.to_csv(os.path.join(args.output_dir, "winloss_vs_conf_unc.csv"), index=False)
    relimp_df.to_csv(os.path.join(args.output_dir, "relative_improvement_vs_conf_unc.csv"), index=False)
    deg_df.to_csv(os.path.join(args.output_dir, "degeneracy_detection.csv"), index=False)

    # plots
    make_plots(summary_df, point_df, os.path.join(args.output_dir, "plots"))

    # evaluation report
    report_path = os.path.join(args.output_dir, "evaluation_report.md")
    write_evaluation_report(report_path, summary_df, ranking_df, winloss_df, relimp_df, deg_df, jac_df, ab_df)

    print("[DONE] Outputs:")
    print(" -", summary_csv)
    print(" -", summary_md)
    print(" -", report_path)
    print(" -", os.path.join(args.output_dir, "plots"))
    print(" -", os.path.join(args.output_dir, "jacobian_check.csv"))
    print(" -", os.path.join(args.output_dir, "ablation_equal_conf_sigma.csv"))

if __name__ == "__main__":
    main()
