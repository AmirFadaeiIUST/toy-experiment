import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- math utils ----------------
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

# ---------------- camera ----------------
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

def jacobian_2x6(K, Xc):
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
        J[2 * i:2 * i + 2, :] = jacobian_2x6(K, Xc[i])
    return r, J

# ---------------- weighting/info ----------------
def base_weights(C, sigma, mode, eps=1e-9):
    if mode == "conf":
        w = C.copy()
    elif mode == "unc":
        w = 1.0 / (sigma**2 + eps)
    else:
        w = C / (sigma**2 + eps)  # conf_unc + ig variants
    return np.clip(w, eps, 1e9)

def Hi_list(J, w):
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
        shift = max(0.0, -float(eig.min()) + reg)
        sign, ld = np.linalg.slogdet(Ar + shift * np.eye(A.shape[0]))
    return float(ld)

def info_gain_per_point(Hi, lam=1e-6):
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
        g = ig / (ig.mean() + eps)
    elif mode == "conf_unc_ig_softmax":
        z = ig / max(T, eps)
        z = z - np.max(z)
        p = np.exp(z)
        p = p / (np.sum(p) + eps)
        g = p * len(ig)  # keep average around 1
    else:
        g = np.ones_like(ig)
    return np.clip(g, eps, 1e6)

# ---------------- solver ----------------
def solve_gn(K, X, uv_obs, w, R0, t0, max_iter=20, lam=1e-6):
    R, t = R0.copy(), t0.copy()
    prev_cost = np.inf
    fail = False

    for _ in range(max_iter):
        r, J = residual_and_jac(K, R, t, X, uv_obs)
        W = np.repeat(w, 2)
        sw = np.sqrt(np.clip(W, 1e-12, 1e12))
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
        rn, _ = residual_and_jac(K, Rn, tn, X, uv_obs)
        cost = 0.5 * float(np.sum((rn * sw) ** 2))

        if cost > prev_cost + 1e-8:
            dxi *= 0.5
            Rn, tn = apply_inc(R, t, dxi)
            rn, _ = residual_and_jac(K, Rn, tn, X, uv_obs)
            cost = 0.5 * float(np.sum((rn * sw) ** 2))

        R, t = Rn, tn
        prev_cost = cost

        if np.linalg.norm(dxi) < 1e-8:
            break
        if (not np.isfinite(cost)) or cost > 1e12:
            fail = True
            break

    return R, t, float(prev_cost), fail

# ---------------- scenarios ----------------
def sample_normal(rng, n):
    X = np.empty((n, 3), dtype=float)
    X[:, 0] = rng.uniform(-2, 2, size=n)
    X[:, 1] = rng.uniform(-1.5, 1.5, size=n)
    X[:, 2] = rng.uniform(4, 12, size=n)
    return X

def sample_planar(rng, n):
    X = np.empty((n, 3), dtype=float)
    X[:, 0] = rng.uniform(-3, 3, size=n)
    X[:, 1] = rng.uniform(-2, 2, size=n)
    X[:, 2] = 8.0 + rng.normal(0, 0.05, size=n)
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
        X[:, 2] = rng.uniform(15, 30, size=n)

    R_gt, t_gt = pose_for(name)
    X2 = transform(R_gt, t_gt, X)
    uv2 = project(K, X2)
    uv_obs = uv2 + rng.normal(0, noise_px, size=uv2.shape)

    C = np.clip(rng.normal(0.85, 0.08, size=n), 0.05, 1.0)
    sigma = np.clip(rng.lognormal(np.log(max(noise_px, 0.3)), 0.25, size=n), 0.05, 10.0)

    if name == "outlier_contamination":
        m = max(1, n // 8)
        idx = rng.choice(n, size=m, replace=False)
        uv_obs[idx] += rng.normal(0, 25.0, size=(m, 2))
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

# ---------------- experiment ----------------
METHODS = ["conf", "unc", "conf_unc", "conf_unc_ig_ratio", "conf_unc_ig_softmax"]
SCENARIOS = [
    "normal_3d",
    "planar_scene",
    "forward_motion",
    "pure_rotation",
    "low_parallax",
    "redundant_correspondences",
    "outlier_contamination",
]

def info_metrics(H):
    e = np.linalg.eigvalsh(0.5 * (H + H.T))
    lam_min = float(np.min(e))
    lam_max = float(np.max(e))
    kappa = float(lam_max / max(lam_min, 1e-12))
    return lam_min, kappa

def run_method(scene, method, seed):
    rng = np.random.default_rng(seed + 1000)
    R0 = so3_exp(np.array([0.02, -0.01, 0.015]) + rng.normal(0, 0.01, 3))
    t0 = np.array([0.05, -0.02, 0.03]) + rng.normal(0, 0.02, 3)

    _, J0 = residual_and_jac(scene["K"], R0, t0, scene["X"], scene["uv_obs"])
    base_mode = method if method in ["conf", "unc", "conf_unc"] else "conf_unc"
    wb = base_weights(scene["C"], scene["sigma"], base_mode)

    Hi = Hi_list(J0, wb)
    ig, H = info_gain_per_point(Hi)
    lam_min, kappa = info_metrics(H)

    if "ig_" in method:
        g = ig_gate(ig, method)
        w = np.clip(wb * g, 1e-9, 1e9)
    else:
        w = wb

    R_est, t_est, final_cost, fail = solve_gn(scene["K"], scene["X"], scene["uv_obs"], w, R0, t0)

    t_err = float(np.linalg.norm(t_est - scene["t_gt"]))
    r_err = rot_err_deg(R_est, scene["R_gt"])

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
    }

def ablation_equal_conf_sigma(scene):
    R0 = so3_exp(np.array([0.01, 0.0, 0.01]))
    t0 = np.array([0.02, 0.0, 0.02])

    _, J = residual_and_jac(scene["K"], R0, t0, scene["X"], scene["uv_obs"])
    wb = base_weights(scene["C"], scene["sigma"], "conf_unc")
    Hi = Hi_list(J, wb)
    ig, _ = info_gain_per_point(Hi)
    tr = np.array([np.trace(h) for h in Hi])

    C = scene["C"]
    sigma = scene["sigma"]
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
    else:
        _, i, j = best

    return pd.DataFrame([
        {"id": i, "conf": C[i], "sigma": sigma[i], "trace_Hi": tr[i], "IG": ig[i]},
        {"id": j, "conf": C[j], "sigma": sigma[j], "trace_Hi": tr[j], "IG": ig[j]},
    ])

def plot_metrics(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    metrics = ["translation_error", "rotation_error_deg", "lambda_min_H", "kappa_H"]

    for m in metrics:
        plt.figure(figsize=(10, 4))
        piv = df.pivot(index="scenario", columns="method", values=m)
        piv.plot(kind="bar", ax=plt.gca())
        plt.title(m)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{m}.png"), dpi=140)
        plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-points", type=int, default=120)
    ap.add_argument("--noise-px", type=float, default=1.0)
    ap.add_argument("--output-dir", type=str, default="results")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []

    for sidx, sname in enumerate(SCENARIOS):
        print(f"[INFO] scenario: {sname}")
        scene = gen_scene(sname, args.seed + 10 * sidx, args.num_points, args.noise_px)
        for midx, method in enumerate(METHODS):
            row = run_method(scene, method, args.seed + 100 * sidx + midx)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)

    with open(os.path.join(args.output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Summary Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Mean by method\n\n")
        mean_df = df.groupby("method", as_index=False).mean(numeric_only=True)
        f.write(mean_df.to_markdown(index=False))

    scene_ab = gen_scene("pure_rotation", args.seed + 999, args.num_points, args.noise_px)
    ab = ablation_equal_conf_sigma(scene_ab)
    ab.to_csv(os.path.join(args.output_dir, "ablation_equal_conf_sigma.csv"), index=False)

    with open(os.path.join(args.output_dir, "ablation_equal_conf_sigma.md"), "w", encoding="utf-8") as f:
        f.write("# Ablation: C≈ and sigma≈ but IG different\n\n")
        f.write(ab.to_markdown(index=False))
        f.write("\n\nInterpretation: reliability مشابه لزوماً utility هندسی مشابه نمی‌دهد.\n")

    plot_metrics(df, os.path.join(args.output_dir, "plots"))
    print(f"[DONE] outputs in: {args.output_dir}")

if __name__ == "__main__":
    main()
