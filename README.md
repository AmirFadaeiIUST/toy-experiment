# toy-experiment
# Observability-Aware Correspondence Weighting (Toy Monocular VO)

این پروژه یک toy experiment برای monocular VO می‌سازد تا بررسی کند آیا
**Marginal Information Gain (IG)** می‌تواند بهتر از confidence/uncertainty weighting معمولی عمل کند یا نه.

## ایده

برای correspondence شماره `i`:

- `C_i`: confidence/reliability
- `sigma_i`: uncertainty
- `J_i = ∂r_i/∂ξ`, با `ξ ∈ R^6` (pose در SE(3))

Baseline weights:

- `w_i = C_i`
- `w_i = 1/(sigma_i^2 + eps)`
- `w_i = C_i/(sigma_i^2 + eps)`

Local information:

- `H_i = J_i^T w_i J_i`

Global information:

- `H = Σ_i H_i`

Marginal information gain:

- `IG_i = logdet(H + λI) - logdet(H - H_i + λI)`

روش IG-aware:

- `w_i^final = w_i^base * g(IG_i)`
- `g`:
  - ratio: `IG / mean(IG)`
  - softmax: `softmax(IG/T)` (scaled to mean≈1)

## سناریوها

1. normal_3d
2. planar_scene
3. forward_motion
4. pure_rotation
5. low_parallax
6. redundant_correspondences
7. outlier_contamination

## اجرا

```bash
python scripts/run_experiments.py --seed 42 --num-points 120 --noise-px 1.0 --output-dir results
```

## خروجی‌ها

- `results/summary.csv`
- `results/summary.md`
- `results/ablation_equal_conf_sigma.csv`
- `results/ablation_equal_conf_sigma.md`
- `results/plots/*.png`

## نکات

- toy setup است؛ claim نهایی real-world نیست.
- monocular scale ambiguity ذاتاً باقی می‌ماند.
- conditioning metrics (`lambda_min`, `kappa`) برای degenerate motions مهم‌اند.
