# Methodology & Engineering Log

This document captures the modeling architecture, the design decisions and their rationale, and — most
importantly — the discipline used to separate genuine improvements from overfitting on a very small
dataset. For the headline summary see the [top-level README](../README.md); for the full chronological
scoreboard see [`../docs/RESULTS.md`](../docs/RESULTS.md).

---

## The competition in one paragraph

Predict per-structure wildfire survival probability at 12h / 24h / 48h / 72h. Scored by
`Hybrid = 0.3·C-index + 0.7·(1−WBS)`, where `WBS = 0.3·BS@24h + 0.4·BS@48h + 0.3·BS@72h`. There are 221
training fires and 95 test fires. Only the **69 close fires** (`dist < 5000 m`) carry timing signal, and
all 69 are `event=1`; all 152 far fires are `event=0`, with no exceptions. The score is effectively
decided by how well the model ranks and calibrates the **28 test close fires**, 89% of which are static
(zero closing speed). Timing — not ranking — is the hard problem.

---

## Architecture — canonical two-stage model

**Stage 1 · Backbone (all 221 fires).** RSF + GBM + Coxnet, trained with 10-fold StratifiedKFold to
produce out-of-fold predictions, then blended with Nelder-Mead optimizing the competition hybrid score.
The regularized linear **Coxnet dominates the blend** (~0.72–0.78 weight) because it is the best single
generalizer on this small, distribution-shifted data.

**Stage 2 · Close-fire timing (69 close fires).** A separate `CoxnetSurvivalAnalysis`
(`alpha=0.20, l1_ratio=0.5`) trained only on the close fires, with 7-fold KFold OOF repeated over 5 seeds
and averaged. This produces timing-calibrated probabilities for the close fires.

**Stage 3 · Blend + calibration.** For close fires, `mixed = (1−β)·backbone + β·close_cox` with a beta
grid search over `[0.00 → 0.60]`. Calibration is **Platt @12h/24h, isotonic @48h/72h**, fit on the mixed
OOF predictions. A `cummax` post-processing step enforces monotonicity.

### V28 (shipped) configuration

- 42 features (`BASE_FEATURES + ENGINEERED`, see `engineer()` in the notebook).
- Backbone blend weights: `coxnet=0.778, gbm=0.178, rsf=0.044`.
- Backbone Coxnet: `alpha=0.05, l1_ratio=0.5`.
- Close-fire Coxnet: `alpha=0.20, l1_ratio=0.5`, 5-seed average (`[42,100,200,300,400]`), 7-fold KFold.
- Distance threshold: 5000 m → 69 train / 28 test close fires.
- `beta=0.55`; Platt @12h/24h, isotonic @48h/72h.
- OOF: hybrid 0.9734, C-index 0.9448, WBS 0.0143.

---

## Design decisions & locked hyperparameters (with rationale)

These were locked after experiments confirmed they hurt when relaxed. Each is a small-data robustness
choice, not an arbitrary constant.

1. **Close-fire Coxnet `alpha=0.20, l1_ratio=0.5`.** Loosening to `l1_ratio=0.3` gained +0.0002 OOF but
   lost **−0.00278 LB**. The L1 component drops noise features; it is not safely tunable at n=69.
2. **`beta ≤ 0.60`, never higher.** A run at `beta=0.70` scored 0.96426 — worse than the V8 baseline.
   Over-weighting the n=69 timing model overrides the better-generalizing backbone.
3. **Distance threshold = 5000 m.** Tested 4000 / 4500 / 5000 / 5500 / 6000 m; 5000 m is optimal.
4. **Coxnet must anchor the backbone (weight ≥ 0.60).** Relaxing this to add a 4th backbone model cost
   −0.00165 LB.
5. **Platt @12h/24h, isotonic @48h/72h.** Isotonic at early horizons memorizes training timing →
   inflated OOF → worse LB. The probability cliff at 48h/72h is physically real; keep it sharp.
6. **Backbone seed = 42.** 5-seed averaging is used *only* for the close-fire Coxnet OOF (variance
   reduction for beta calibration), not for the backbone.
7. **No non-parametric models on the 69 close fires.** RSF / k-NN memorize n=69 catastrophically.
8. **Monotonicity is mandatory.** `P(12h) ≤ P(24h) ≤ P(48h) ≤ P(72h)` for every fire; assert zero
   violations before every submission.

---

## What works

- **Coxnet as the backbone anchor** — regularized linear, anti-overfit, best single generalizer.
- **Optimizing the hybrid objective directly** with Nelder-Mead — aligns the blend with the metric.
- **5-seed averaging of the close-fire OOF** — reduces split variance, yields better beta calibration.
- **10-fold StratifiedKFold for the backbone** — never fewer than 10 folds.
- **Independent Platt slope per horizon** — beats a shared/joint Platt fit.
- **`cummax` post-processing** — the correct, simple monotonicity interface.
- **Genuine bias fixes amplify on the LB** — an OOF gain of +0.0005 from a real fix showed ~3–5× on LB.

---

## What fails (and why)

| Approach | OOF | LB | Why it failed |
|---|---|---|---|
| RSF on the 69 close fires | 0.9738 | 0.96426 | Memorized n=69 |
| `l1_ratio=0.3` for close Coxnet | 0.9736 | 0.97180 | Denser model overfit n=69 |
| Bootstrap test predictions | 0.9734 | 0.97217 | Subset training biases predictions |
| AFT + k=12 feature selection | 0.9740 | 0.96981 | Feature-selection leakage on n=69 |
| k-NN timing (k=3) | 0.9743 | 0.96312 | Memorized 3 neighbors; **−12.7× amplification** |
| Growth / area features | high | −0.012 | Train/test distribution shift |
| Per-horizon LGBM classifiers | 0.9673 | 0.95962 | No information sharing across horizons |
| Isotonic at 12h/24h | 0.9733 | 0.97031 | Memorizes training timing |
| LogNormalAFT as 4th backbone | 0.9734 | 0.97293 | Structural diversity didn't help |

**Red flags for overfit:** a large WBS drop **and** an OOF spike at a single parameter value **and** a
C-index drop, appearing together. Several such changes looked like clear wins and were catastrophic on
the LB.

---

## The core lesson: the n=69 problem

The close-fire timing model has only 69 training examples, all `event=1`. **No cross-validation scheme
can reliably detect overfitting at this sample size when you change model complexity or regularization.**
The 5-seed OOF average is the best available tool — it reduces variance from random splits but cannot
detect *bias* from the wrong model complexity. So:

- The **leaderboard is the only unbiased judge.** A high OOF is a *red flag* when it comes from isotonic
  at early horizons, feature selection on n=69, loosened regularization, or non-parametric models.
- A well-calibrated model has OOF *slightly below* its LB. V28's OOF (0.9734) sits just under its LB
  (0.97458) — the correct, healthy relationship.
- **OOF→LB amplification** is the key diagnostic: genuine bias fixes amplify ~3–5×; variance reductions
  ~0.4×; overfit "improvements" amplify negatively (one change hit ×−14).

---

## Workflow

Experiments were run as standalone Python scripts first; notebooks were built only for confirmed
versions. Locked parameters were never changed without explicit, deliberate justification — because at
n=69 the cross-validation signal that would normally guide tuning is unreliable.
