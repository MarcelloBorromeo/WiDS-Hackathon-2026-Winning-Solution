# Results

## Headline

| | Score | Notes |
|---|---|---|
| **Private leaderboard** | **0.97272** | **2nd place**, team `btt_mapping` |
| Best public leaderboard | 0.97458 | `V28_seed_avg` (shipped) |
| V28 out-of-fold | 0.9734 | C-index 0.9448 · WBS 0.0143 |

The metric is `Hybrid = 0.3·C-index + 0.7·(1−WBS)`, with `WBS = 0.3·BS@24h + 0.4·BS@48h + 0.3·BS@72h`.

---

## Full version history

The progression below shows how the public-LB score moved with each idea. Note how often a higher OOF
came with a *lower* LB — the central challenge described in
[`../research/METHODOLOGY.md`](../research/METHODOLOGY.md).

| Version | OOF | Public LB | Notes |
|---|---|---|---|
| V1 | 0.9579 | 0.94728 | Single RSF baseline |
| V2 | 0.9733 | 0.97031 | 5-model ensemble, isotonic all horizons |
| V3 | 0.9726 | 0.96807 | Growth features — overfit |
| V5 | 0.9730 | 0.95970 | Specialist blending — overfit |
| V7 | 0.9746 | 0.96856 | Best OOF ever, worst LB — growth features don't transfer |
| V8 | 0.9717 | 0.97090 | Hybrid objective + Platt@12h/24h + isotonic@48h/72h |
| V9 | 0.9719 | 0.97026 | Joint Platt — worse than independent |
| V10 | 0.9717 | 0.97048 | 42→31 features |
| V12 | 0.9720 | 0.96956 | +tti_log — adding features hurt |
| V13 | 0.9673 | 0.95962 | 4× LGBM binary classifiers — collapsed |
| V15 | 0.9722 | 0.97198 | **Two-stage: Ridge timing on 69 close fires, beta=0.40** |
| V16 | 0.9736 | 0.97059 | Top-20 feature selection + beta=1.0 — overfit |
| V18 | 0.9733 | 0.97136 | seed=1024 — worse than seed=42 |
| V21 | 0.9729 | 0.97438 | **Coxnet replaces Ridge for timing — biggest jump** |
| V28 | 0.9734 | **0.97458** | **5-seed averaged close-fire OOF — shipped model** |
| V32 | 0.9736 | 0.97180 | l1=0.3 — OOF +0.0002 → LB −0.00278 |
| V35 | 0.9734 | 0.97217 | Bootstrap test predictions — subset bias |
| V37 | 0.9740 | 0.96981 | AFT + feature-selection leakage — catastrophic |
| V38 | 0.9737 | 0.97445 | +4 engineered features — a wash vs. V28 |
| C_k3 | 0.9743 | 0.96312 | k-NN (k=3) timing — worst regression (−12.7× amplification) |
| L_AFT | 0.9734 | 0.97293 | LogNormalAFT as 4th backbone — diversity didn't help |

---

## OOF-vs-LB amplification — the key diagnostic

On this dataset the cross-validation score and the leaderboard diverge in a *predictable* way, and that
divergence is itself the most useful signal:

- **Genuine bias fixes amplify 3–5×** on the LB (a real OOF gain of +0.0005 → ~+0.0002 LB, monotonically).
- **Variance reductions amplify ~0.4×** (small, safe gains — e.g. V28's 5-seed averaging).
- **Overfit "improvements" amplify negatively and catastrophically** — V32 (×−14), V37 (×−8),
  k-NN k=3 (×−12.7). Each looked like a clear win on OOF.

A model is healthy when its **OOF sits slightly below its LB** (V28: OOF 0.9734 < LB 0.97458). When OOF
*exceeds* LB it usually means the model memorized the n=69 close fires.

See [`../research/METHODOLOGY.md`](../research/METHODOLOGY.md) for the full "what works / what fails"
breakdown and the locked-hyperparameter rationale.
