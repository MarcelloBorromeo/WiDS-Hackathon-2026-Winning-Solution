# Submissions

A curated set of submission files that trace the leaderboard story. These are the actual prediction
CSVs produced by the corresponding model versions (full version history in
[`../docs/RESULTS.md`](../docs/RESULTS.md)).

| File | Version | Public LB | Milestone |
|---|---|---|---|
| `submission_v08_hybrid.csv` | V8 | 0.97090 | Hybrid objective + Platt@12h/24h + isotonic@48h/72h |
| `submission_v15_two_stage.csv` | V15 | 0.97198 | Two-stage architecture introduced (Ridge close-fire timing) |
| `submission_v21_close_cox.csv` | V21 | 0.97438 | **Coxnet replaces Ridge for timing — the biggest single jump** |
| `submission_v28_seed_avg.csv` | V28 | **0.97458** | **Shipped model** — 5-seed averaged close-fire OOF |
| `submission_v38_new_features.csv` | V38 | 0.97445 | +4 engineered features — a wash vs. V28 |

Each file maps `event_id` to survival probabilities at the 12h / 24h / 48h / 72h horizons, monotonic
across horizons by construction.
