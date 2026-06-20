# Entry Points

This solution is implemented as a single end-to-end Jupyter notebook. There are no separate prepare/train/predict scripts — all three stages run sequentially within the notebook.

---

## Environment Setup

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

---

## Full Pipeline (Train + Predict)

Open and run all cells in:

```
model/wids_v28_best.ipynb
```

The notebook locates `SETTINGS.json` by walking up from its directory, so it runs correctly whether
Jupyter is launched from the repo root or from `model/`. All paths resolve relative to the repo root.

This single notebook performs the complete pipeline:

| Step | Description | Output |
|------|-------------|--------|
| Data loading | Reads `train.csv` and `test.csv` from `RAW_DATA_DIR` | — |
| Feature engineering | Engineers 42 features via `engineer()` | — |
| Preprocessing | Median imputation + standard scaling | `pre` (fitted Pipeline) |
| Stage 1 — Backbone OOF | 10-fold StratifiedKFold CV across RSF, GBM, Coxnet | `oof` dict |
| Stage 1 — Blend | Nelder-Mead optimization of blend weights | `weights` dict |
| Stage 2 — Close-fire OOF | 5-seed × 7-fold Coxnet on 69 close fires | `close_cox_oof` |
| Stage 3 — Beta + calibration | Beta grid search, Platt (12h/24h), isotonic (48h/72h) | `best_cals`, `best_beta` |
| Final training | All models retrained on full 221-fire dataset | `final_models`, `m_cox_close` |
| Prediction | Test predictions generated and monotonicity-checked | `submissions/submission_v28_seed_avg.csv` |
| Serialization | Full artifact bundle saved to disk | `model/model_artifacts_v28.pkl` |

**Expected runtime:** ~8–10 minutes on Apple M4 (10 cores, 16 GB RAM).

---

## Prediction Only (No Retraining)

To generate predictions from the saved model without retraining:

1. Ensure `model/model_artifacts_v28.pkl` is present (path set by `MODEL_PATH` in `SETTINGS.json`)
2. Load the artifact bundle and run the inference cells in the notebook (cells under "Generating test predictions" onward)

```python
import pickle
with open('model/model_artifacts_v28.pkl', 'rb') as f:
    art = pickle.load(f)
```

The bundle is a dict with keys:
`version, feat_cols, preprocessor, weights, best_beta, best_cals, final_models, close_model,`
`close_train, close_test, close_seeds, oof_hybrid, oof_c, oof_wbs, lb_score`.

**Expected runtime:** ~5 seconds.

---

## Notes

- `SETTINGS.json` must be updated if data or output paths differ from the defaults.
- Do not run cells out of order — the pipeline is stateful and each stage depends on the previous.
