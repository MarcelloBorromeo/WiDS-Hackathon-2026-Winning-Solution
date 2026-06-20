# Data

The competition data is **not redistributed** in this repository. It belongs to the
**WiDS Datathon 2026** Kaggle competition and is subject to that competition's data-use terms.

## How to obtain it

1. Accept the competition rules on Kaggle (WiDS Datathon 2026).
2. Download `train.csv` and `test.csv`.
3. Place both files in this `data/` directory:

   ```
   data/
   ├── train.csv
   └── test.csv
   ```

Paths are configured in [`../SETTINGS.json`](../SETTINGS.json) and default to this directory, so no
further changes are needed if you place the files here.

## Dataset shape

- **221** training fires, **95** test fires.
- A fire is "close" if `dist_min_ci_0_5h < 5000` (metres). All 69 close training fires are `event=1`;
  all 152 far fires are `event=0`.

## Required columns

The pipeline depends on these columns being present:

| Column | Where | Purpose |
|---|---|---|
| `dist_min_ci_0_5h` | train & test | Defines the 5000 m close-fire threshold. |
| `event_id` | test | Row identifier for the submission file. |
| event / time-to-event fields | train | Survival targets used to fit the models. |

See [`../research/METHODOLOGY.md`](../research/METHODOLOGY.md) for the full feature list and the
`engineer()` feature-engineering step.
