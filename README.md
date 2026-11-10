# DSA211 Group 2: Nonlinearity in Bike-Sharing Demand Models

UCI Bike Sharing (hourly) data, target `cnt`. Chronological split: train Jan 2011 – Sep 2012, locked test Oct – Dec 2012.

## Notebooks (`notebooks/`, run in order)

| Notebook | What it does |
| :--- | :--- |
| `00 Train Test Split` | Reads `data/hour.csv`, splits once by date, writes `train_clean.csv` and `test_locked.csv`. |
| `01 EDA` | Data quality audit and exploratory analysis (training set only). |
| `02 Data Processing` | Drops leakage/collinear columns, dummy-encodes categoricals, writes `train_base.csv`, shows the 8 expanding-window CV folds. |
| `03 Baseline Model` | M1 OLS, VIF/rank check, residual diagnostics, outlier adjudication, CV score. |
| `04 Model Refinement` | M2–M4: log target, squared weather terms, hour × working-day interaction. |
| `05 Regularization Extension` | Ridge and LASSO on the base and M4 designs, tuned by CV. |
| `06 Model Selection and Scoring` | Picks the best CV model, scores the test set once, refits on all data. Reruns load from cache. |
| `07 Sensitivity Polynomial Degree` | Post-hoc check of weather polynomial degree (done after the test was scored; training set only). |

## Other folders

- `src/`: `data_utils.py` (loading and split), `cv_utils.py` (CV splitter, metrics, log back-transform).
- `data/`: raw data, train/test splits, processed training set, Notebook 05/06 outputs. `notebook06_*` files are the test-access guard and cache; don't delete them.
- `figures/`: saved plots used in the report.
- `docs/`: proposal, grader feedback, CV workflow guide.
- `requirements.txt`: pinned package versions.
- `agents.md`: project context and rules for AI assistants.
