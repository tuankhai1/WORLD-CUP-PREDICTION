# 2026 FIFA World Cup Prediction Model

A machine learning and Monte Carlo simulation pipeline for forecasting the
2026 FIFA World Cup.

The project combines modern international match results, tournament fixtures,
team form, squad/player strength, gradient-boosted tabular models, and
tournament simulation. Its final output is a standalone HTML dashboard with
winner probabilities, group advancement projections, matchup predictions,
training metrics, and an explanatory bracket view.

## Highlights

- End-to-end pipeline from raw football data to dashboard output.
- Feature engineering for Elo, short-term form Elo, rolling goals/xG,
  pressing proxies, recent momentum, confederation encodings, FIFA ratings,
  and squad club-form power.
- Stacked ensemble model using CatBoost, XGBoost, LightGBM, logistic
  regression, Ridge regression, and probability calibration.
- Monte Carlo tournament simulator with a C++/pybind11 engine when compiled
  and a Python fallback when fixed/live results are locked in.
- Update workflow for refreshing local results and regenerating predictions
  without retraining the full model.

## Repository Structure

```text
.
|-- config.py                  # Central tournament, model, Elo, and path settings
|-- main.py                    # CLI entry point for full, quick, update, predict, simulate, dashboard modes
|-- update.py                  # Daily/update workflow wrapper
|-- setup.py                   # Builds the optional C++ Monte Carlo extension
|-- requirements.txt           # Python dependencies
|-- README.md                  # Project documentation
|-- data/
|   |-- README.md              # Detailed raw data requirements
|   |-- raw/                   # Local raw inputs, not tracked by git
|   |-- processed/             # Generated parquet feature/data artifacts
|   `-- cache/                 # Local ingestion/cache files
|-- models/                    # Trained models, feature columns, and pipeline state
|-- output/                    # Generated dashboard and simulation JSON
|-- catboost_info/             # CatBoost training logs
|-- build/                     # Native extension build output
`-- src/
    |-- data_ingestion/
    |   |-- github_loader.py           # Downloads international results into data/raw
    |   |-- data_merger.py             # Standardizes and merges historical + local 2026 data
    |   `-- player_stats_loader.py     # Builds squad/player club-form ratings
    |-- features/
    |   |-- elo_rating.py              # Long-term Elo and fast-decay form Elo
    |   |-- pipeline.py                # End-to-end feature matrix builder
    |   |-- rolling_xg.py              # Rolling goals/xG estimates
    |   |-- pressing_intensity.py      # Pressing-style proxy features
    |   |-- form_momentum.py           # Recent form, streak, trend features
    |   `-- encoding.py                # Team/confederation encodings
    |-- model/
    |   |-- base_models.py             # CatBoost, XGBoost, LightGBM wrappers
    |   |-- optuna_tuning.py           # Hyperparameter tuning
    |   |-- stacking.py                # Stacked ensemble and calibration
    |   `-- predict.py                 # Match prediction and probability matrix generation
    |-- simulation/
    |   |-- tournament.py              # Tournament/group definitions and result locking
    |   |-- simulator.py               # Python/C++ simulation orchestrator
    |   |-- mc_engine.cpp              # Native Monte Carlo engine implementation
    |   `-- mc_bindings.cpp            # pybind11 bindings
    `-- output/
        `-- dashboard.py               # HTML dashboard renderer
```

## Architecture

The system is organized as a layered prediction pipeline.

### 1. Data Ingestion

`GithubDataLoader` refreshes historical international results from
`martj42/international_results`. `DataMerger` then combines those results with
local 2026 World Cup fixtures/results from `data/raw/wc2026_matches.parquet`
when that file exists. Local 2026 rows are loaded first, so they win
deduplication for the same date and teams.

`PlayerStatsLoader` computes a squad-level `club_form_power` rating. If
`data/raw/squad_players.csv` exists, the loader performs a roster-aware join
between official squad players and current-season club statistics. If not, it
falls back to a nationality-pool method.

### 2. Feature Engineering

`FeaturePipeline` converts merged match data into team features and matchup
vectors. Elo is processed chronologically from the modern era, while the
machine-learning training set is focused on the current World Cup cycle.

Each matchup is represented from `team_a`'s perspective using differences such
as Elo difference, form-Elo difference, FIFA-rating difference, xG difference,
pressing difference, recent-form difference, and club-form-power difference.
Absolute team values are also retained for the tree models.

### 3. Modeling

The modeling layer tunes and trains three base learners:

- CatBoost
- XGBoost
- LightGBM

Each base model predicts both win/draw/loss probabilities and expected goal
differential. Their out-of-fold outputs become meta-features for a second-level
stacked model:

- Logistic regression for the final win/draw/loss blend.
- Ridge regression for expected goal differential.
- Sigmoid calibration for probability calibration.

### 4. Match Prediction

`MatchPredictor` loads the saved feature pipeline and ensemble, predicts any
ordered matchup, and then symmetry-corrects the result by averaging
`predict(A, B)` with the complement of `predict(B, A)`. It also generates a
complete probability matrix for every tournament-team pairing.

### 5. Tournament Simulation

`TournamentSimulator` consumes the probability matrix and simulates the full
48-team tournament:

- Group-stage round robins.
- Top two teams from each group.
- Eight best third-place teams.
- Knockout rounds with extra time and penalties for unresolved draws.
- Optional locking of completed 2026 tournament results from local data.

When the native `mc_simulation` module is available and no locked results are
present, the simulator uses the C++ backend. Otherwise, it falls back to the
Python simulator. In live-update mode, the Python simulator is capped at
100,000 iterations to keep runs practical.

### 6. Output

`DashboardGenerator` writes `output/dashboard.html`, using simulation
aggregates, group-match predictions, pairwise probabilities, and model metrics.
Simulation results are also saved as `output/simulation_results.json`.

## Data Requirements

Large/raw data is intentionally excluded from git. Before running a full
training workflow, prepare the local inputs under `data/raw/`.

| File | Required | Purpose |
| --- | --- | --- |
| `github_historical.parquet` | Yes, generated by ingestion | Historical international results fetched by `GithubDataLoader`. |
| `wc2026_matches.parquet` | Recommended | Local 2026 fixtures/results used for tournament prediction and result locking. |
| `SquadLists-English.pdf` | Recommended | Official squad PDF used by `PlayerStatsLoader`. |
| `players_data-2025_2026.csv` | Recommended | Current-season player statistics for squad form. |
| `player_stats/*.csv` | Optional | Extra league/player exports beyond the main player file. |
| `squad_players.csv` | Optional | Normalized official roster table for more accurate squad-aware form. |

See `data/README.md` for the expected columns and the fallback behavior for
player and squad statistics.

## Installation

### 1. Create an environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Build the optional C++ simulation engine

```bash
python setup.py build_ext --inplace
```

This step is optional. Without it, the project still works through the Python
simulation fallback, but large Monte Carlo runs will be slower.

On Windows, the C++ build expects a compiler toolchain compatible with
`/std:c++17` and OpenMP. On Linux/macOS, it uses `-std=c++17` and `-fopenmp`.

## Step-by-Step Implementation Guide

### Step 1: Fetch or place raw data

The full pipeline automatically tries to fetch historical results through
`GithubDataLoader`. To refresh only that dataset before a run, execute:

```bash
python -c "from src.data_ingestion.github_loader import GithubDataLoader; GithubDataLoader().fetch_data()"
```

Then place any local tournament and player inputs in `data/raw/`, especially
`wc2026_matches.parquet` and current-season player statistics if you want
squad-form features.

Use `update.py` or `python main.py --mode update` after a model has already
been trained, because update mode loads saved model artifacts before
regenerating predictions.

### Step 2: Build squad form ratings

If squad/player files are available, generate squad ratings:

```bash
python src/data_ingestion/player_stats_loader.py
```

This writes files such as:

- `data/processed/squad_ratings.parquet`
- `data/processed/squad_player_form.parquet` when roster-aware data exists

### Step 3: Run a quick validation pipeline

Use a small run to verify that ingestion, features, modeling, simulation, and
dashboard generation all connect correctly:

```bash
python main.py --mode test --iterations 10000
```

### Step 4: Train the full model

Run the full pipeline once data is ready:

```bash
python main.py --mode full --iterations 1000000
```

This performs:

1. Historical data refresh.
2. Data merging and standardization.
3. Squad/player rating calculation.
4. Feature matrix generation.
5. Optuna tuning.
6. Stacked ensemble training.
7. Pairwise matchup prediction.
8. Monte Carlo tournament simulation.
9. Dashboard rendering.

### Step 5: Inspect generated artifacts

After a successful run, check:

- `models/stacked_ensemble.joblib`
- `models/feature_pipeline.joblib`
- `models/feature_columns.csv`
- `data/processed/feature_matrix.parquet`
- `data/processed/targets.parquet`
- `output/simulation_results.json`
- `output/dashboard.html`
- `pipeline.log`

### Step 6: Predict a single matchup

After a model has been trained and saved:

```bash
python main.py --mode predict --teams Spain Brazil
```

### Step 7: Re-run simulation with saved models

Use this when models are already trained and only the tournament simulation
needs another run:

```bash
python main.py --mode simulate --iterations 1000000
```

### Step 8: Regenerate only the dashboard

Use this when simulation results already exist:

```bash
python main.py --mode dashboard
```

### Step 9: Update with latest local results

Use update mode when completed 2026 tournament matches have been added to the
local fixture/result file:

```bash
python main.py --mode update --iterations 100000
```

The simulator locks finished 2026 rows when they include explicit tournament
metadata such as `season = 2026`, a non-empty `stage`, and a finished status.

## Command Line Reference

```bash
# Complete pipeline: ingest, engineer features, tune, train, simulate, dashboard
python main.py --mode full --iterations 1000000

# Faster model run with fewer Optuna trials
python main.py --mode quick --iterations 100000

# Medium model run with an intermediate number of Optuna trials
python main.py --mode medium --iterations 250000

# Very fast smoke test
python main.py --mode test --iterations 10000

# Update saved model predictions with refreshed data and locked results
python main.py --mode update --iterations 100000

# Predict a specific matchup
python main.py --mode predict --teams Spain Brazil

# Re-run simulation only using saved model artifacts
python main.py --mode simulate --iterations 1000000

# Rebuild dashboard HTML from saved simulation/model artifacts
python main.py --mode dashboard

# Benchmark base models and refresh the README result table
python main.py --mode benchmark --benchmark-cv time --benchmark-folds 5

# Paper-style 10-fold stratified K-fold benchmark
python main.py --mode benchmark --benchmark-cv kfold --benchmark-folds 10
```

## Mathematical Notes

The implementation keeps the formulas simple enough to audit while still
capturing match importance, margin of victory, recency, and inactivity decay.

### Elo Expected Result

For team A against team B:

<p align="center">
  <strong>E<sub>A</sub></strong> =
  <strong>1</strong> /
  (1 + 10<sup>-((R<sub>A</sub> + H - R<sub>B</sub>) / 400)</sup>)
</p>

Where:

- <strong>E<sub>A</sub></strong> is team A's expected result.
- <strong>R<sub>A</sub></strong> and <strong>R<sub>B</sub></strong> are the current Elo ratings.
- <strong>H</strong> is home advantage, set to zero for neutral World Cup matches.

### Match Result Encoding

<p align="center">
S<sub>A</sub> =
{ 1 for win, 0.5 for draw, 0 for loss }
</p>

The model target uses `2 = win`, `1 = draw`, and `0 = loss` from the home/team-A
perspective, while Elo uses the standard score above.

### Margin-of-Victory Multiplier

<p align="center">
M = 1 + ln(min(|G<sub>A</sub> - G<sub>B</sub>|, C) + 1)
</p>

Where <strong>C</strong> is the configured margin cap. In this project,
`ELO_MOV_CAP = 3`.

### Elo Update

<p align="center">
R'<sub>A</sub> = R<sub>A</sub> + K &times; M &times; (S<sub>A</sub> - E<sub>A</sub>)
</p>

<p align="center">
R'<sub>B</sub> = R<sub>B</sub> - K &times; M &times; (S<sub>A</sub> - E<sub>A</sub>)
</p>

The K-factor is competition-aware:

| Competition type | K-factor |
| --- | ---: |
| Friendly | 20 |
| Qualifier | 30 |
| Continental tournament | 40 |
| World Cup | 50 |

The form Elo system uses the same update structure with a higher K multiplier
to react faster to recent results.

### Inactivity Decay

Before each new match, both long-term Elo and form Elo decay toward the neutral
baseline:

<p align="center">
R<sub>decayed</sub> =
B + (R<sub>previous</sub> - B) &times; exp(-ln(2) &times; D / L)
</p>

Where:

- <strong>B</strong> is the neutral baseline rating.
- <strong>D</strong> is elapsed days since the team's previous match.
- <strong>L</strong> is the half-life.

The configured half-lives are:

- Long-term Elo: 1,095 days.
- Form Elo: 180 days.

### Training Sample Weight

Training examples are weighted by recency and competition importance:

<p align="center">
W<sub>recency</sub> = max(W<sub>min</sub>, 0.5<sup>A / L</sup>)
</p>

<p align="center">
W<sub>sample</sub> = W<sub>recency</sub> &times; W<sub>importance</sub>
</p>

Where:

- <strong>A</strong> is match age in days relative to the latest training match.
- <strong>L</strong> is the recency half-life, currently 270 days.
- <strong>W<sub>min</sub></strong> is the minimum retained weight, currently 0.03.
- <strong>W<sub>importance</sub></strong> is based on match type.

### Squad Club-Form Power

When roster-aware squad data is available:

<p align="center">
P<sub>club</sub> = 0.65 &times; mean(top XI player form) + 0.35 &times; mean(depth player form)
</p>

This value is written as `club_form_power` and included in the team feature
matrix.

## Model Benchmark Results

<!-- MODEL_BENCHMARK_START -->
Latest benchmark: `5`-fold `TimeSeriesSplit` on `data/processed/feature_matrix.parquet`.

| Category | Model | CV accuracy (%) | F1 micro (%) | AUROC micro | Logloss |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | Class-prior baseline | 47.35 | 47.35 | 0.619 | 1.0535 |
| Baseline | Elo/FIFA Logistic Regression | 57.64 | 57.64 | 0.749 | 0.9189 |
| Baseline | Logistic Regression | 58.35 | 58.35 | 0.756 | 0.9205 |
| Neural network | Neural Net (MLP) | 58.40 | 58.40 | 0.753 | 0.9169 |
| Tree-based | Decision Tree | 56.30 | 56.30 | 0.732 | 1.1952 |
| Tree-based | Random Forest | 56.75 | 56.75 | 0.746 | 0.9304 |
| Tree-based | Extra Trees | 56.80 | 56.80 | 0.736 | 0.9575 |
| Tree-based | Gradient Boosting tree | 58.24 | 58.24 | 0.757 | 0.9108 |
| Tree-based | AdaBoost tree | 58.61 | 58.61 | 0.758 | 1.0083 |
| Tree-based | HistGradientBoosting | 56.22 | 56.22 | 0.740 | 1.0886 |
| Tree-based | CatBoost | 59.48 | 59.48 | 0.763 | 0.8999 |
| Tree-based | XGBoost | 58.69 | 58.69 | 0.758 | 0.9100 |
| Tree-based | LightGBM | 57.56 | 57.56 | 0.749 | 0.9347 |
| Ultimate ensemble | Ultimate Ensemble (weighted soft vote) | 59.06 | 59.06 | 0.762 | 0.9024 |
<!-- MODEL_BENCHMARK_END -->

The default benchmark uses `TimeSeriesSplit` because match data is temporal:
models should train on earlier matches and validate on later matches. A
10-fold stratified K-fold run is available for comparison with papers or
classroom-style result tables, but it can be optimistic because future-era
matches are allowed to help train folds that validate on earlier-era matches.
Use K-fold for broad model comparison, not as the final tournament-selection
metric.

## Modeling Caveats

- The configured groups in `config.py` include placeholder teams for unresolved
  playoff spots. Update `GROUPS`, `ALL_TEAMS`, rankings, aliases, and raw
  fixture data once the final field is known.
- The dashboard bracket view is explanatory bracketology. The simulator
  advances the top two teams and eight best third-place teams, but the exact
  official FIFA third-place knockout mapping may need refinement when FIFA's
  final assignment rules are confirmed.
- Live result locking currently uses the Python simulator because the compiled
  engine does not accept fixed scores yet.
- Matchup predictions are symmetry-corrected by averaging both team orders,
  but model quality still depends heavily on the quality of the underlying
  raw data and squad inputs.

## Development Notes

- Generated data, model files, build outputs, logs, and dashboards are ignored
  by git to keep the repository lightweight.
- Main generated locations are `data/`, `models/`, `output/`, `build/`,
  `catboost_info/`, and `*.log`.
- Use `data/README.md` as the source of truth for local raw-data preparation.
- Use `pipeline.log` and `update.log` for run diagnostics.
