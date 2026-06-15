# Data inputs

This repository intentionally does not commit large/raw football datasets. The
pipeline expects the following local files under `data/raw/` before running the
full training workflow.

## Required files

| File | Used by | Purpose | Expected shape |
| --- | --- | --- | --- |
| `github_historical.parquet` | `DataMerger` | International match results fetched from `martj42/international_results` by `GithubDataLoader`. | Match-level rows with `date`, `home_team`, `away_team`, `home_score`, `away_score`, and competition/status metadata when available. |
| `SquadLists-English.pdf` | `PlayerStatsLoader` | Official tournament squad list PDF. | FIFA-style country pages containing player rows and country three-letter codes. |
| `players_data-2025_2026.csv` | `PlayerStatsLoader` | Backward-compatible club-season player statistics export. | Player rows with at least `Nation`, `Player`, `Pos`, `Gls`, `Ast`, `SoT`, `+/-`, `Int`, `TklW`, `PPM`, `Saves`, and `+/-90`; missing metric columns are filled with zero. |
| `player_stats/*.csv` | `PlayerStatsLoader` | Optional extra league/player-stat exports beyond the Top 5 leagues. | Same preferred columns as above. Common aliases like `Goals`, `Assists`, `Team`, `Club`, `League`, and `Nationality` are normalized automatically. |
| `squad_players.csv` | `PlayerStatsLoader` | Optional normalized official roster table for true squad-aware form. | Rows with `team` plus one of `player_name`, `Player`, `player`, `Name`, or `name`; used to match each squad player to club-season stats. |

## How to regenerate

1. Run `python update.py` or the project ingestion step to refresh historical
   international results into `data/raw/github_historical.parquet`.
2. Place the official squad PDF at `data/raw/SquadLists-English.pdf`.
3. Optionally export normalized official roster rows to `data/raw/squad_players.csv`.
4. Export the latest season player statistics to
   `data/raw/players_data-2025_2026.csv`, or place multiple league exports in
   `data/raw/player_stats/`.
5. Run `python src/data_ingestion/player_stats_loader.py` to build
   `data/processed/squad_ratings.parquet`.
6. Run `python main.py --mode full --iterations 1000000` to train, simulate,
   and generate `output/dashboard.html`.

## Current squad-form workflow

If `data/raw/squad_players.csv` is present, `PlayerStatsLoader` now performs a
roster-aware join from each official squad player to the current-season player
stats table, writes `data/processed/squad_player_form.parquet`, and computes
`club_form_power` as `0.65 * top_xi_mean + 0.35 * depth_mean`.

If `squad_players.csv` is absent, the loader falls back to the older
nationality-pool method: grouping by player nationality code and summing each
nation's top 15 player form scores. That fallback is useful for experimentation,
but it is less accurate than actual squad-aware form.

## Extending player coverage beyond Top 5 leagues

Create `data/raw/player_stats/` and drop one CSV per provider/competition, for
example `turkey-super-lig.csv`, `portugal-primeira.csv`, `mls.csv`,
`saudi-pro-league.csv`, or `brazil-serie-a.csv`. The loader merges those rows
with the legacy `players_data-2025_2026.csv`, de-duplicates repeated player
rows, and recalculates `club_form_power` from the combined pool.
