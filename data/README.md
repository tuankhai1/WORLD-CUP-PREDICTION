# Data Inputs

This repository intentionally does not commit large/raw football datasets. The
pipeline expects the following local files under `data/raw/` before running the
full training workflow. The cleaned outputs under `data/processed/` are
generated artifacts and can be rebuilt from these sources.

## Raw input files

| File | Used by | Purpose | Expected shape |
| --- | --- | --- | --- |
| `github_historical.parquet` | `DataMerger` | International match results fetched from `martj42/international_results` by `GithubDataLoader`, excluding current WC 2026 final-tournament rows. | Match-level rows with `date`, `home_team`, `away_team`, `home_score`, `away_score`, and competition/status metadata when available. |
| `github_wc2026_results.parquet` | `DataMerger` | Current WC 2026 final-tournament rows separated from the martj42 feed. | Same match-level schema as `github_historical.parquet`; used only as current-tournament result candidates. |
| `wc2026_matches.parquet` | `DataMerger`, simulator | Local 2026 World Cup fixtures/results. These rows win de-duplication and result locking when they overlap with GitHub rows. | Fixture/result rows with `date`, teams, scores when finished, `stage`, `group`, `status`, `matchday`, `competition`, and `season`. |
| `wc2026_standings.parquet` | dashboard/diagnostics | Current tournament standings snapshot. | Group/table rows with team, played, won, drawn, lost, goals, goal difference, and points. |
| `SquadLists-English.pdf` | `PlayerStatsLoader` | Official tournament squad list PDF. | FIFA-style country pages containing player rows and country three-letter codes. |
| `players_data-2025_2026.csv` | `PlayerStatsLoader` | Backward-compatible club-season player statistics export. | Player rows with at least `Nation`, `Player`, `Pos`, `Gls`, `Ast`, `SoT`, `+/-`, `Int`, `TklW`, `PPM`, `Saves`, and `+/-90`; missing metric columns are filled with zero. |
| `player_stats/*.csv` | `PlayerStatsLoader` | Optional extra league/player-stat exports beyond the Top 5 leagues. | Same preferred columns as above. Common aliases like `Goals`, `Assists`, `Team`, `Club`, `League`, and `Nationality` are normalized automatically. |
| `squad_players.csv` | `PlayerStatsLoader` | Optional normalized official roster table for true squad-aware form. | Rows with `team` plus one of `player_name`, `Player`, `player`, `Name`, or `name`; used to match each squad player to club-season stats. |

## Source classification

| Source | Classification | Provides past results? | Provides current WC 2026 results? | Main pipeline use |
| --- | --- | ---: | ---: | --- |
| `github_historical.parquet` | Historical match-results source from `martj42/international_results`. | Yes | No | Training rows, Elo history, and rolling form after current WC 2026 rows are separated. |
| `github_wc2026_results.parquet` | Current-tournament result candidates from `martj42/international_results`. | No | Yes | Fills fresh WC 2026 scores when local fixture rows are still blank. Excluded from historical training labels. |
| `wc2026_matches.parquet` | Local current-tournament fixture/result source. | No | Yes | Preferred source for WC 2026 fixture metadata and locked completed match scores. Excluded from historical training labels. |
| `wc2026_standings.parquet` | Current-tournament standings snapshot. | No | No direct match rows | Diagnostics/dashboard context. |
| `players_data-2025_2026.csv` | Current player club-form source. | No | No | Squad/player strength features through `club_form_power`. |
| `player_stats/*.csv` | Additional player club-form exports. | No | No | Expanded player coverage beyond the main CSV. |
| `SquadLists-English.pdf` | Official squad-reference source. | No | No | Squad parsing and roster-aware player matching. |
| `squad_players.csv` | Normalized squad roster source. | No | No | Preferred roster-aware join when available. |

`DataMerger` also writes `data/processed/data_sources.csv`, a machine-readable
manifest with the same classification plus availability and row-count checks.

## Generated cleaned outputs

| File | Description |
| --- | --- |
| `data/processed/matches.parquet` | Unified match table after team-name standardization, source tagging, WC 2026 flags, result labels, goal difference, confederations, and FIFA-rating priors. |
| `data/processed/wc2026_results.csv` | Scored 2026 World Cup final-tournament matches only. This excludes qualifiers, unplayed fixtures, placeholder knockout rows, and source duplicates. |
| `data/processed/wc2026_results.parquet` | Parquet copy of the exclusive WC 2026 scored-results table. |
| `data/processed/combined_international_results.csv` | Scored combined international history with current WC 2026 final-tournament results excluded. |
| `data/processed/combined_international_results.parquet` | Parquet copy of the clean combined international-results table. |
| `data/processed/data_sources.csv` | Data source classification manifest. |
| `data/processed/team_stats.parquet` | Team-level aggregate fallback stats derived from finished match history. |

## How to regenerate

1. Run `python update.py` or the project ingestion step to refresh historical
   international results into `data/raw/github_historical.parquet` and current
   WC 2026 rows into `data/raw/github_wc2026_results.parquet`.
2. Refresh or replace `data/raw/wc2026_matches.parquet` with current
   tournament fixtures/results.
3. Place the official squad PDF at `data/raw/SquadLists-English.pdf`.
4. Optionally export normalized official roster rows to `data/raw/squad_players.csv`.
5. Export the latest season player statistics to
   `data/raw/players_data-2025_2026.csv`, or place multiple league exports in
   `data/raw/player_stats/`.
6. Run `python src/data_ingestion/player_stats_loader.py` to build
   `data/processed/squad_ratings.parquet`.
7. Run `python -c "from src.data_ingestion.data_merger import DataMerger; DataMerger().merge()"`
   to rebuild `matches.parquet`, `wc2026_results.csv`,
   `combined_international_results.csv`, and `data_sources.csv`.
8. Run `python main.py --mode full --iterations 1000000` to train, simulate,
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
