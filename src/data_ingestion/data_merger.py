"""
Combines data from all three sources (football-data.
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TOURNAMENT_YEAR,
    ALL_TEAMS,
    TEAM_TO_CONFEDERATION,
    FIFA_RANKINGS,
    resolve_team_name,
)

logger = logging.getLogger(__name__)


class DataMerger:
    """
    Merges and standardizes data from multiple football data sources
    into a unified dataset suitable for feature engineering.
    
    Data sources:
    1. Local 2026 World Cup fixture/result file, when present
    2. GitHub (martj42/international_results): Comprehensive match results
    
    Output:
    - Unified match-level DataFrame with standardized team names
    - Team-level aggregate statistics DataFrame
    """

    def __init__(self):
        """Initialize the merger."""
        self.matches_df: pd.DataFrame = pd.DataFrame()
        self.team_stats_df: pd.DataFrame = pd.DataFrame()

    def load_all_raw_data(self) -> dict:
        """
        Load all available raw data files.
        
        Returns:
            Dict mapping source name to DataFrame
        """
        data = {}

        # Live/current tournament file. This lets update jobs lock in actual
        # 2026 results without waiting for the historical GitHub dataset.
        wc_path = RAW_DATA_DIR / "wc2026_matches.parquet"
        if wc_path.exists():
            data["wc2026"] = pd.read_parquet(wc_path)
            logger.info(f"Loaded {len(data['wc2026'])} matches (WC 2026 local)")
        
        # GitHub data
        gh_path = RAW_DATA_DIR / "github_historical.parquet"
        if gh_path.exists():
            data["github"] = pd.read_parquet(gh_path)
            logger.info(f"Loaded {len(data['github'])} matches (GitHub)")

        gh_wc_path = RAW_DATA_DIR / "github_wc2026_results.parquet"
        if gh_wc_path.exists():
            data["github_wc2026"] = pd.read_parquet(gh_wc_path)
            logger.info(f"Loaded {len(data['github_wc2026'])} WC 2026 matches (GitHub)")

        if not data:
            logger.warning("No raw data files found! Run data ingestion first.")
            
        return data

    def _standardize_team_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply team name resolution to all team columns."""
        for col in ["home_team", "away_team", "team"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: resolve_team_name(str(x)) if pd.notna(x) else x)
        return df

    @staticmethod
    def _source_metadata(source: str) -> dict:
        """Return provenance labels for each raw source used by the merger."""
        registry = {
            "wc2026": {
                "source_role": "current_tournament_fixture_results",
                "source_provider": "local_fixture_file",
            },
            "github": {
                "source_role": "historical_past_results",
                "source_provider": "martj42/international_results",
            },
            "github_wc2026": {
                "source_role": "current_tournament_result_candidates",
                "source_provider": "martj42/international_results",
            },
        }
        return registry.get(
            source,
            {
                "source_role": "auxiliary_data",
                "source_provider": "local_file",
            },
        )

    @staticmethod
    def _series(df: pd.DataFrame, column: str, default=np.nan) -> pd.Series:
        """Return an existing column or an index-aligned default series."""
        if column in df.columns:
            return df[column]
        return pd.Series(default, index=df.index)

    def _tag_source(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Attach consistent source provenance columns."""
        metadata = self._source_metadata(source)
        df["source"] = source
        df["source_role"] = metadata["source_role"]
        df["source_provider"] = metadata["source_provider"]
        return df

    def _github_wc2026_mask(self, df: pd.DataFrame) -> pd.Series:
        """Identify final-tournament WC 2026 rows inside the GitHub source."""
        dates = pd.to_datetime(self._series(df, "date"), errors="coerce")
        competition = (
            self._series(df, "competition", "")
            .fillna("")
            .astype(str)
            .str.lower()
        )
        is_world_cup = competition.str.contains("world cup", na=False)
        is_qualification = competition.str.contains("qual", na=False)
        return dates.dt.year.eq(TOURNAMENT_YEAR) & is_world_cup & ~is_qualification

    def _add_match_context_flags(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Mark scored rows and isolate 2026 World Cup final-tournament matches."""
        matches = matches.copy()

        dates = pd.to_datetime(self._series(matches, "date"), errors="coerce")
        competition = (
            self._series(matches, "competition", "")
            .fillna("")
            .astype(str)
            .str.lower()
        )
        season = pd.to_numeric(self._series(matches, "season"), errors="coerce")
        stage = self._series(matches, "stage")

        has_teams = (
            self._series(matches, "home_team").notna()
            & self._series(matches, "away_team").notna()
        )
        has_scores = (
            self._series(matches, "home_score").notna()
            & self._series(matches, "away_score").notna()
        )

        is_world_cup_comp = competition.str.contains("world cup", na=False)
        is_qualification = competition.str.contains("qual", na=False)
        is_final_tournament = is_world_cup_comp & ~is_qualification

        # Local fixture files carry explicit season/stage metadata. GitHub rows
        # carry competition/date metadata only, so support both forms.
        is_local_wc2026 = season.eq(TOURNAMENT_YEAR) | stage.notna()
        is_github_wc2026 = dates.dt.year.eq(TOURNAMENT_YEAR) & is_final_tournament

        matches["has_result"] = has_teams & has_scores
        matches["is_world_cup_final_tournament"] = is_final_tournament | is_local_wc2026
        matches["is_wc2026"] = is_local_wc2026 | is_github_wc2026
        matches["is_wc2026_result"] = matches["is_wc2026"] & matches["has_result"]
        matches["is_combined_international_result"] = (
            matches["has_result"] & ~matches["is_wc2026_result"]
        )
        matches["use_for_model_training"] = matches["is_combined_international_result"]

        return matches

    def _add_result_columns(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Compute result columns only when both scores are present."""
        if "home_score" not in matches.columns or "away_score" not in matches.columns:
            return matches

        score_mask = matches["home_score"].notna() & matches["away_score"].notna()
        matches["result"] = np.nan
        matches.loc[score_mask & (matches["home_score"] > matches["away_score"]), "result"] = 2
        matches.loc[score_mask & (matches["home_score"] == matches["away_score"]), "result"] = 1
        matches.loc[score_mask & (matches["home_score"] < matches["away_score"]), "result"] = 0
        matches["goal_diff"] = np.nan
        matches.loc[score_mask, "goal_diff"] = (
            matches.loc[score_mask, "home_score"] - matches.loc[score_mask, "away_score"]
        )

        return matches

    def _coalesce_exact_duplicate_matches(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Merge same-date/team duplicates while preferring scored rows."""
        keys = ["date", "home_team", "away_team"]
        if matches.empty or any(key not in matches.columns for key in keys):
            return matches

        working = matches.copy()
        working["_row_order"] = np.arange(len(working))
        working["_has_scores"] = (
            working["home_score"].notna() & working["away_score"].notna()
            if {"home_score", "away_score"}.issubset(working.columns)
            else False
        )
        source_priority = {"wc2026": 0, "github_wc2026": 1, "github": 2}
        working["_source_priority"] = working["source"].map(source_priority).fillna(9)
        working["score_source"] = pd.NA
        working.loc[working["_has_scores"], "score_source"] = working.loc[
            working["_has_scores"], "source"
        ]
        working["fixture_source"] = pd.NA
        working.loc[working["source"].eq("wc2026"), "fixture_source"] = "wc2026"

        key_mask = working[keys].notna().all(axis=1)
        keyed = working[key_mask].copy()
        unkeyed = working[~key_mask].copy()

        records = []
        for _, group in keyed.groupby(keys, dropna=False, sort=False):
            row_order = group.sort_values(
                ["_has_scores", "_source_priority", "_row_order"],
                ascending=[False, True, True],
            )
            base = row_order.iloc[0].copy()

            metadata_order = group.sort_values(["_source_priority", "_row_order"])
            for col in working.columns:
                if col.startswith("_"):
                    continue
                value = base[col]
                is_missing = pd.isna(value) or (isinstance(value, str) and not value.strip())
                if not is_missing:
                    continue

                candidates = metadata_order[col].dropna()
                if not candidates.empty:
                    base[col] = candidates.iloc[0]

            records.append(base)

        coalesced = pd.DataFrame(records)
        cleaned = pd.concat([coalesced, unkeyed], ignore_index=True, sort=False)
        cleaned = cleaned.drop(columns=["_row_order", "_has_scores", "_source_priority"])
        removed = len(matches) - len(cleaned)
        if removed:
            logger.info(f"Coalesced {removed} exact duplicate match rows")

        return cleaned

    def _drop_near_duplicate_wc2026_results(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Collapse duplicate scored WC 2026 rows caused by source date offsets."""
        if matches.empty or "is_wc2026_result" not in matches.columns:
            return matches

        working = matches.copy()
        working["_row_id"] = np.arange(len(working))
        result_rows = working[working["is_wc2026_result"]].copy()
        if result_rows.empty:
            return matches

        result_rows["_date"] = pd.to_datetime(result_rows["date"], errors="coerce")
        result_rows["_source_priority"] = (
            result_rows["source"].map({"wc2026": 0, "github_wc2026": 1, "github": 2}).fillna(9)
        )

        duplicate_keys = ["home_team", "away_team", "home_score", "away_score"]
        drop_ids = set()

        for _, group in result_rows.groupby(duplicate_keys, dropna=False):
            if len(group) <= 1:
                continue

            dates = group["_date"].dropna()
            if dates.empty:
                continue

            # Treat rows as source duplicates only when they describe the same
            # scored match within a small timestamp/date-publication offset.
            if (dates.max() - dates.min()).days <= 2:
                keep_id = (
                    group.sort_values(["_source_priority", "_date"])["_row_id"].iloc[0]
                )
                drop_ids.update(set(group["_row_id"]) - {keep_id})

        if not drop_ids:
            return matches

        cleaned = working[~working["_row_id"].isin(drop_ids)].drop(columns=["_row_id"])
        logger.info(f"Dropped {len(drop_ids)} near-duplicate WC 2026 result rows")
        return cleaned.reset_index(drop=True)

    def _coalesce_near_duplicate_wc2026_fixtures(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Attach local fixture metadata to scored rows with small date offsets."""
        required = {"date", "home_team", "away_team", "is_wc2026", "has_result", "source"}
        if matches.empty or not required.issubset(matches.columns):
            return matches

        working = matches.copy()
        working["_row_id"] = np.arange(len(working))
        working["_date"] = pd.to_datetime(working["date"], errors="coerce")

        scored = working[working["is_wc2026"] & working["has_result"]].copy()
        fixtures = working[
            working["is_wc2026"]
            & ~working["has_result"]
            & working["source"].eq("wc2026")
            & working["home_team"].notna()
            & working["away_team"].notna()
            & working["_date"].notna()
        ].copy()

        if scored.empty or fixtures.empty:
            return matches

        drop_ids = set()
        metadata_columns = [
            "stage",
            "group",
            "matchday",
            "competition",
            "season",
            "fixture_source",
        ]

        for scored_idx, scored_row in scored.iterrows():
            scored_date = scored_row["_date"]
            if pd.isna(scored_date):
                continue

            candidates = fixtures[
                fixtures["home_team"].eq(scored_row["home_team"])
                & fixtures["away_team"].eq(scored_row["away_team"])
            ].copy()
            if candidates.empty:
                continue

            candidates["_date_delta_days"] = (candidates["_date"] - scored_date).abs().dt.days
            candidates = candidates[candidates["_date_delta_days"] <= 2]
            if candidates.empty:
                continue

            fixture = candidates.sort_values("_date_delta_days").iloc[0]
            for col in metadata_columns:
                if col not in working.columns:
                    continue
                current = working.at[scored_idx, col]
                fixture_value = fixture.get(col)
                current_missing = pd.isna(current) or (
                    isinstance(current, str) and not current.strip()
                )
                if current_missing and pd.notna(fixture_value):
                    working.at[scored_idx, col] = fixture_value

            if "fixture_source" in working.columns:
                working.at[scored_idx, "fixture_source"] = "wc2026"
            drop_ids.add(fixture["_row_id"])

        if not drop_ids:
            return matches

        cleaned = working[~working["_row_id"].isin(drop_ids)].drop(columns=["_row_id", "_date"])
        logger.info(f"Coalesced {len(drop_ids)} near-duplicate WC 2026 fixture rows")
        return cleaned.reset_index(drop=True)

    def _build_match_dataset(self, raw_data: dict) -> pd.DataFrame:
        """
        Build unified match dataset from all sources.
        """
        all_matches = []

        # 1. Current tournament fixtures/results should win de-duplication.
        if "wc2026" in raw_data:
            df = raw_data["wc2026"].copy()
            df = self._standardize_team_names(df)
            df = self._tag_source(df, "wc2026")
            all_matches.append(df)

        # 2. Current tournament rows from GitHub should be available for live
        # scores, but not mixed into the historical GitHub result bucket.
        if "github_wc2026" in raw_data:
            df = raw_data["github_wc2026"].copy()
            df = self._standardize_team_names(df)
            df = self._tag_source(df, "github_wc2026")
            all_matches.append(df)

        # 3. GitHub historical matches
        if "github" in raw_data:
            df = raw_data["github"].copy()
            df = self._standardize_team_names(df)
            wc2026_mask = self._github_wc2026_mask(df)
            if wc2026_mask.any():
                current_wc = self._tag_source(df[wc2026_mask].copy(), "github_wc2026")
                all_matches.append(current_wc)
                logger.info(
                    f"Separated {len(current_wc)} GitHub WC 2026 rows "
                    "from historical results"
                )

            historical = df[~wc2026_mask].copy()
            if not historical.empty:
                historical = self._tag_source(historical, "github")
                all_matches.append(historical)

        if not all_matches:
            return pd.DataFrame()

        matches = pd.concat(all_matches, ignore_index=True)

        # Deduplicate: same date + teams = same match. Scored rows win over
        # blank fixture placeholders, while local fixture metadata fills gaps.
        matches = self._coalesce_exact_duplicate_matches(matches)

        # Add confederation info
        matches["home_confederation"] = matches["home_team"].map(TEAM_TO_CONFEDERATION)
        matches["away_confederation"] = matches["away_team"].map(TEAM_TO_CONFEDERATION)
        
        # Add FIFA ranking
        matches["home_fifa_rating"] = matches["home_team"].map(FIFA_RANKINGS)
        matches["away_fifa_rating"] = matches["away_team"].map(FIFA_RANKINGS)

        matches = self._add_match_context_flags(matches)
        matches = self._add_result_columns(matches)
        matches = self._coalesce_near_duplicate_wc2026_fixtures(matches)
        matches = self._drop_near_duplicate_wc2026_results(matches)

        # Sort by date
        if "date" in matches.columns:
            matches = matches.sort_values("date").reset_index(drop=True)

        return matches

    def _export_wc2026_results(self) -> pd.DataFrame:
        """Export scored 2026 World Cup final-tournament matches only."""
        if self.matches_df.empty or "is_wc2026_result" not in self.matches_df.columns:
            return pd.DataFrame()

        results = self.matches_df[self.matches_df["is_wc2026_result"]].copy()
        results = results.dropna(subset=["home_team", "away_team", "home_score", "away_score"])

        results["_date"] = pd.to_datetime(results["date"], errors="coerce")
        results["_source_priority"] = (
            results["source"].map({"wc2026": 0, "github_wc2026": 1, "github": 2}).fillna(9)
        )
        results = results.sort_values(["_date", "_source_priority", "home_team", "away_team"])

        export_columns = [
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "result",
            "goal_diff",
            "stage",
            "group",
            "status",
            "matchday",
            "competition",
            "season",
            "source",
            "source_role",
            "source_provider",
            "score_source",
            "fixture_source",
            "has_result",
            "is_wc2026",
            "is_wc2026_result",
        ]
        export_columns = [col for col in export_columns if col in results.columns]
        results = results[export_columns].reset_index(drop=True)

        csv_path = PROCESSED_DATA_DIR / "wc2026_results.csv"
        parquet_path = PROCESSED_DATA_DIR / "wc2026_results.parquet"
        results.to_csv(csv_path, index=False)
        results.to_parquet(parquet_path, index=False)
        logger.info(f"Saved {len(results)} scored WC 2026 results to {csv_path}")

        return results

    def _export_combined_international_results(self) -> pd.DataFrame:
        """Export scored international results excluding current WC 2026 matches."""
        if (
            self.matches_df.empty
            or "is_combined_international_result" not in self.matches_df.columns
        ):
            return pd.DataFrame()

        results = self.matches_df[self.matches_df["is_combined_international_result"]].copy()
        results = results.dropna(subset=["home_team", "away_team", "home_score", "away_score"])

        export_columns = [
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "result",
            "goal_diff",
            "status",
            "competition",
            "source",
            "source_role",
            "source_provider",
            "has_result",
            "is_wc2026_result",
            "is_combined_international_result",
            "use_for_model_training",
        ]
        export_columns = [col for col in export_columns if col in results.columns]
        results = results[export_columns].sort_values("date").reset_index(drop=True)

        csv_path = PROCESSED_DATA_DIR / "combined_international_results.csv"
        parquet_path = PROCESSED_DATA_DIR / "combined_international_results.parquet"
        results.to_csv(csv_path, index=False)
        results.to_parquet(parquet_path, index=False)
        logger.info(f"Saved {len(results)} combined international results to {csv_path}")

        return results

    def _export_source_manifest(self, raw_data: dict) -> pd.DataFrame:
        """Write a compact manifest describing what each local source provides."""
        raw_specs = [
            {
                "source_key": "wc2026",
                "file": RAW_DATA_DIR / "wc2026_matches.parquet",
                "source_role": "current_tournament_fixture_results",
                "provider": "local_fixture_file",
                "provides_past_results": False,
                "provides_current_wc2026_fixtures": True,
                "provides_current_wc2026_results": True,
                "used_for_training": False,
                "used_for_result_locking": True,
                "notes": "Preferred source for 2026 World Cup fixture metadata and locked completed tournament scores; excluded from historical training labels.",
            },
            {
                "source_key": "github",
                "file": RAW_DATA_DIR / "github_historical.parquet",
                "source_role": "historical_past_results",
                "provider": "martj42/international_results",
                "provides_past_results": True,
                "provides_current_wc2026_fixtures": False,
                "provides_current_wc2026_results": False,
                "used_for_training": True,
                "used_for_result_locking": False,
                "notes": "Primary past-results source for features/training after current WC 2026 final-tournament rows are separated.",
            },
            {
                "source_key": "github_wc2026",
                "file": RAW_DATA_DIR / "github_wc2026_results.parquet",
                "source_role": "current_tournament_result_candidates",
                "provider": "martj42/international_results",
                "provides_past_results": False,
                "provides_current_wc2026_fixtures": False,
                "provides_current_wc2026_results": True,
                "used_for_training": False,
                "used_for_result_locking": True,
                "notes": "Recent WC 2026 final-tournament rows separated from martj42 history; scores can fill local fixture placeholders.",
            },
            {
                "source_key": "wc2026_standings",
                "file": RAW_DATA_DIR / "wc2026_standings.parquet",
                "source_role": "current_tournament_standings",
                "provider": "local_standings_file",
                "provides_past_results": False,
                "provides_current_wc2026_fixtures": False,
                "provides_current_wc2026_results": False,
                "used_for_training": False,
                "used_for_result_locking": False,
                "notes": "Current standings snapshot; useful for diagnostics/dashboard context, not direct match-result training rows.",
            },
            {
                "source_key": "players_data",
                "file": RAW_DATA_DIR / "players_data-2025_2026.csv",
                "source_role": "current_player_club_form",
                "provider": "local_player_stats_export",
                "provides_past_results": False,
                "provides_current_wc2026_fixtures": False,
                "provides_current_wc2026_results": False,
                "used_for_training": False,
                "used_for_result_locking": False,
                "notes": "Current-season player statistics used to estimate squad club-form power.",
            },
            {
                "source_key": "squad_pdf",
                "file": RAW_DATA_DIR / "SquadLists-English.pdf",
                "source_role": "official_squad_reference",
                "provider": "local_official_pdf",
                "provides_past_results": False,
                "provides_current_wc2026_fixtures": False,
                "provides_current_wc2026_results": False,
                "used_for_training": False,
                "used_for_result_locking": False,
                "notes": "Official squad document parsed by PlayerStatsLoader when available.",
            },
            {
                "source_key": "squad_players",
                "file": RAW_DATA_DIR / "squad_players.csv",
                "source_role": "normalized_squad_roster",
                "provider": "local_roster_file",
                "provides_past_results": False,
                "provides_current_wc2026_fixtures": False,
                "provides_current_wc2026_results": False,
                "used_for_training": False,
                "used_for_result_locking": False,
                "notes": "Optional normalized roster table for more accurate squad-aware player form.",
            },
            {
                "source_key": "player_stats_dir",
                "file": RAW_DATA_DIR / "player_stats",
                "source_role": "additional_player_club_form",
                "provider": "local_player_stats_exports",
                "provides_past_results": False,
                "provides_current_wc2026_fixtures": False,
                "provides_current_wc2026_results": False,
                "used_for_training": False,
                "used_for_result_locking": False,
                "notes": "Optional folder of extra league/player CSV exports merged with the main player-stat file.",
            },
        ]

        records = []
        for spec in raw_specs:
            path = Path(spec["file"])
            row = spec.copy()
            row["file"] = str(path.relative_to(RAW_DATA_DIR.parent.parent))
            row["available"] = path.exists()
            row["row_count"] = len(raw_data.get(row["source_key"], [])) if row["source_key"] in raw_data else np.nan
            if row["source_key"] == "github" and "github" in raw_data:
                row["row_count"] = int((~self._github_wc2026_mask(raw_data["github"])).sum())
            if row["source_key"] == "github_wc2026" and "github" in raw_data:
                separated_count = len(raw_data.get("github_wc2026", []))
                legacy_count = int(self._github_wc2026_mask(raw_data["github"]).sum())
                row["row_count"] = separated_count + legacy_count
            if path.exists() and path.is_file() and pd.isna(row["row_count"]):
                try:
                    if path.suffix == ".parquet":
                        row["row_count"] = len(pd.read_parquet(path))
                    elif path.suffix == ".csv":
                        row["row_count"] = len(pd.read_csv(path))
                except Exception as exc:
                    logger.warning(f"Could not count rows for {path}: {exc}")
            records.append(row)

        manifest = pd.DataFrame(records)
        path = PROCESSED_DATA_DIR / "data_sources.csv"
        manifest.to_csv(path, index=False)
        logger.info(f"Saved data source manifest to {path}")

        return manifest

    def _build_team_stats(self, matches: pd.DataFrame, raw_data: dict) -> pd.DataFrame:
        """
        Build team-level aggregate statistics from match history.
        
        Used when StatsBomb/FBref data is unavailable for a team.
        Derives approximate metrics from match results.
        """
        if matches.empty:
            return pd.DataFrame()

        # Only use finished matches
        finished = matches[matches["status"] == "FINISHED"].copy() if "status" in matches.columns else matches.copy()
        
        if finished.empty:
            return pd.DataFrame()

        team_records = []

        for team in ALL_TEAMS:
            home_matches = finished[finished["home_team"] == team]
            away_matches = finished[finished["away_team"] == team]
            
            total_matches = len(home_matches) + len(away_matches)
            if total_matches == 0:
                # No match history - use FIFA ranking as baseline
                team_records.append({
                    "team": team,
                    "matches_played": 0,
                    "avg_goals_scored": 1.2,  # Global average
                    "avg_goals_conceded": 1.2,
                    "win_rate": 0.33,
                    "draw_rate": 0.33,
                    "clean_sheet_rate": 0.25,
                    "fifa_rating": FIFA_RANKINGS.get(team, 1400),
                    "confederation": TEAM_TO_CONFEDERATION.get(team, "Unknown"),
                })
                continue

            # Goals
            goals_scored = (
                home_matches["home_score"].sum() + away_matches["away_score"].sum()
            )
            goals_conceded = (
                home_matches["away_score"].sum() + away_matches["home_score"].sum()
            )
            
            # Results
            home_wins = (home_matches["home_score"] > home_matches["away_score"]).sum()
            away_wins = (away_matches["away_score"] > away_matches["home_score"]).sum()
            home_draws = (home_matches["home_score"] == home_matches["away_score"]).sum()
            away_draws = (away_matches["away_score"] == away_matches["home_score"]).sum()
            
            wins = home_wins + away_wins
            draws = home_draws + away_draws
            
            # Clean sheets
            home_cs = (home_matches["away_score"] == 0).sum()
            away_cs = (away_matches["home_score"] == 0).sum()

            team_records.append({
                "team": team,
                "matches_played": total_matches,
                "avg_goals_scored": round(goals_scored / total_matches, 2),
                "avg_goals_conceded": round(goals_conceded / total_matches, 2),
                "win_rate": round(wins / total_matches, 3),
                "draw_rate": round(draws / total_matches, 3),
                "clean_sheet_rate": round((home_cs + away_cs) / total_matches, 3),
                "fifa_rating": FIFA_RANKINGS.get(team, 1400),
                "confederation": TEAM_TO_CONFEDERATION.get(team, "Unknown"),
            })

        team_stats = pd.DataFrame(team_records)
        
        # Merge FBref advanced stats if available
        if "fbref" in raw_data and not raw_data["fbref"].empty:
            fbref = raw_data["fbref"].copy()
            fbref = self._standardize_team_names(fbref)
            team_stats = team_stats.merge(fbref, on="team", how="left")
            logger.info("Enriched team stats with FBref data")

        return team_stats

    def merge(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the full merge pipeline.
        
        Returns:
            Tuple of (matches_df, team_stats_df)
        """
        logger.info("Starting data merge pipeline...")
        
        raw_data = self.load_all_raw_data()
        
        if not raw_data:
            logger.error("No data available for merging!")
            return pd.DataFrame(), pd.DataFrame()

        # Build unified match dataset
        self.matches_df = self._build_match_dataset(raw_data)
        logger.info(f"Built unified match dataset: {len(self.matches_df)} matches")

        # Build team-level stats
        self.team_stats_df = self._build_team_stats(self.matches_df, raw_data)
        logger.info(f"Built team stats for {len(self.team_stats_df)} teams")

        # Save processed data
        if not self.matches_df.empty:
            path = PROCESSED_DATA_DIR / "matches.parquet"
            self.matches_df.to_parquet(path, index=False)
            logger.info(f"Saved processed matches to {path}")
            self._export_wc2026_results()
            self._export_combined_international_results()

        if not self.team_stats_df.empty:
            path = PROCESSED_DATA_DIR / "team_stats.parquet"
            self.team_stats_df.to_parquet(path, index=False)
            logger.info(f"Saved team stats to {path}")

        self._export_source_manifest(raw_data)

        return self.matches_df, self.team_stats_df
