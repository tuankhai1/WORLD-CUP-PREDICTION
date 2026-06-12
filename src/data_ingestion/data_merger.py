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
    1. football-data.org: Match results, scores, tournament structure
    2. StatsBomb: Event-level xG, pressing, passing (historical only)
    3. FBref: Advanced team-level aggregate stats (when available)
    
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
        
        # football-data.org data
        fd_hist_path = RAW_DATA_DIR / "historical_wc_matches.parquet"
        if fd_hist_path.exists():
            data["fd_historical"] = pd.read_parquet(fd_hist_path)
            logger.info(f"Loaded {len(data['fd_historical'])} historical matches (football-data)")

        fd_current_path = RAW_DATA_DIR / "wc2026_matches.parquet"
        if fd_current_path.exists():
            data["fd_current"] = pd.read_parquet(fd_current_path)
            logger.info(f"Loaded {len(data['fd_current'])} current matches (football-data)")

        # StatsBomb data
        sb_path = RAW_DATA_DIR / "statsbomb_historical.parquet"
        if sb_path.exists():
            data["statsbomb"] = pd.read_parquet(sb_path)
            logger.info(f"Loaded {len(data['statsbomb'])} matches (StatsBomb)")
            
        # GitHub data
        gh_path = RAW_DATA_DIR / "github_historical.parquet"
        if gh_path.exists():
            data["github"] = pd.read_parquet(gh_path)
            logger.info(f"Loaded {len(data['github'])} matches (GitHub)")

        # FBref data
        fbref_path = RAW_DATA_DIR / "fbref_advanced_stats.parquet"
        if fbref_path.exists():
            data["fbref"] = pd.read_parquet(fbref_path)
            logger.info(f"Loaded FBref stats for {len(data['fbref'])} teams")

        if not data:
            logger.warning("No raw data files found! Run data ingestion first.")
            
        return data

    def _standardize_team_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply team name resolution to all team columns."""
        for col in ["home_team", "away_team", "team"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: resolve_team_name(str(x)) if pd.notna(x) else x)
        return df

    def _build_match_dataset(self, raw_data: dict) -> pd.DataFrame:
        """
        Build unified match dataset from all sources.
        
        The priority for match data:
        1. football-data.org (most reliable for scores)
        2. StatsBomb (enriches with xG/pressing stats where available)
        """
        all_matches = []

        # 1. Historical matches from football-data.org
        if "fd_historical" in raw_data:
            df = raw_data["fd_historical"].copy()
            df = self._standardize_team_names(df)
            df["source"] = "football_data"
            all_matches.append(df)

        # 2. Current 2026 WC matches
        if "fd_current" in raw_data:
            df = raw_data["fd_current"].copy()
            df = self._standardize_team_names(df)
            df["source"] = "football_data"
            all_matches.append(df)
            
        # 3. GitHub historical matches
        if "github" in raw_data:
            df = raw_data["github"].copy()
            df = self._standardize_team_names(df)
            df["source"] = "github"
            all_matches.append(df)

        if not all_matches:
            return pd.DataFrame()

        matches = pd.concat(all_matches, ignore_index=True)

        # Deduplicate: same date + teams = same match
        matches = matches.drop_duplicates(
            subset=["date", "home_team", "away_team"],
            keep="first",
        )

        # Enrich with StatsBomb xG data where available
        if "statsbomb" in raw_data:
            sb = raw_data["statsbomb"].copy()
            sb = self._standardize_team_names(sb)
            
            # Merge on date and teams
            sb_cols = [c for c in sb.columns if c.startswith("home_") or c.startswith("away_")]
            sb_cols = [c for c in sb_cols if c not in ["home_team", "away_team", "home_score", "away_score"]]
            sb_merge = sb[["date", "home_team", "away_team"] + sb_cols].copy()
            
            if not sb_merge.empty:
                matches = matches.merge(
                    sb_merge,
                    on=["date", "home_team", "away_team"],
                    how="left",
                    suffixes=("", "_sb"),
                )
                logger.info("Enriched matches with StatsBomb event data")

        # Add confederation info
        matches["home_confederation"] = matches["home_team"].map(TEAM_TO_CONFEDERATION)
        matches["away_confederation"] = matches["away_team"].map(TEAM_TO_CONFEDERATION)
        
        # Add FIFA ranking
        matches["home_fifa_rating"] = matches["home_team"].map(FIFA_RANKINGS)
        matches["away_fifa_rating"] = matches["away_team"].map(FIFA_RANKINGS)

        # Compute result from team_a (home) perspective
        if "home_score" in matches.columns and "away_score" in matches.columns:
            matches["result"] = np.where(
                matches["home_score"] > matches["away_score"], 2,  # Win
                np.where(matches["home_score"] == matches["away_score"], 1, 0)  # Draw / Loss
            )
            matches["goal_diff"] = matches["home_score"] - matches["away_score"]

        # Sort by date
        if "date" in matches.columns:
            matches = matches.sort_values("date").reset_index(drop=True)

        return matches

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
                # No match history — use FIFA ranking as baseline
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

        if not self.team_stats_df.empty:
            path = PROCESSED_DATA_DIR / "team_stats.parquet"
            self.team_stats_df.to_parquet(path, index=False)
            logger.info(f"Saved team stats to {path}")

        return self.matches_df, self.team_stats_df
