"""
Fetches the martj42/international_results dataset from GitHub.
"""

import logging
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RAW_DATA_DIR, TOURNAMENT_YEAR, resolve_team_name

logger = logging.getLogger(__name__)

class GithubDataLoader:
    def __init__(self):
        self.url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

    def fetch_data(self) -> pd.DataFrame:
        logger.info("Fetching historical match data from GitHub (martj42/international_results)...")
        try:
            df = pd.read_csv(self.url)
            
            # Filter to modern matches to ensure relevance (post 2015)
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] >= "2015-01-01"].copy()
            
            # Standardize team names
            df["home_team"] = df["home_team"].apply(resolve_team_name)
            df["away_team"] = df["away_team"].apply(resolve_team_name)
            
            # Standardize schema to match the pipeline
            df = df.rename(columns={
                "home_score": "home_score",
                "away_score": "away_score",
                "tournament": "competition"
            })
            df["status"] = "FINISHED"
            
            # Keep only relevant columns
            cols = ["date", "home_team", "away_team", "home_score", "away_score", "competition", "status"]
            df = df[[c for c in cols if c in df.columns]]

            competition = df["competition"].fillna("").astype(str).str.lower()
            wc2026_mask = (
                df["date"].dt.year.eq(TOURNAMENT_YEAR)
                & competition.str.contains("world cup", na=False)
                & ~competition.str.contains("qual", na=False)
            )
            wc2026_results = df[wc2026_mask].copy()
            historical = df[~wc2026_mask].copy()
            
            path = RAW_DATA_DIR / "github_historical.parquet"
            historical.to_parquet(path, index=False)
            logger.info(f"Saved {len(historical)} modern historical matches to {path}")

            wc_path = RAW_DATA_DIR / "github_wc2026_results.parquet"
            wc2026_results.to_parquet(wc_path, index=False)
            logger.info(f"Saved {len(wc2026_results)} WC 2026 current rows to {wc_path}")
            
            return historical
        except Exception as e:
            logger.error(f"Failed to fetch GitHub dataset: {e}")
            return pd.DataFrame()
