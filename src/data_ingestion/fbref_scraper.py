"""
Scrapes advanced team-level statistics from FBref for international.
"""

import logging
import time
import random
from typing import Optional

import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RAW_DATA_DIR, resolve_team_name

logger = logging.getLogger(__name__)

# User agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

# FBref URL patterns for international team stats
FBREF_URLS = {
    "standard": "https://fbref.com/en/comps/550/stats/FIFA-World-Cup-Stats",
    "shooting": "https://fbref.com/en/comps/550/shooting/FIFA-World-Cup-Stats",
    "passing": "https://fbref.com/en/comps/550/passing/FIFA-World-Cup-Stats",
    "defense": "https://fbref.com/en/comps/550/defense/FIFA-World-Cup-Stats",
    "possession": "https://fbref.com/en/comps/550/possession/FIFA-World-Cup-Stats",
}


class FBrefScraper:
    """
    Scraper for advanced team statistics from FBref.
    
    Features:
    - Polite scraping with delays and user-agent rotation
    - Cloudflare detection and graceful fallback
    - Multi-level header handling for FBref tables
    - Data cleaning and normalization
    
    Note: FBref has increasingly strict anti-scraping measures.
    If scraping fails, the pipeline falls back to derived features
    from football-data.org data only.
    """

    def __init__(self, delay: float = 5.0):
        """
        Initialize the scraper.
        
        Args:
            delay: Seconds to wait between requests (be polite!)
        """
        self.delay = delay
        self._session = None

    @property
    def session(self):
        """Lazy-initialized requests session."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": random.choice(USER_AGENTS),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            })
        return self._session

    def _fetch_table(self, url: str, table_index: int = 0) -> Optional[pd.DataFrame]:
        """
        Fetch and parse an HTML table from FBref.
        
        Args:
            url: FBref URL to scrape
            table_index: Which table to extract (0-indexed)
            
        Returns:
            DataFrame or None if blocked/failed
        """
        try:
            # Rotate user agent
            self.session.headers["User-Agent"] = random.choice(USER_AGENTS)
            
            logger.info(f"Fetching: {url}")
            response = self.session.get(url, timeout=30)
            
            # Check for Cloudflare block
            if response.status_code == 403:
                logger.warning("FBref returned 403 (Cloudflare block). Falling back.")
                return None
            elif response.status_code == 429:
                logger.warning("Rate limited by FBref. Waiting 60s...")
                time.sleep(60)
                return None
            elif response.status_code != 200:
                logger.warning(f"FBref returned {response.status_code}")
                return None

            # Parse tables from HTML
            tables = pd.read_html(response.text)
            
            if table_index >= len(tables):
                logger.warning(f"Table index {table_index} not found (found {len(tables)} tables)")
                return None

            df = tables[table_index]
            
            # Handle multi-level headers (common in FBref)
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten by joining levels
                df.columns = [
                    f"{c[0]}_{c[1]}" if c[0] != c[1] and "Unnamed" not in str(c[0])
                    else str(c[1])
                    for c in df.columns
                ]

            # Polite delay before next request
            time.sleep(self.delay)
            
            return df

        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")
            return None

    def get_standard_stats(self) -> Optional[pd.DataFrame]:
        """Fetch standard team stats (goals, assists, xG, etc.)."""
        df = self._fetch_table(FBREF_URLS["standard"])
        if df is None:
            return None
            
        # Clean and standardize team names
        if "Squad" in df.columns:
            df["team"] = df["Squad"].apply(resolve_team_name)
        
        return df

    def get_shooting_stats(self) -> Optional[pd.DataFrame]:
        """Fetch shooting statistics (shots, SoT, xG per shot, etc.)."""
        df = self._fetch_table(FBREF_URLS["shooting"])
        if df is None:
            return None

        if "Squad" in df.columns:
            df["team"] = df["Squad"].apply(resolve_team_name)
        
        return df

    def get_passing_stats(self) -> Optional[pd.DataFrame]:
        """Fetch passing statistics (completion %, progressive passes, etc.)."""
        df = self._fetch_table(FBREF_URLS["passing"])
        if df is None:
            return None

        if "Squad" in df.columns:
            df["team"] = df["Squad"].apply(resolve_team_name)
        
        return df

    def get_defensive_stats(self) -> Optional[pd.DataFrame]:
        """Fetch defensive statistics (tackles, pressures, blocks, etc.)."""
        df = self._fetch_table(FBREF_URLS["defense"])
        if df is None:
            return None

        if "Squad" in df.columns:
            df["team"] = df["Squad"].apply(resolve_team_name)
        
        return df

    def get_possession_stats(self) -> Optional[pd.DataFrame]:
        """Fetch possession statistics (touches, carries, etc.)."""
        df = self._fetch_table(FBREF_URLS["possession"])
        if df is None:
            return None

        if "Squad" in df.columns:
            df["team"] = df["Squad"].apply(resolve_team_name)
        
        return df

    def build_advanced_stats_dataset(self) -> pd.DataFrame:
        """
        Attempt to build a comprehensive advanced stats dataset from FBref.
        
        If any individual scrape fails, we continue with what we have.
        If all scrapes fail (Cloudflare), returns empty DataFrame and the
        pipeline falls back to derived features only.
        
        Returns:
            DataFrame with team-level advanced metrics, or empty DataFrame.
        """
        logger.info("Attempting to scrape FBref advanced stats...")
        
        datasets = {}
        scrapers = {
            "standard": self.get_standard_stats,
            "shooting": self.get_shooting_stats,
            "passing": self.get_passing_stats,
            "defense": self.get_defensive_stats,
            "possession": self.get_possession_stats,
        }

        for name, scraper_fn in scrapers.items():
            df = scraper_fn()
            if df is not None and not df.empty:
                datasets[name] = df
                logger.info(f"Successfully scraped {name} stats ({len(df)} teams)")
            else:
                logger.warning(f"Failed to scrape {name} stats")

        if not datasets:
            logger.warning(
                "All FBref scrapes failed (likely Cloudflare). "
                "Pipeline will use derived features only."
            )
            return pd.DataFrame()

        # Merge all datasets on team name
        merged = None
        for name, df in datasets.items():
            if "team" not in df.columns:
                continue
                
            # Select relevant columns and add prefix
            cols_to_keep = ["team"] + [
                c for c in df.columns 
                if c != "team" and df[c].dtype in ["float64", "int64", "float32", "int32"]
            ]
            subset = df[cols_to_keep].copy()
            
            # Prefix columns to avoid conflicts
            subset.columns = ["team"] + [
                f"fbref_{name}_{c}" for c in subset.columns if c != "team"
            ]

            if merged is None:
                merged = subset
            else:
                merged = merged.merge(subset, on="team", how="outer")

        if merged is not None and not merged.empty:
            path = RAW_DATA_DIR / "fbref_advanced_stats.parquet"
            merged.to_parquet(path, index=False)
            logger.info(f"Saved FBref stats for {len(merged)} teams to {path}")
            return merged

        return pd.DataFrame()
