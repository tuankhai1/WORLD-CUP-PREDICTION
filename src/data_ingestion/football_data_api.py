"""
Handles fetching match data, standings, and historical results from the.
"""

import json
import time
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    FOOTBALL_DATA_API_KEY,
    FOOTBALL_DATA_BASE_URL,
    FOOTBALL_DATA_RATE_LIMIT,
    CACHE_DIR,
    RAW_DATA_DIR,
    TEAM_ALIASES,
    resolve_team_name,
)

logger = logging.getLogger(__name__)


class FootballDataClient:
    """
    Client for the football-data.org API v4.
    
    Provides access to:
    - World Cup 2026 matches and standings (competition code: WC)
    - Historical international matches
    - Team information and scores
    
    Features:
    - Automatic rate limiting (10 req/min on free tier)
    - Disk-based response caching with configurable TTL
    - Retry logic with exponential backoff
    """

    # Competition codes
    WORLD_CUP = "WC"
    
    # Historical seasons available
    WC_SEASONS = {
        2026: 2146,  # Current tournament
        2022: 2000,
        2018: 1903,
    }

    def __init__(self, api_key: Optional[str] = None, cache_ttl_hours: int = 1):
        """
        Initialize the API client.
        
        Args:
            api_key: football-data.org API key. Falls back to env var.
            cache_ttl_hours: How long cached responses remain valid (hours).
        """
        self.api_key = api_key or FOOTBALL_DATA_API_KEY
        self.base_url = FOOTBALL_DATA_BASE_URL
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.cache_dir = CACHE_DIR / "football_data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting
        self._request_times: list[float] = []
        self._rate_limit = FOOTBALL_DATA_RATE_LIMIT

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            "X-Auth-Token": self.api_key,
            "Accept": "application/json",
        })

        if not self.api_key:
            logger.warning(
                "No API key set. Get a free key at https://www.football-data.org/client/register "
                "and set FOOTBALL_DATA_API_KEY environment variable."
            )

    def _rate_limit_wait(self):
        """Enforce rate limiting: max 10 requests per minute."""
        now = time.time()
        # Remove timestamps older than 60 seconds
        self._request_times = [t for t in self._request_times if now - t < 60]
        
        if len(self._request_times) >= self._rate_limit:
            wait_time = 60 - (now - self._request_times[0]) + 0.5
            if wait_time > 0:
                logger.info(f"Rate limit reached. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
        
        self._request_times.append(time.time())

    def _cache_key(self, endpoint: str, params: dict) -> str:
        """Generate a unique cache key from endpoint and parameters."""
        raw = f"{endpoint}|{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_cached(self, cache_key: str) -> Optional[dict]:
        """Retrieve a cached response if it exists and is not expired."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            cached_at = datetime.fromisoformat(data.get("_cached_at", "2000-01-01"))
            if datetime.now() - cached_at < self.cache_ttl:
                logger.debug(f"Cache hit: {cache_key}")
                return data.get("_response")
        return None

    def _set_cache(self, cache_key: str, response: dict):
        """Store a response in the disk cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        data = {
            "_cached_at": datetime.now().isoformat(),
            "_response": response,
        }
        cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _request(self, endpoint: str, params: Optional[dict] = None, 
                 max_retries: int = 3) -> Optional[dict]:
        """
        Make an API request with caching, rate limiting, and retry logic.
        
        Args:
            endpoint: API endpoint path (e.g., "/competitions/WC/matches")
            params: Query parameters
            max_retries: Maximum number of retry attempts
            
        Returns:
            JSON response as dict, or None on failure
        """
        params = params or {}
        cache_key = self._cache_key(endpoint, params)
        
        # Check cache first
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        url = f"{self.base_url}{endpoint}"

        for attempt in range(max_retries):
            self._rate_limit_wait()
            
            try:
                response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    self._set_cache(cache_key, data)
                    return data
                elif response.status_code == 429:
                    wait = int(response.headers.get("X-RequestCounter-Reset", 60))
                    logger.warning(f"Rate limited (429). Waiting {wait}s...")
                    time.sleep(wait)
                elif response.status_code == 403:
                    logger.error("API key invalid or missing. Check FOOTBALL_DATA_API_KEY.")
                    return None
                else:
                    logger.warning(
                        f"API error {response.status_code}: {response.text[:200]}"
                    )
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)

        logger.error(f"Failed after {max_retries} attempts: {endpoint}")
        return None

    
    def get_matches(self, season: int = 2026, 
                    status: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch all matches for a World Cup season.
        
        Args:
            season: World Cup year (2018, 2022, 2026)
            status: Filter by status (SCHEDULED, LIVE, IN_PLAY, PAUSED, 
                    FINISHED, POSTPONED, SUSPENDED, CANCELLED)
        
        Returns:
            DataFrame with columns: date, home_team, away_team, 
            home_score, away_score, stage, status, matchday
        """
        params = {}
        if status:
            params["status"] = status
        # Use season filter
        params["season"] = season

        data = self._request(f"/competitions/{self.WORLD_CUP}/matches", params)
        if not data or "matches" not in data:
            logger.warning(f"No match data returned for season {season}")
            return pd.DataFrame()

        records = []
        for match in data["matches"]:
            home = match.get("homeTeam", {}).get("name", "Unknown")
            away = match.get("awayTeam", {}).get("name", "Unknown")
            score = match.get("score", {})
            full_time = score.get("fullTime", {})

            records.append({
                "date": match.get("utcDate", "")[:10],
                "home_team": resolve_team_name(home),
                "away_team": resolve_team_name(away),
                "home_score": full_time.get("home"),
                "away_score": full_time.get("away"),
                "stage": match.get("stage", ""),
                "group": match.get("group", ""),
                "status": match.get("status", ""),
                "matchday": match.get("matchday"),
                "competition": "World Cup",
                "season": season,
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        
        logger.info(f"Fetched {len(df)} matches for WC {season}")
        return df

    def get_standings(self, season: int = 2026) -> pd.DataFrame:
        """
        Fetch current group standings.
        
        Returns:
            DataFrame with columns: group, position, team, played,
            won, drawn, lost, goals_for, goals_against, goal_diff, points
        """
        data = self._request(
            f"/competitions/{self.WORLD_CUP}/standings",
            {"season": season}
        )
        if not data or "standings" not in data:
            return pd.DataFrame()

        records = []
        for standing in data["standings"]:
            group = standing.get("group", "")
            for entry in standing.get("table", []):
                team_name = entry.get("team", {}).get("name", "Unknown")
                records.append({
                    "group": group,
                    "position": entry.get("position"),
                    "team": resolve_team_name(team_name),
                    "played": entry.get("playedGames", 0),
                    "won": entry.get("won", 0),
                    "drawn": entry.get("draw", 0),
                    "lost": entry.get("lost", 0),
                    "goals_for": entry.get("goalsFor", 0),
                    "goals_against": entry.get("goalsAgainst", 0),
                    "goal_diff": entry.get("goalDifference", 0),
                    "points": entry.get("points", 0),
                })

        return pd.DataFrame(records)

    def get_historical_matches(self) -> pd.DataFrame:
        """
        Fetch match data from previous World Cups (2018, 2022) for training.
        
        Returns:
            Combined DataFrame of all historical WC matches.
        """
        all_matches = []
        for year in [2018, 2022]:
            df = self.get_matches(season=year)
            if not df.empty:
                all_matches.append(df)
                logger.info(f"Loaded {len(df)} matches from WC {year}")

        if all_matches:
            return pd.concat(all_matches, ignore_index=True)
        return pd.DataFrame()

    def get_current_wc_matches(self) -> pd.DataFrame:
        """
        Fetch current 2026 World Cup matches.
        Separates finished and upcoming matches.
        
        Returns:
            DataFrame with all 2026 WC matches
        """
        return self.get_matches(season=2026)

    def get_finished_matches(self) -> pd.DataFrame:
        """Get only completed 2026 WC matches."""
        df = self.get_current_wc_matches()
        if df.empty:
            return df
        return df[df["status"] == "FINISHED"].reset_index(drop=True)

    def get_upcoming_matches(self) -> pd.DataFrame:
        """Get only upcoming/scheduled 2026 WC matches."""
        df = self.get_current_wc_matches()
        if df.empty:
            return df
        return df[df["status"] == "SCHEDULED"].reset_index(drop=True)

    def save_raw_data(self):
        """Fetch and save all available data to disk."""
        # Historical matches
        hist = self.get_historical_matches()
        if not hist.empty:
            path = RAW_DATA_DIR / "historical_wc_matches.parquet"
            hist.to_parquet(path, index=False)
            logger.info(f"Saved historical matches to {path}")

        # Current tournament
        current = self.get_current_wc_matches()
        if not current.empty:
            path = RAW_DATA_DIR / "wc2026_matches.parquet"
            current.to_parquet(path, index=False)
            logger.info(f"Saved 2026 WC matches to {path}")

        # Standings
        standings = self.get_standings()
        if not standings.empty:
            path = RAW_DATA_DIR / "wc2026_standings.parquet"
            standings.to_parquet(path, index=False)
            logger.info(f"Saved standings to {path}")
