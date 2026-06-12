"""
Loads historical match-level and event-level data from StatsBomb's.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RAW_DATA_DIR, resolve_team_name

logger = logging.getLogger(__name__)


class StatsBombLoader:
    """
    Loads and processes StatsBomb open data for international football matches.
    
    Available free competitions typically include:
    - FIFA World Cup 2018 (competition_id=43, season_id=3)
    - FIFA World Cup 2022 (competition_id=43, season_id=106)
    - Various international tournaments
    
    Extracts:
    - Match-level results and metadata
    - Shot events with xG values
    - Pressure events for pressing metrics
    - Pass events for possession analysis
    """

    # Known competition/season IDs for StatsBomb open data
    COMPETITIONS = {
        "WC2018": {"competition_id": 43, "season_id": 3},
        "WC2022": {"competition_id": 43, "season_id": 106},
    }

    def __init__(self):
        """Initialize the StatsBomb loader."""
        self._sb = None  # Lazy import of statsbombpy

    @property
    def sb(self):
        """Lazy import of statsbombpy to avoid import errors if not installed."""
        if self._sb is None:
            try:
                import warnings
                warnings.filterwarnings("ignore", module="statsbombpy")
                from statsbombpy import sb
                self._sb = sb
            except ImportError:
                logger.error(
                    "statsbombpy not installed. Install with: pip install statsbombpy"
                )
                raise
        return self._sb

    def get_available_competitions(self) -> pd.DataFrame:
        """
        List all available free competitions from StatsBomb.
        
        Returns:
            DataFrame with competition_id, season_id, competition_name, etc.
        """
        try:
            comps = self.sb.competitions()
            return comps
        except Exception as e:
            logger.error(f"Failed to fetch competitions: {e}")
            return pd.DataFrame()

    def load_matches(self, competition_key: str = "WC2022") -> pd.DataFrame:
        """
        Load all matches for a given competition.
        
        Args:
            competition_key: Key from COMPETITIONS dict (e.g., "WC2018", "WC2022")
            
        Returns:
            DataFrame with match metadata
        """
        if competition_key not in self.COMPETITIONS:
            logger.error(f"Unknown competition: {competition_key}")
            return pd.DataFrame()

        comp = self.COMPETITIONS[competition_key]
        try:
            matches = self.sb.matches(
                competition_id=comp["competition_id"],
                season_id=comp["season_id"],
            )
            logger.info(f"Loaded {len(matches)} matches from {competition_key}")
            return matches
        except Exception as e:
            logger.error(f"Failed to load matches for {competition_key}: {e}")
            return pd.DataFrame()

    def load_events(self, match_id: int) -> pd.DataFrame:
        """
        Load all events for a specific match.
        
        Args:
            match_id: StatsBomb match ID
            
        Returns:
            DataFrame with all event data (shots, passes, pressures, etc.)
        """
        try:
            events = self.sb.events(match_id=match_id)
            return events
        except Exception as e:
            logger.error(f"Failed to load events for match {match_id}: {e}")
            return pd.DataFrame()

    def extract_shot_xg(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Extract shot events with xG values from event data.
        
        Args:
            events: Full event DataFrame for a match
            
        Returns:
            DataFrame with columns: team, player, xg, outcome, minute
        """
        if events.empty:
            return pd.DataFrame()

        shots = events[events["type"] == "Shot"].copy()
        if shots.empty:
            return pd.DataFrame()

        records = []
        for _, shot in shots.iterrows():
            xg = shot.get("shot_statsbomb_xg", 0.0)
            outcome = shot.get("shot_outcome", "Unknown")
            team = resolve_team_name(str(shot.get("team", "")))
            
            records.append({
                "team": team,
                "player": shot.get("player", "Unknown"),
                "xg": float(xg) if pd.notna(xg) else 0.0,
                "outcome": outcome,
                "is_goal": outcome == "Goal",
                "minute": shot.get("minute", 0),
            })

        return pd.DataFrame(records)

    def extract_pressures(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Extract pressure events for pressing intensity analysis.
        
        Args:
            events: Full event DataFrame for a match
            
        Returns:
            DataFrame with pressure event details
        """
        if events.empty:
            return pd.DataFrame()

        pressures = events[events["type"] == "Pressure"].copy()
        if pressures.empty:
            return pd.DataFrame()

        records = []
        for _, p in pressures.iterrows():
            location = p.get("location", [0, 0])
            if isinstance(location, list) and len(location) >= 2:
                x, y = location[0], location[1]
            else:
                x, y = 0, 0

            team = resolve_team_name(str(p.get("team", "")))
            records.append({
                "team": team,
                "minute": p.get("minute", 0),
                "x": x,
                "y": y,
                "is_high_press": x > 80,  # In opponent's third (pitch is 120m)
                "duration": p.get("duration", 0),
            })

        return pd.DataFrame(records)

    def extract_passes(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Extract pass events for possession and buildup analysis.
        
        Args:
            events: Full event DataFrame for a match
            
        Returns:
            DataFrame with pass event details
        """
        if events.empty:
            return pd.DataFrame()

        passes = events[events["type"] == "Pass"].copy()
        if passes.empty:
            return pd.DataFrame()

        records = []
        for _, p in passes.iterrows():
            location = p.get("location", [0, 0])
            end_location = p.get("pass_end_location", [0, 0])
            
            if isinstance(location, list) and len(location) >= 2:
                start_x, start_y = location[0], location[1]
            else:
                start_x, start_y = 0, 0
                
            if isinstance(end_location, list) and len(end_location) >= 2:
                end_x, end_y = end_location[0], end_location[1]
            else:
                end_x, end_y = 0, 0

            team = resolve_team_name(str(p.get("team", "")))
            outcome = p.get("pass_outcome", "Complete")  # None means complete
            
            records.append({
                "team": team,
                "minute": p.get("minute", 0),
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
                "is_complete": pd.isna(outcome) or outcome == "Complete",
                "is_progressive": (end_x - start_x) > 10,  # Forward by >10m
                "length": np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2),
            })

        return pd.DataFrame(records)

    def compute_match_stats(self, match_id: int) -> Optional[dict]:
        """
        Compute aggregated match-level statistics from events.
        
        Args:
            match_id: StatsBomb match ID
            
        Returns:
            Dict with team-level xG, pressing, and passing stats
        """
        events = self.load_events(match_id)
        if events.empty:
            return None

        shots_df = self.extract_shot_xg(events)
        pressures_df = self.extract_pressures(events)
        passes_df = self.extract_passes(events)

        teams = events["team"].dropna().unique()
        if len(teams) < 2:
            return None

        stats = {}
        for team in teams:
            team_name = resolve_team_name(str(team))
            
            # xG stats
            team_shots = shots_df[shots_df["team"] == team_name]
            total_xg = team_shots["xg"].sum() if not team_shots.empty else 0.0
            shot_count = len(team_shots)
            goals = team_shots["is_goal"].sum() if not team_shots.empty else 0

            # Pressing stats
            team_pressures = pressures_df[pressures_df["team"] == team_name]
            pressure_count = len(team_pressures)
            high_press_count = team_pressures["is_high_press"].sum() if not team_pressures.empty else 0
            high_press_pct = (high_press_count / pressure_count * 100) if pressure_count > 0 else 0

            # Pass stats
            team_passes = passes_df[passes_df["team"] == team_name]
            total_passes = len(team_passes)
            complete_passes = team_passes["is_complete"].sum() if not team_passes.empty else 0
            progressive_passes = team_passes["is_progressive"].sum() if not team_passes.empty else 0
            pass_accuracy = (complete_passes / total_passes * 100) if total_passes > 0 else 0

            # Opponent passes (for PPDA calculation)
            opp_passes = passes_df[passes_df["team"] != team_name]
            opp_pass_count = len(opp_passes)
            
            # PPDA: opponent passes allowed per defensive action
            ppda = (opp_pass_count / pressure_count) if pressure_count > 0 else 20.0

            stats[team_name] = {
                "xg": round(total_xg, 3),
                "shots": shot_count,
                "goals": goals,
                "xg_per_shot": round(total_xg / shot_count, 3) if shot_count > 0 else 0,
                "pressures": pressure_count,
                "high_press_pct": round(high_press_pct, 1),
                "ppda": round(ppda, 2),
                "passes": total_passes,
                "pass_accuracy": round(pass_accuracy, 1),
                "progressive_passes": int(progressive_passes),
            }

        return stats

    def build_historical_dataset(self) -> pd.DataFrame:
        """
        Build a comprehensive match-level dataset from all available
        StatsBomb open World Cup data.
        
        Returns:
            DataFrame with one row per match, containing aggregated
            stats for both teams.
        """
        all_records = []

        for comp_key in ["WC2018", "WC2022"]:
            logger.info(f"Processing {comp_key}...")
            matches = self.load_matches(comp_key)
            
            if matches.empty:
                logger.warning(f"No matches found for {comp_key}, skipping")
                continue

            for _, match in matches.iterrows():
                match_id = match.get("match_id")
                if pd.isna(match_id):
                    continue

                try:
                    stats = self.compute_match_stats(int(match_id))
                    if stats is None or len(stats) < 2:
                        continue

                    team_names = list(stats.keys())
                    home_team = resolve_team_name(str(match.get("home_team", team_names[0])))
                    away_team = resolve_team_name(str(match.get("away_team", team_names[1])))
                    
                    # Find matching stats (team names might differ slightly)
                    home_stats = stats.get(home_team, stats.get(team_names[0], {}))
                    away_stats = stats.get(away_team, stats.get(team_names[1], {}))

                    record = {
                        "match_id": int(match_id),
                        "competition": comp_key,
                        "date": match.get("match_date", ""),
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_score": match.get("home_score", 0),
                        "away_score": match.get("away_score", 0),
                    }
                    
                    # Add prefixed stats for each team
                    for prefix, team_stats in [("home", home_stats), ("away", away_stats)]:
                        for key, value in team_stats.items():
                            record[f"{prefix}_{key}"] = value

                    all_records.append(record)
                    
                except Exception as e:
                    logger.warning(f"Failed to process match {match_id}: {e}")
                    continue

        df = pd.DataFrame(all_records)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            # Save to disk
            path = RAW_DATA_DIR / "statsbomb_historical.parquet"
            df.to_parquet(path, index=False)
            logger.info(f"Saved {len(df)} matches to {path}")

        return df
