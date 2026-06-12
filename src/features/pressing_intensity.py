"""
Computes team-level pressing and defensive intensity metrics.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import ROLLING_WINDOWS

logger = logging.getLogger(__name__)

# Default pressing values for teams without event data
# Based on typical international football averages
DEFAULT_PRESSING = {
    "ppda": 12.0,           # Average PPDA in international football
    "high_press_pct": 30.0, # % of pressures in attacking third
    "counterpress_rate": 0.35,  # Ball recovery rate after turnover
    "def_line_height": 45.0,    # Average defensive line Y-position
}

# Pressing style estimates based on confederation tendencies
# These are rough archetypes — will be overridden by actual data
CONFEDERATION_PRESSING_PRIORS = {
    "UEFA": {"ppda": 10.5, "high_press_pct": 35.0, "def_line_height": 48.0},
    "CONMEBOL": {"ppda": 11.0, "high_press_pct": 32.0, "def_line_height": 46.0},
    "CONCACAF": {"ppda": 13.0, "high_press_pct": 28.0, "def_line_height": 42.0},
    "CAF": {"ppda": 12.5, "high_press_pct": 30.0, "def_line_height": 43.0},
    "AFC": {"ppda": 13.5, "high_press_pct": 27.0, "def_line_height": 41.0},
    "OFC": {"ppda": 14.0, "high_press_pct": 25.0, "def_line_height": 40.0},
}


def compute_pressing_from_events(matches: pd.DataFrame) -> dict:
    """
    Extract pressing features from StatsBomb event data columns
    that were merged into the match DataFrame.
    
    Args:
        matches: Match DataFrame with optional StatsBomb pressure columns
        
    Returns:
        Dict mapping team -> DataFrame of pressing features indexed by date
    """
    df = matches.copy()
    has_sb_data = any(c.startswith("home_ppda") or c.startswith("home_pressures") 
                      for c in df.columns)
    
    if not has_sb_data:
        logger.info("No event-level pressing data found. Using derived estimates.")
        return {}

    teams = set(df["home_team"].dropna()) | set(df["away_team"].dropna())
    team_features = {}

    for team in teams:
        home_mask = df["home_team"] == team
        away_mask = df["away_team"] == team
        
        records = []
        
        for idx in df[home_mask].index:
            records.append({
                "date": df.loc[idx, "date"],
                "ppda": df.loc[idx].get("home_ppda", DEFAULT_PRESSING["ppda"]),
                "high_press_pct": df.loc[idx].get("home_high_press_pct", DEFAULT_PRESSING["high_press_pct"]),
                "pressures": df.loc[idx].get("home_pressures", 0),
                "match_idx": idx,
            })
        
        for idx in df[away_mask].index:
            records.append({
                "date": df.loc[idx, "date"],
                "ppda": df.loc[idx].get("away_ppda", DEFAULT_PRESSING["ppda"]),
                "high_press_pct": df.loc[idx].get("away_high_press_pct", DEFAULT_PRESSING["high_press_pct"]),
                "pressures": df.loc[idx].get("away_pressures", 0),
                "match_idx": idx,
            })

        if records:
            team_df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
            team_features[team] = team_df

    return team_features


def estimate_pressing_from_results(matches: pd.DataFrame, 
                                    team_stats: pd.DataFrame) -> dict:
    """
    Estimate pressing intensity from match results and team stats
    when event-level data is unavailable.
    
    Uses several proxies:
    - Possession correlates inversely with PPDA (high possession → low PPDA)
    - Goal difference as proxy for pressing efficiency
    - FIFA ranking as proxy for tactical sophistication
    
    Args:
        matches: Match DataFrame
        team_stats: Team-level aggregate statistics
        
    Returns:
        Dict mapping team -> dict of estimated pressing metrics
    """
    from config import TEAM_TO_CONFEDERATION, FIFA_RANKINGS
    
    team_pressing = {}

    teams = set(matches["home_team"].dropna()) | set(matches["away_team"].dropna())
    
    for team in teams:
        conf = TEAM_TO_CONFEDERATION.get(team, "Unknown")
        rating = FIFA_RANKINGS.get(team, 1400)
        
        # Base pressing from confederation prior
        base = CONFEDERATION_PRESSING_PRIORS.get(conf, DEFAULT_PRESSING)
        
        # Adjust based on FIFA rating (higher-rated teams tend to press more)
        # Normalize rating to [0, 1] range
        rating_factor = (rating - 1400) / (1900 - 1400)  # 0 for lowest, 1 for highest
        rating_factor = max(0, min(1, rating_factor))
        
        # Higher-rated teams: lower PPDA, higher press %, higher line
        ppda = base["ppda"] - (rating_factor * 4.0)  # Range: base to base-4
        high_press = base["high_press_pct"] + (rating_factor * 15.0)  # Range: base to base+15
        def_line = base.get("def_line_height", 45.0) + (rating_factor * 10.0)
        counterpress = 0.25 + (rating_factor * 0.25)  # Range: 0.25 to 0.50
        
        # Check if team stats have possession data (from FBref)
        if not team_stats.empty and team in team_stats["team"].values:
            ts = team_stats[team_stats["team"] == team].iloc[0]
            # If we have avg goals scored, use as additional signal
            avg_gf = ts.get("avg_goals_scored", 1.2)
            avg_ga = ts.get("avg_goals_conceded", 1.2)
            goal_diff_factor = (avg_gf - avg_ga) / 3.0  # Normalize
            
            ppda -= goal_diff_factor * 1.5
            high_press += goal_diff_factor * 5.0

        team_pressing[team] = {
            "ppda": round(max(5.0, ppda), 2),
            "high_press_pct": round(min(55.0, max(15.0, high_press)), 1),
            "counterpress_rate": round(max(0.15, min(0.55, counterpress)), 3),
            "defensive_line_height": round(max(35.0, min(55.0, def_line)), 1),
        }

    logger.info(f"Estimated pressing features for {len(team_pressing)} teams")
    return team_pressing


def compute_pressing_features(matches: pd.DataFrame, 
                               team_stats: pd.DataFrame,
                               windows: Optional[list] = None) -> dict:
    """
    Compute pressing intensity features for all teams.
    
    Tries event-level data first, falls back to estimation.
    Then computes rolling averages over specified windows.
    
    Args:
        matches: Match DataFrame
        team_stats: Team statistics DataFrame
        windows: Rolling window sizes
        
    Returns:
        Dict mapping team -> dict of pressing feature values (latest)
    """
    if windows is None:
        windows = ROLLING_WINDOWS

    # Try to get event-level pressing data
    event_pressing = compute_pressing_from_events(matches)
    
    if event_pressing:
        # Compute rolling averages from event data
        result = {}
        for team, team_df in event_pressing.items():
            latest = {}
            for w in windows:
                for col in ["ppda", "high_press_pct"]:
                    if col in team_df.columns:
                        rolling_val = team_df[col].rolling(w, min_periods=1).mean().iloc[-1]
                        latest[f"rolling_{col}_{w}"] = round(rolling_val, 2)
            
            # Also include latest values
            latest["ppda"] = round(team_df["ppda"].iloc[-1], 2) if "ppda" in team_df.columns else DEFAULT_PRESSING["ppda"]
            latest["high_press_pct"] = round(team_df["high_press_pct"].iloc[-1], 1) if "high_press_pct" in team_df.columns else DEFAULT_PRESSING["high_press_pct"]
            latest["counterpress_rate"] = DEFAULT_PRESSING["counterpress_rate"]
            latest["defensive_line_height"] = DEFAULT_PRESSING["def_line_height"]
            
            result[team] = latest
        
        logger.info(f"Computed pressing features from events for {len(result)} teams")
        return result
    else:
        # Fall back to estimation
        estimated = estimate_pressing_from_results(matches, team_stats)
        
        # Wrap estimated values with rolling keys for consistency
        result = {}
        for team, vals in estimated.items():
            team_result = dict(vals)
            for w in windows:
                team_result[f"rolling_ppda_{w}"] = vals["ppda"]
                team_result[f"rolling_high_press_pct_{w}"] = vals["high_press_pct"]
            result[team] = team_result
        
        return result
