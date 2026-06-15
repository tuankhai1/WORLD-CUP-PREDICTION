"""
Computes rolling window xG statistics for each team across their.
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


def _estimate_xg_from_goals(goals_scored: float, goals_conceded: float) -> tuple[float, float]:
    """
    Estimate xG from goal counts when event-level data is unavailable.
    
    Uses a regression-to-mean approach: xG = 0.7 * goals + 0.3 * league_avg.
    This accounts for the fact that goals scored is a noisy estimator of
    underlying attacking quality.
    
    Args:
        goals_scored: Actual goals scored in the match
        goals_conceded: Actual goals conceded in the match
        
    Returns:
        Tuple of (estimated_xg_for, estimated_xg_against)
    """
    LEAGUE_AVG_XG = 1.25  # Average xG per team per match in international football
    REGRESSION_FACTOR = 0.7
    
    xg_for = REGRESSION_FACTOR * goals_scored + (1 - REGRESSION_FACTOR) * LEAGUE_AVG_XG
    xg_against = REGRESSION_FACTOR * goals_conceded + (1 - REGRESSION_FACTOR) * LEAGUE_AVG_XG
    
    return round(xg_for, 3), round(xg_against, 3)


def compute_match_xg(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure each match has xG values. If StatsBomb data is available,
    use those values. Otherwise, estimate from goals scored.
    
    Args:
        matches: Match DataFrame with home/away scores and optional xG columns
        
    Returns:
        DataFrame with home_xg, away_xg columns guaranteed to exist
    """
    df = matches.copy()
    
    # Use StatsBomb xG if available
    if "home_xg" not in df.columns:
        df["home_xg"] = np.nan
    if "away_xg" not in df.columns:
        df["away_xg"] = np.nan

    # Fill missing xG with estimates from goals
    mask_home = df["home_xg"].isna()
    
    if mask_home.any():
        for idx in df[mask_home].index:
            hs = df.loc[idx, "home_score"]
            as_ = df.loc[idx, "away_score"]
            if pd.notna(hs) and pd.notna(as_):
                xg_for, xg_against = _estimate_xg_from_goals(hs, as_)
                df.loc[idx, "home_xg"] = xg_for
                df.loc[idx, "away_xg"] = xg_against
            else:
                df.loc[idx, "home_xg"] = 1.25
                df.loc[idx, "away_xg"] = 1.25

    return df


def compute_rolling_xg_features(matches: pd.DataFrame, 
                                 windows: Optional[list] = None) -> dict:
    """
    Compute rolling xG features for all teams across their match history.
    
    For each team, we build a chronological sequence of their matches
    (both home and away), then compute rolling averages of xG created
    and xG conceded over specified windows.
    
    Args:
        matches: Match DataFrame with date, home/away teams, and xG values
        windows: Rolling window sizes (default: [5, 10])
        
    Returns:
        Dict mapping team -> DataFrame of rolling xG features indexed by date
    """
    if windows is None:
        windows = ROLLING_WINDOWS

    # Ensure xG values exist
    df = compute_match_xg(matches)
    df = df.sort_values("date").reset_index(drop=True)
    
    # Get all unique teams
    teams = set(df["home_team"].dropna()) | set(df["away_team"].dropna())
    
    team_features = {}

    for team in teams:
        # Build team's match history (both home and away)
        home_mask = df["home_team"] == team
        away_mask = df["away_team"] == team
        
        team_matches = []
        
        for idx in df[home_mask].index:
            opponent_elo = df.loc[idx, "elo_away"] if "elo_away" in df.columns and pd.notna(df.loc[idx, "elo_away"]) else 1500.0
            adj_factor_for = opponent_elo / 1500.0
            adj_factor_against = 1500.0 / opponent_elo
            
            team_matches.append({
                "date": df.loc[idx, "date"],
                "xg_for": df.loc[idx, "home_xg"] * adj_factor_for,
                "xg_against": df.loc[idx, "away_xg"] * adj_factor_against,
                "goals_for": df.loc[idx, "home_score"] * adj_factor_for,
                "goals_against": df.loc[idx, "away_score"] * adj_factor_against,
                "match_idx": idx,
            })
        
        for idx in df[away_mask].index:
            opponent_elo = df.loc[idx, "elo_home"] if "elo_home" in df.columns and pd.notna(df.loc[idx, "elo_home"]) else 1500.0
            adj_factor_for = opponent_elo / 1500.0
            adj_factor_against = 1500.0 / opponent_elo
            
            team_matches.append({
                "date": df.loc[idx, "date"],
                "xg_for": df.loc[idx, "away_xg"] * adj_factor_for,
                "xg_against": df.loc[idx, "home_xg"] * adj_factor_against,
                "goals_for": df.loc[idx, "away_score"] * adj_factor_for,
                "goals_against": df.loc[idx, "home_score"] * adj_factor_against,
                "match_idx": idx,
            })

        if not team_matches:
            continue

        team_df = pd.DataFrame(team_matches).sort_values("date").reset_index(drop=True)

        # Compute rolling features for each window
        features = {"date": team_df["date"], "match_idx": team_df["match_idx"]}
        
        for w in windows:
            # Rolling xG created (shift to avoid leakage - do not include current match)
            features[f"rolling_xg_for_{w}"] = (
                team_df["xg_for"].shift(1).rolling(w, min_periods=1).mean()
            )
            # Rolling xG conceded
            features[f"rolling_xg_against_{w}"] = (
                team_df["xg_against"].shift(1).rolling(w, min_periods=1).mean()
            )
            # Net xG differential
            features[f"rolling_xg_diff_{w}"] = (
                features[f"rolling_xg_for_{w}"] - features[f"rolling_xg_against_{w}"]
            )
            # Rolling goals scored (actual)
            features[f"rolling_goals_for_{w}"] = (
                team_df["goals_for"].shift(1).rolling(w, min_periods=1).mean()
            )
            # Rolling goals conceded
            features[f"rolling_goals_against_{w}"] = (
                team_df["goals_against"].shift(1).rolling(w, min_periods=1).mean()
            )

        # xG overperformance (finishing quality / luck) - cumulative
        cum_goals = team_df["goals_for"].shift(1).expanding().sum()
        cum_xg = team_df["xg_for"].shift(1).expanding().sum()
        features["xg_overperformance"] = (cum_goals - cum_xg) / (
            team_df.index + 1
        )  # Per-match average

        team_features[team] = pd.DataFrame(features)

    logger.info(f"Computed rolling xG features for {len(team_features)} teams")
    return team_features
