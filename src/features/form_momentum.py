"""
Computes recent team form indicators and momentum metrics.
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


def _compute_points(goals_for: int, goals_against: int) -> int:
    """Convert a match result to points (W=3, D=1, L=0)."""
    if goals_for > goals_against:
        return 3
    elif goals_for == goals_against:
        return 1
    return 0


def _compute_streak(results: list[str], target: str) -> int:
    """
    Count the current streak of a specific result type.
    
    Args:
        results: List of results ('W', 'D', 'L') in chronological order
        target: Result type to count ('W' for wins, 'WD' for unbeaten)
        
    Returns:
        Current streak length
    """
    streak = 0
    for r in reversed(results):
        if target == "W" and r == "W":
            streak += 1
        elif target == "WD" and r in ("W", "D"):
            streak += 1
        else:
            break
    return streak


def _compute_goals_trend(goals: list[float], window: int = 10) -> float:
    """
    Compute linear trend of goals scored over recent matches.
    
    Positive slope = improving, negative = declining.
    Uses simple linear regression on the last N goals.
    
    Args:
        goals: List of goals scored per match (chronological)
        window: Number of recent matches to consider
        
    Returns:
        Slope of the linear trend (goals per match)
    """
    recent = goals[-window:] if len(goals) >= window else goals
    if len(recent) < 2:
        return 0.0
    
    x = np.arange(len(recent))
    y = np.array(recent, dtype=float)
    
    # Simple linear regression
    slope = np.polyfit(x, y, 1)[0]
    return round(float(slope), 4)


def compute_form_features(matches: pd.DataFrame, 
                           windows: Optional[list] = None) -> dict:
    """
    Compute form and momentum features for all teams.
    
    Args:
        matches: Match DataFrame sorted by date
        windows: Rolling window sizes (default: [5, 10])
        
    Returns:
        Dict mapping team -> dict of form feature values (latest snapshot)
    """
    if windows is None:
        windows = ROLLING_WINDOWS

    df = matches.sort_values("date").copy()
    teams = set(df["home_team"].dropna()) | set(df["away_team"].dropna())
    
    team_features = {}

    for team in teams:
        home_mask = df["home_team"] == team
        away_mask = df["away_team"] == team
        
        # Build chronological match list
        records = []
        
        for idx in df[home_mask].index:
            hs = df.loc[idx, "home_score"]
            as_ = df.loc[idx, "away_score"]
            if pd.isna(hs) or pd.isna(as_):
                continue
            hs, as_ = int(hs), int(as_)
            result = "W" if hs > as_ else ("D" if hs == as_ else "L")
            records.append({
                "date": df.loc[idx, "date"],
                "goals_for": hs,
                "goals_against": as_,
                "points": _compute_points(hs, as_),
                "result": result,
                "clean_sheet": 1 if as_ == 0 else 0,
            })
        
        for idx in df[away_mask].index:
            hs = df.loc[idx, "home_score"]
            as_ = df.loc[idx, "away_score"]
            if pd.isna(hs) or pd.isna(as_):
                continue
            hs, as_ = int(hs), int(as_)
            result = "W" if as_ > hs else ("D" if hs == as_ else "L")
            records.append({
                "date": df.loc[idx, "date"],
                "goals_for": as_,
                "goals_against": hs,
                "points": _compute_points(as_, hs),
                "result": result,
                "clean_sheet": 1 if hs == 0 else 0,
            })

        if not records:
            # No match history — use neutral defaults
            team_features[team] = {
                f"recent_form_{w}": 1.0 for w in windows
            }
            team_features[team].update({
                "win_streak": 0,
                "unbeaten_streak": 0,
                "goals_scored_trend": 0.0,
                f"clean_sheet_rate_{windows[0]}": 0.25,
                "days_since_last_match": 30,
            })
            continue

        team_df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
        results_list = team_df["result"].tolist()
        goals_list = team_df["goals_for"].tolist()
        
        features = {}
        
        # Rolling form (points per game) for each window using EWMA
        for w in windows:
            recent_points = team_df["points"].tail(w)
            if len(recent_points) > 0:
                ewma_points = recent_points.ewm(span=min(w, len(recent_points)), adjust=False).mean().iloc[-1]
                features[f"recent_form_{w}"] = round(ewma_points, 3)
            else:
                features[f"recent_form_{w}"] = 1.0
                
            recent_cs = team_df["clean_sheet"].tail(w)
            if len(recent_cs) > 0:
                ewma_cs = recent_cs.ewm(span=min(w, len(recent_cs)), adjust=False).mean().iloc[-1]
                features[f"clean_sheet_rate_{w}"] = round(ewma_cs, 3)
            else:
                features[f"clean_sheet_rate_{w}"] = 0.25

        # Streaks
        features["win_streak"] = _compute_streak(results_list, "W")
        features["unbeaten_streak"] = _compute_streak(results_list, "WD")
        
        # Goals trend
        features["goals_scored_trend"] = _compute_goals_trend(goals_list)
        
        # Days since last match
        last_date = team_df["date"].iloc[-1]
        if pd.notna(last_date):
            today = pd.Timestamp.now()
            days = (today - pd.Timestamp(last_date)).days
            features["days_since_last_match"] = max(0, days)
        else:
            features["days_since_last_match"] = 30
        
        # Additional features
        total_matches = len(team_df)
        features["total_matches_played"] = total_matches
        features["overall_win_rate"] = round(
            sum(1 for r in results_list if r == "W") / total_matches, 3
        )
        features["overall_draw_rate"] = round(
            sum(1 for r in results_list if r == "D") / total_matches, 3
        )
        features["avg_goals_scored"] = round(team_df["goals_for"].mean(), 2)
        features["avg_goals_conceded"] = round(team_df["goals_against"].mean(), 2)
        
        team_features[team] = features

    logger.info(f"Computed form features for {len(team_features)} teams")
    return team_features
