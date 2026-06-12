"""
Implements frequency-based encoding for high-cardinality categorical.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TEAM_TO_CONFEDERATION, CONFEDERATIONS

logger = logging.getLogger(__name__)


def frequency_encode(series: pd.Series, smooth: float = 1.0) -> pd.Series:
    """
    Encode a categorical series by the frequency of each value.
    
    Args:
        series: Categorical Series to encode
        smooth: Smoothing factor (Laplace smoothing)
        
    Returns:
        Series of frequency-encoded values (0 to 1)
    """
    counts = series.value_counts()
    total = len(series) + smooth * len(counts)
    freq_map = {val: (count + smooth) / total for val, count in counts.items()}
    return series.map(freq_map).fillna(smooth / total)


def target_encode(series: pd.Series, target: pd.Series, 
                  smooth: float = 10.0) -> pd.Series:
    """
    Smoothed target encoding for categorical variables.
    
    Uses the formula:
        encoded = (n * category_mean + smooth * global_mean) / (n + smooth)
    
    Where n is the count of the category. This prevents overfitting for
    rare categories by shrinking toward the global mean.
    
    Args:
        series: Categorical Series to encode
        target: Target variable (numeric)
        smooth: Smoothing factor (higher = more shrinkage to global mean)
        
    Returns:
        Series of target-encoded values
    """
    global_mean = target.mean()
    
    # Group stats
    stats = target.groupby(series).agg(["mean", "count"])
    
    # Smoothed encoding
    smoothed = (stats["count"] * stats["mean"] + smooth * global_mean) / (
        stats["count"] + smooth
    )
    
    return series.map(smoothed).fillna(global_mean)


def compute_encoding_features(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Apply frequency encoding to high-cardinality categorical features
    in the match DataFrame.
    
    Args:
        matches: Match DataFrame
        
    Returns:
        DataFrame with added frequency-encoded columns
    """
    df = matches.copy()
    
    # 1. Confederation frequency
    if "home_confederation" in df.columns:
        df["home_conf_freq"] = frequency_encode(df["home_confederation"])
        df["away_conf_freq"] = frequency_encode(df["away_confederation"])
    else:
        # Add confederation if missing
        df["home_confederation"] = df["home_team"].map(TEAM_TO_CONFEDERATION)
        df["away_confederation"] = df["away_team"].map(TEAM_TO_CONFEDERATION)
        df["home_conf_freq"] = frequency_encode(df["home_confederation"])
        df["away_conf_freq"] = frequency_encode(df["away_confederation"])

    # 2. Team frequency (how often does this team appear in the dataset)
    home_freq = frequency_encode(df["home_team"])
    away_freq = frequency_encode(df["away_team"])
    df["home_team_freq"] = home_freq
    df["away_team_freq"] = away_freq

    # 3. Cross-confederation match indicator
    df["cross_conf_match"] = (
        df["home_confederation"] != df["away_confederation"]
    ).astype(int)

    # 4. Target encoding for team (if target is available)
    if "result" in df.columns:
        # Target encode home team (higher value = team wins more as home)
        df["home_team_target_enc"] = target_encode(
            df["home_team"], df["result"], smooth=15.0
        )
        # For away team, invert the result (0=Win becomes 2, 2=Win becomes 0)
        away_result = 2 - df["result"]
        df["away_team_target_enc"] = target_encode(
            df["away_team"], away_result, smooth=15.0
        )
    
    # 5. Competition stage frequency
    if "stage" in df.columns:
        df["stage_freq"] = frequency_encode(df["stage"])

    # 6. Match importance proxy from stage
    if "stage" in df.columns:
        stage_importance = {
            "GROUP_STAGE": 0.4,
            "LAST_32": 0.6,
            "LAST_16": 0.7,
            "QUARTER_FINALS": 0.8,
            "SEMI_FINALS": 0.9,
            "THIRD_PLACE": 0.85,
            "FINAL": 1.0,
        }
        df["match_importance"] = df["stage"].map(stage_importance).fillna(0.3)

    logger.info(f"Applied frequency encoding to {len(df)} matches")
    return df


def compute_team_level_encodings(team_stats: pd.DataFrame) -> dict:
    """
    Compute team-level frequency encodings for use in prediction.
    
    Args:
        team_stats: Team statistics DataFrame
        
    Returns:
        Dict mapping team -> dict of encoding features
    """
    result = {}
    
    if team_stats.empty:
        return result

    # Confederation frequency across all teams
    conf_counts = {}
    for team in team_stats["team"]:
        conf = TEAM_TO_CONFEDERATION.get(team, "Unknown")
        conf_counts[conf] = conf_counts.get(conf, 0) + 1
    
    total = sum(conf_counts.values())
    conf_freq = {conf: count / total for conf, count in conf_counts.items()}

    for _, row in team_stats.iterrows():
        team = row["team"]
        conf = TEAM_TO_CONFEDERATION.get(team, "Unknown")
        
        result[team] = {
            "conf_freq": round(conf_freq.get(conf, 0.1), 4),
            "team_freq": round(1.0 / len(team_stats), 4),  # Uniform for WC teams
            "conf_size": len(CONFEDERATIONS.get(conf, [])),
        }

    return result
