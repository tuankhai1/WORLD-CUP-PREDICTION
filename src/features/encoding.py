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
