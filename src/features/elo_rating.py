"""
Custom Elo rating implementation for international football teams.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    FIFA_RANKINGS,
    ELO_K_FACTORS,
    ELO_HOME_ADVANTAGE,
    ELO_NEUTRAL_VENUE_FACTOR,
    ELO_MOV_CAP,
)

logger = logging.getLogger(__name__)


class EloRatingSystem:
    """
    International football Elo rating system.
    
    Key differences from standard Elo:
    1. K-factor varies by match importance (friendly=20, WC=50)
    2. Margin-of-victory multiplier (capped at 3 goals)
    3. Home advantage adjustment (reduced for neutral venues)
    4. No rating floor (ratings can go below starting value)
    
    Initialized from FIFA rankings, then updated with match history.
    """

    def __init__(self, initial_ratings: Optional[dict] = None):
        """
        Initialize the Elo system.
        
        Args:
            initial_ratings: Dict of team -> initial Elo rating.
                            Defaults to FIFA rankings.
        """
        if initial_ratings:
            self.ratings = dict(initial_ratings)
        else:
            self.ratings = dict(FIFA_RANKINGS)
        
        # Track rating history for visualization
        self.history: list[dict] = []
        
        # Default rating for unknown teams
        self.default_rating = 1400.0

    def get_rating(self, team: str) -> float:
        """Get current Elo rating for a team."""
        return self.ratings.get(team, self.default_rating)

    def expected_result(self, rating_a: float, rating_b: float, 
                        home_advantage: float = 0.0) -> float:
        """
        Calculate expected result for team A (0 to 1).
        
        Args:
            rating_a: Elo rating of team A
            rating_b: Elo rating of team B
            home_advantage: Elo points added for home team
            
        Returns:
            Expected score for team A (1.0 = certain win, 0.0 = certain loss)
        """
        diff = rating_a + home_advantage - rating_b
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def _match_importance_k(self, competition: str) -> float:
        """
        Get K-factor based on match importance.
        
        Higher K means ratings change more for important matches.
        """
        comp_lower = str(competition).lower()
        
        if "world cup" in comp_lower or "wc" in comp_lower:
            return ELO_K_FACTORS["world_cup"]
        elif "qualifier" in comp_lower or "qualif" in comp_lower:
            return ELO_K_FACTORS["qualifier"]
        elif any(c in comp_lower for c in ["euro", "copa", "nations", "afcon", "asian cup"]):
            return ELO_K_FACTORS["continental"]
        else:
            return ELO_K_FACTORS["friendly"]

    def _margin_of_victory_multiplier(self, goal_diff: int) -> float:
        """
        Calculate margin-of-victory multiplier.
        
        Bigger wins lead to larger rating changes, but with diminishing
        returns. Capped at ELO_MOV_CAP goals difference.
        
        Formula: ln(|goal_diff| + 1) with cap
        """
        abs_diff = min(abs(goal_diff), ELO_MOV_CAP)
        if abs_diff == 0:
            return 1.0
        return np.log(abs_diff + 1) + 1.0

    def _actual_result(self, goals_a: int, goals_b: int) -> float:
        """Convert match result to numerical score (Win=1, Draw=0.5, Loss=0)."""
        if goals_a > goals_b:
            return 1.0
        elif goals_a == goals_b:
            return 0.5
        else:
            return 0.0

    def update(self, team_a: str, team_b: str, 
               goals_a: int, goals_b: int,
               competition: str = "friendly",
               is_neutral: bool = True,
               home_team: Optional[str] = None) -> tuple[float, float]:
        """
        Update Elo ratings based on a match result.
        
        Args:
            team_a: First team name
            team_b: Second team name
            goals_a: Goals scored by team A
            goals_b: Goals scored by team B
            competition: Competition type for K-factor
            is_neutral: Whether the match is at a neutral venue
            home_team: Which team is playing at home (if not neutral)
            
        Returns:
            Tuple of (new_rating_a, new_rating_b)
        """
        rating_a = self.get_rating(team_a)
        rating_b = self.get_rating(team_b)
        
        # Home advantage
        home_adv = 0.0
        if not is_neutral and home_team:
            if home_team == team_a:
                home_adv = ELO_HOME_ADVANTAGE
            elif home_team == team_b:
                home_adv = -ELO_HOME_ADVANTAGE
        
        # Expected result
        exp_a = self.expected_result(rating_a, rating_b, home_adv)
        
        # Actual result
        act_a = self._actual_result(goals_a, goals_b)
        
        # K-factor and MoV multiplier
        k = self._match_importance_k(competition)
        mov = self._margin_of_victory_multiplier(goals_a - goals_b)
        
        # Update ratings
        delta = k * mov * (act_a - exp_a)
        new_rating_a = rating_a + delta
        new_rating_b = rating_b - delta
        
        self.ratings[team_a] = round(new_rating_a, 2)
        self.ratings[team_b] = round(new_rating_b, 2)
        
        return new_rating_a, new_rating_b

    def process_matches(self, matches: pd.DataFrame) -> pd.DataFrame:
        """
        Process a DataFrame of matches chronologically and update ratings.
        
        Adds 'elo_home' and 'elo_away' columns showing each team's
        rating BEFORE the match (for use as features).
        
        Args:
            matches: DataFrame sorted by date with home_team, away_team,
                    home_score, away_score columns
                    
        Returns:
            DataFrame with added Elo columns
        """
        df = matches.sort_values("date").copy()
        
        elo_home = []
        elo_away = []
        elo_diff = []
        
        for idx, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            
            # Record current ratings BEFORE the match
            r_home = self.get_rating(home)
            r_away = self.get_rating(away)
            elo_home.append(r_home)
            elo_away.append(r_away)
            elo_diff.append(r_home - r_away)
            
            # Update ratings if match is finished
            home_score = row.get("home_score")
            away_score = row.get("away_score")
            
            if pd.notna(home_score) and pd.notna(away_score):
                competition = row.get("competition", "friendly")
                # World Cup matches are at neutral venues
                is_neutral = "world cup" in str(competition).lower() or "wc" in str(competition).lower()
                
                self.update(
                    team_a=home,
                    team_b=away,
                    goals_a=int(home_score),
                    goals_b=int(away_score),
                    competition=competition,
                    is_neutral=is_neutral,
                    home_team=home if not is_neutral else None,
                )
                
                # Record history
                self.history.append({
                    "date": row["date"],
                    "home_team": home,
                    "away_team": away,
                    "elo_home_after": self.get_rating(home),
                    "elo_away_after": self.get_rating(away),
                })
        
        df["elo_home"] = elo_home
        df["elo_away"] = elo_away
        df["elo_diff"] = elo_diff
        
        logger.info(f"Processed Elo for {len(df)} matches. "
                    f"Top 5: {sorted(self.ratings.items(), key=lambda x: -x[1])[:5]}")
        
        return df

    def get_all_ratings(self) -> dict:
        """Get current ratings for all teams, sorted by rating."""
        return dict(sorted(self.ratings.items(), key=lambda x: -x[1]))

    def predict_match(self, team_a: str, team_b: str, 
                      is_neutral: bool = True) -> dict:
        """
        Predict match outcome probabilities from Elo ratings alone.
        
        Uses the logistic Elo model with a draw adjustment.
        
        Args:
            team_a: First team
            team_b: Second team
            is_neutral: Whether at a neutral venue
            
        Returns:
            Dict with win_a, draw, win_b probabilities
        """
        r_a = self.get_rating(team_a)
        r_b = self.get_rating(team_b)
        
        home_adv = 0.0 if is_neutral else ELO_HOME_ADVANTAGE
        
        exp_a = self.expected_result(r_a, r_b, home_adv)
        
        # Approximate W/D/L split using empirical draw rate in international football
        # Draw probability is highest when teams are evenly matched
        draw_base = 0.24  # Base draw rate in international football
        draw_boost = 1.0 - abs(exp_a - 0.5) * 2.0  # Higher when evenly matched
        draw_prob = draw_base * draw_boost
        
        # Distribute remaining probability
        remaining = 1.0 - draw_prob
        win_a = remaining * exp_a
        win_b = remaining * (1.0 - exp_a)
        
        return {
            "win_a": round(win_a, 4),
            "draw": round(draw_prob, 4),
            "win_b": round(win_b, 4),
            "elo_a": r_a,
            "elo_b": r_b,
            "elo_diff": r_a - r_b,
        }
