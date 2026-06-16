"""
High-level interface for predicting individual matches and generating.
"""

import logging
from typing import Optional
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import ALL_TEAMS, MODEL_DIR, GROUPS
from src.features.pipeline import FeaturePipeline
from src.model.stacking import StackedEnsemble

logger = logging.getLogger(__name__)


class MatchPredictor:
    """
    Prediction interface for match outcomes.
    
    Combines the feature pipeline and stacked ensemble model
    to predict any matchup between two teams.
    """

    def __init__(self, pipeline: Optional[FeaturePipeline] = None,
                 ensemble: Optional[StackedEnsemble] = None):
        """
        Initialize the predictor.
        
        Args:
            pipeline: Fitted feature pipeline (or None to load from disk)
            ensemble: Fitted ensemble model (or None to load from disk)
        """
        self.pipeline = pipeline
        self.ensemble = ensemble
        self._prediction_cache: dict = {}
        self._raw_prediction_cache: dict = {}

    def load(self):
        """Load saved pipeline and model from disk."""
        self.pipeline = FeaturePipeline.load()
        self.ensemble = StackedEnsemble.load()
        logger.info("Predictor loaded from disk")

    def predict_match(self, team_a: str, team_b: str) -> dict:
        """
        Predict the outcome of a match between two teams.
        
        Args:
            team_a: First team (positive perspective)
            team_b: Second team
            
        Returns:
            Dict with:
            - win_prob: P(team_a wins)
            - draw_prob: P(draw)
            - loss_prob: P(team_a loses)
            - expected_goal_diff: Expected goal difference (team_a - team_b)
            - predicted_score: Most likely scoreline
        """
        # Check cache
        cache_key = f"{team_a}_vs_{team_b}"
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]

        if self.pipeline is None or self.ensemble is None:
            self.load()

        raw_ab = self._raw_model_prediction(team_a, team_b)
        raw_ba = self._raw_model_prediction(team_b, team_a)
        if raw_ab is None or raw_ba is None:
            logger.warning(f"Cannot predict {team_a} vs {team_b}: missing features")
            return self._elo_fallback(team_a, team_b)

        win_prob = (raw_ab["win_prob"] + raw_ba["loss_prob"]) / 2
        draw_prob = (raw_ab["draw_prob"] + raw_ba["draw_prob"]) / 2
        loss_prob = (raw_ab["loss_prob"] + raw_ba["win_prob"]) / 2
        total = win_prob + draw_prob + loss_prob
        if total > 0:
            win_prob /= total
            draw_prob /= total
            loss_prob /= total

        xgd = (raw_ab["expected_goal_diff"] - raw_ba["expected_goal_diff"]) / 2
        predicted_score = self._xgd_to_score(xgd, win_prob, draw_prob, loss_prob)

        prediction = {
            "team_a": team_a,
            "team_b": team_b,
            "win_prob": round(win_prob, 4),
            "draw_prob": round(draw_prob, 4),
            "loss_prob": round(loss_prob, 4),
            "expected_goal_diff": round(xgd, 2),
            "predicted_score": predicted_score,
        }

        self._prediction_cache[cache_key] = prediction
        return prediction

    def _raw_model_prediction(self, team_a: str, team_b: str) -> Optional[dict]:
        """Run the model for one ordered matchup without symmetry correction."""
        cache_key = f"{team_a}_vs_{team_b}"
        if cache_key in self._raw_prediction_cache:
            return self._raw_prediction_cache[cache_key]

        if self.pipeline is None or self.ensemble is None:
            self.load()

        X = self.pipeline.predict_matchup(team_a, team_b)
        if X is None:
            return None

        result = self.ensemble.predict_match(X)
        raw = {
            "win_prob": round(result["win_prob"], 4),
            "draw_prob": round(result["draw_prob"], 4),
            "loss_prob": round(result["loss_prob"], 4),
            "expected_goal_diff": round(result["expected_goal_diff"], 4),
        }
        self._raw_prediction_cache[cache_key] = raw
        return raw

    def _elo_fallback(self, team_a: str, team_b: str) -> dict:
        """Elo-based prediction fallback when model isn't available."""
        from src.features.elo_rating import EloRatingSystem
        
        elo = EloRatingSystem()
        result = elo.predict_match(team_a, team_b, is_neutral=True)
        
        return {
            "team_a": team_a,
            "team_b": team_b,
            "win_prob": round(result["win_a"], 4),
            "draw_prob": round(result["draw"], 4),
            "loss_prob": round(result["win_b"], 4),
            "expected_goal_diff": round(result["elo_diff"] / 400, 2),
            "predicted_score": "1-1",
        }

    def _xgd_to_score(self, xgd: float, win_p: float, draw_p: float, 
                       loss_p: float) -> str:
        """
        Convert expected goal differential to a predicted scoreline.
        
        Uses xGD magnitude and W/D/L probabilities to generate the
        most likely exact score.
        """
        if draw_p > max(win_p, loss_p):
            # Most likely a draw
            if abs(xgd) < 0.3:
                return "1-1"
            elif abs(xgd) < 0.8:
                return "1-1"
            else:
                return "2-2"
        
        if win_p > loss_p:
            # Team A likely wins
            if xgd < 0.5:
                return "1-0"
            elif xgd < 1.5:
                return "2-1"
            elif xgd < 2.5:
                return "2-0"
            else:
                return "3-1"
        else:
            # Team B likely wins
            if xgd > -0.5:
                return "0-1"
            elif xgd > -1.5:
                return "1-2"
            elif xgd > -2.5:
                return "0-2"
            else:
                return "1-3"

    def generate_probability_matrix(self) -> dict:
        """
        Generate W/D/L probabilities and xGD for all possible matchups
        between tournament teams.
        
        This matrix is fed to the Monte Carlo simulator.
        
        Returns:
            Dict with structure:
            {
                "team_a|team_b": {
                    "win": P(A wins),
                    "draw": P(draw),
                    "loss": P(A loses),
                    "xgd": expected_goal_diff
                }
            }
        """
        logger.info("Generating probability matrix for all matchups...")
        
        matrix = {}
        teams = ALL_TEAMS
        
        for i, team_a in enumerate(teams):
            for team_b in teams[i+1:]:
                pred = self.predict_match(team_a, team_b)
                
                key_ab = f"{team_a}|{team_b}"
                matrix[key_ab] = {
                    "win": pred["win_prob"],
                    "draw": pred["draw_prob"],
                    "loss": pred["loss_prob"],
                    "xgd": pred["expected_goal_diff"],
                }
                
                # Also store the reverse matchup
                key_ba = f"{team_b}|{team_a}"
                matrix[key_ba] = {
                    "win": pred["loss_prob"],   # Reversed!
                    "draw": pred["draw_prob"],
                    "loss": pred["win_prob"],    # Reversed!
                    "xgd": -pred["expected_goal_diff"],
                }

        logger.info(f"Generated {len(matrix)} matchup predictions")
        return matrix

    def predict_group_matches(self) -> list[dict]:
        """
        Predict all group stage matches.
        
        Returns:
            List of prediction dicts for each group match
        """
        predictions = []
        
        for group_name, teams in GROUPS.items():
            # Each team plays every other team in the group once
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    pred = self.predict_match(teams[i], teams[j])
                    pred["group"] = group_name
                    pred["stage"] = "GROUP_STAGE"
                    predictions.append(pred)

        return predictions

    def save(self, path: Optional[Path] = None):
        """Save the predictor components."""
        if self.pipeline:
            self.pipeline.save()
        if self.ensemble:
            self.ensemble.save()
        
        # Save prediction cache
        if self._prediction_cache:
            cache_path = path or MODEL_DIR / "prediction_cache.joblib"
            joblib.dump(self._prediction_cache, cache_path)