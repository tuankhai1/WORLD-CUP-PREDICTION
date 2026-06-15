"""
Orchestrates all feature modules into a single pipeline that transforms.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    ALL_TEAMS,
    FIFA_RANKINGS,
    TEAM_TO_CONFEDERATION,
    ROLLING_WINDOWS,
    TARGET_WDL,
    TARGET_GD,
    RANDOM_SEED,
    RECENCY_HALF_LIFE_DAYS,
    TRAINING_CUTOFF_DATE,
    RECENCY_MIN_WEIGHT,
    MATCH_IMPORTANCE_WEIGHTS,
)
from config import RAW_DATA_DIR
from src.features.rolling_xg import compute_rolling_xg_features, compute_match_xg
from src.features.pressing_intensity import compute_pressing_features
from src.features.elo_rating import EloRatingSystem
from src.features.form_momentum import compute_form_features
from src.features.encoding import compute_team_level_encodings

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    End-to-end feature engineering pipeline for match prediction.
    
    Transforms raw match data into feature vectors suitable for the
    stacked ensemble model. Features are computed for each team, then
    combined into a matchup vector as the difference (team_A - team_B).
    
    This symmetry-aware approach ensures the model treats 
    predict(A, B) and predict(B, A) consistently.
    """

    def __init__(self):
        """Initialize the pipeline."""
        self.elo_system = EloRatingSystem()
        self.scaler = StandardScaler()
        self.team_features: dict = {}  # team -> {feature_name: value}
        self.feature_columns: list[str] = []

    def build_team_features(self, matches: pd.DataFrame, 
                             team_stats: pd.DataFrame) -> dict:
        """
        Compute all features for every team.
        
        Args:
            matches: Unified match DataFrame (sorted by date)
            team_stats: Team-level statistics DataFrame
            
        Returns:
            Dict mapping team -> dict of all features
        """
        logger.info("Building team features...")
        
        # --- Elo Reset (Modern Era) ---
        # Instead of calculating from 1872, we force the Elo system to start at 2010.
        # This completely erases "historical legacy" from past decades.
        matches["date"] = pd.to_datetime(matches["date"])
        matches_modern = matches[matches["date"] >= "2010-01-01"].copy()
        
        # 1. Elo ratings - process all matches chronologically
        logger.info("  Computing Elo ratings (Modern Era Reset 2010+)...")
        matches_with_elo = self.elo_system.process_matches(matches_modern)
        elo_ratings = self.elo_system.get_all_ratings()
        form_elo_ratings = self.elo_system.get_all_form_ratings()
        finished_with_elo = matches_with_elo.dropna(subset=["home_score", "away_score"]).copy()

        # 2. Rolling xG features
        logger.info("  Computing rolling xG features...")
        xg_features = compute_rolling_xg_features(finished_with_elo)

        # 3. Pressing intensity features
        logger.info("  Computing pressing features...")
        pressing_features = compute_pressing_features(finished_with_elo, team_stats)

        # 4. Form and momentum features
        logger.info("  Computing form features...")
        form_features = compute_form_features(finished_with_elo)

        # 5. Encoding features
        logger.info("  Computing encoding features...")
        encoding_features = compute_team_level_encodings(team_stats)

        # Merge all features for each team
        all_teams = set(ALL_TEAMS) | set(elo_ratings.keys())
        
        squad_ratings_path = RAW_DATA_DIR.parent / "processed" / "squad_ratings.parquet"
        club_form_power_map = {}
        if squad_ratings_path.exists():
            squad_df = pd.read_parquet(squad_ratings_path)
            if 'club_form_power' in squad_df.columns:
                club_form_power_map = dict(zip(squad_df['team'], squad_df['club_form_power']))
            logger.info("  Loaded Club Form Ratings for feature matrix.")
        
        for team in all_teams:
            features = {}
            
            # Elo & FIFA Ranking & Squad Power
            features["elo_rating"] = elo_ratings.get(team, 1500.0)
            features["form_elo_rating"] = form_elo_ratings.get(team, 1500.0)
            
            # User requested re-introduction of FIFA ratings as a minor feature
            features["fifa_rating"] = FIFA_RANKINGS.get(team, 1500.0)
            
            features["club_form_power"] = club_form_power_map.get(team, 0.0)
            
            # Rolling xG - get the latest values
            if team in xg_features:
                team_xg = xg_features[team]
                if not team_xg.empty:
                    latest = team_xg.iloc[-1]
                    for col in team_xg.columns:
                        if col not in ["date", "match_idx"]:
                            features[col] = latest[col] if pd.notna(latest[col]) else 0.0
            
            # Fill missing xG features with defaults
            for w in ROLLING_WINDOWS:
                features.setdefault(f"rolling_xg_for_{w}", 1.25)
                features.setdefault(f"rolling_xg_against_{w}", 1.25)
                features.setdefault(f"rolling_xg_diff_{w}", 0.0)
                features.setdefault(f"rolling_goals_for_{w}", 1.2)
                features.setdefault(f"rolling_goals_against_{w}", 1.2)
            features.setdefault("xg_overperformance", 0.0)
            
            # Pressing
            if team in pressing_features:
                for k, v in pressing_features[team].items():
                    features[k] = v
            else:
                features["ppda"] = 12.0
                features["high_press_pct"] = 30.0
                features["counterpress_rate"] = 0.35
                features["defensive_line_height"] = 45.0
                for w in ROLLING_WINDOWS:
                    features[f"rolling_ppda_{w}"] = 12.0
                    features[f"rolling_high_press_pct_{w}"] = 30.0
            
            # Form
            if team in form_features:
                for k, v in form_features[team].items():
                    features[k] = v
            else:
                for w in ROLLING_WINDOWS:
                    features[f"recent_form_{w}"] = 1.0
                    features[f"clean_sheet_rate_{w}"] = 0.25
                features["win_streak"] = 0
                features["unbeaten_streak"] = 0
                features["goals_scored_trend"] = 0.0
                features["days_since_last_match"] = 30
                features["total_matches_played"] = 0
                features["overall_win_rate"] = 0.33
                features["overall_draw_rate"] = 0.33
                features["avg_goals_scored"] = 1.2
                features["avg_goals_conceded"] = 1.2
            
            # Encoding
            if team in encoding_features:
                for k, v in encoding_features[team].items():
                    features[k] = v
            else:
                features["conf_freq"] = 0.1
                features["team_freq"] = 0.02
                features["conf_size"] = 5

            self.team_features[team] = features

        logger.info(f"Built features for {len(self.team_features)} teams, "
                    f"{len(next(iter(self.team_features.values())))} features each")
        
        return self.team_features

    def build_matchup_vector(self, team_a: str, team_b: str) -> dict:
        """
        Build a feature vector for a specific matchup.
        
        The vector is constructed as the DIFFERENCE (team_a - team_b)
        for all numerical features. This ensures symmetry: swapping
        teams simply negates the vector.
        
        Args:
            team_a: First team (positive direction)
            team_b: Second team
            
        Returns:
            Dict of feature differences
        """
        fa = self.team_features.get(team_a, {})
        fb = self.team_features.get(team_b, {})
        
        if not fa or not fb:
            logger.warning(f"Missing features for {team_a} or {team_b}")
            return {}

        vector = {}
        for key in fa:
            val_a = fa[key]
            val_b = fb.get(key, 0)
            
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                vector[f"diff_{key}"] = val_a - val_b
                # Also include absolute values for both teams
                vector[f"team_a_{key}"] = val_a
                vector[f"team_b_{key}"] = val_b

        # Head-to-head features (can't be computed as difference)
        vector["elo_diff"] = fa.get("elo_rating", 1500) - fb.get("elo_rating", 1500)
        vector["form_elo_diff"] = fa.get("form_elo_rating", 1500) - fb.get("form_elo_rating", 1500)
        
        # Re-added FIFA rating difference
        vector["fifa_rating_diff"] = fa.get("fifa_rating", 1500.0) - fb.get("fifa_rating", 1500.0)
        
        vector["club_form_power_diff"] = fa.get("club_form_power", 0.0) - fb.get("club_form_power", 0.0)
        
        return vector

    @staticmethod
    def _match_importance_weight(competition: str) -> float:
        """Map competition labels to training sample-weight multipliers."""
        comp = str(competition).lower()
        if "world cup" in comp and ("qual" not in comp and "qualification" not in comp):
            return MATCH_IMPORTANCE_WEIGHTS["world_cup"]
        if "qual" in comp:
            return MATCH_IMPORTANCE_WEIGHTS["qualifier"]
        if any(token in comp for token in ["euro", "copa", "nations", "afcon", "asian cup", "gold cup"]):
            return MATCH_IMPORTANCE_WEIGHTS["continental"]
        if "friendly" in comp:
            return MATCH_IMPORTANCE_WEIGHTS["friendly"]
        return MATCH_IMPORTANCE_WEIGHTS["default"]

    def build_training_matrix(self, matches: pd.DataFrame, 
                                team_stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Build the full training feature matrix from match history.
        
        Args:
            matches: Match DataFrame with results
            team_stats: Team statistics DataFrame
            
        Returns:
            Tuple of (X features, y_wdl target, y_gd target, sample_weights)
        """
        logger.info("Building training feature matrix...")
        
        # Build team features
        self.build_team_features(matches, team_stats)
        
        # Build matchup vectors for each match
        finished = matches.dropna(subset=["home_score", "away_score"]).copy()
        
        # --- Timeline Truncation ---
        # Elo has been running since 2010. We give it 5 years to "stretch out" and settle
        # into a stable distribution before we start using the values to train the trees.
        # This prevents absolute scale distortion in the tree models.
        finished["date"] = pd.to_datetime(finished["date"])
        finished = finished[finished["date"] >= TRAINING_CUTOFF_DATE].copy()
        logger.info(
            f"Truncating training data to current-cycle ML era "
            f"({TRAINING_CUTOFF_DATE}+): {len(finished)} matches."
        )
        
        records = []
        targets_wdl = []
        targets_gd = []
        weights = []
        latest_match_date = finished["date"].max()
        
        for _, row in finished.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            
            vector = self.build_matchup_vector(home, away)
            if not vector:
                continue
            
            records.append(vector)
            
            # Target: result from home team's perspective
            hs, as_ = int(row["home_score"]), int(row["away_score"])
            if hs > as_:
                targets_wdl.append(2)  # Win
            elif hs == as_:
                targets_wdl.append(1)  # Draw
            else:
                targets_wdl.append(0)  # Loss
            
            targets_gd.append(hs - as_)
            
            # Heavily favor recent performance with exponential sample weighting.
            # Equation: weight = max(min_weight, 0.5 ** (age_days / half_life))
            #                  * match_importance_multiplier
            age_days = max((latest_match_date - row["date"]).days, 0)
            recency_weight = max(
                RECENCY_MIN_WEIGHT,
                0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS),
            )
            importance_weight = self._match_importance_weight(row.get("competition", ""))
            weight = recency_weight * importance_weight
            weights.append(weight)

        X = pd.DataFrame(records)
        y_wdl = pd.Series(targets_wdl, name=TARGET_WDL)
        y_gd = pd.Series(targets_gd, name=TARGET_GD, dtype=float)
        sample_weights = pd.Series(weights, name="sample_weight", dtype=float)
        
        # Store feature columns
        self.feature_columns = list(X.columns)
        
        # Handle any NaN/inf
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        logger.info(f"Training matrix: {X.shape[0]} samples x {X.shape[1]} features")
        
        # Save
        X.to_parquet(PROCESSED_DATA_DIR / "feature_matrix.parquet", index=False)
        pd.DataFrame({"wdl": y_wdl, "gd": y_gd}).to_parquet(
            PROCESSED_DATA_DIR / "targets.parquet", index=False
        )
        
        # Save feature columns list
        pd.Series(self.feature_columns).to_csv(
            MODEL_DIR / "feature_columns.csv", index=False
        )
        
        return X, y_wdl, y_gd, sample_weights

    def predict_matchup(self, team_a: str, team_b: str) -> Optional[pd.DataFrame]:
        """
        Build feature vector for a new matchup (for prediction).
        
        Args:
            team_a: First team
            team_b: Second team
            
        Returns:
            Single-row DataFrame with features, or None if features unavailable
        """
        if not self.team_features:
            logger.error("Team features not built yet. Run build_team_features first.")
            return None

        vector = self.build_matchup_vector(team_a, team_b)
        if not vector:
            return None

        df = pd.DataFrame([vector])
        
        # Ensure column order matches training
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[self.feature_columns]
        
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        return df

    def save(self, path: Optional[Path] = None):
        """Save the pipeline state."""
        path = path or MODEL_DIR / "feature_pipeline.joblib"
        state = {
            "team_features": self.team_features,
            "feature_columns": self.feature_columns,
            "elo_ratings": self.elo_system.get_all_ratings(),
            "form_elo_ratings": self.elo_system.get_all_form_ratings(),
            "scaler": self.scaler,
        }
        joblib.dump(state, path)
        logger.info(f"Pipeline saved to {path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "FeaturePipeline":
        """Load a saved pipeline."""
        path = path or MODEL_DIR / "feature_pipeline.joblib"
        state = joblib.load(path)
        
        pipeline = cls()
        pipeline.team_features = state["team_features"]
        pipeline.feature_columns = state["feature_columns"]
        pipeline.scaler = state["scaler"]
        pipeline.elo_system.ratings = state["elo_ratings"]
        pipeline.elo_system.form_ratings = state.get("form_elo_ratings", state["elo_ratings"])
        
        logger.info(f"Pipeline loaded from {path}")
        return pipeline
