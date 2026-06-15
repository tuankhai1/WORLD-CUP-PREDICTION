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

        return self._build_matchup_vector_from_features(fa, fb)

    def _build_matchup_vector_from_features(self, fa: dict, fb: dict) -> dict:
        """Build the numeric team_a - team_b vector from two feature snapshots."""
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
    def _default_xg_features() -> dict:
        features = {}
        for w in ROLLING_WINDOWS:
            features[f"rolling_xg_for_{w}"] = 1.25
            features[f"rolling_xg_against_{w}"] = 1.25
            features[f"rolling_xg_diff_{w}"] = 0.0
            features[f"rolling_goals_for_{w}"] = 1.2
            features[f"rolling_goals_against_{w}"] = 1.2
        features["xg_overperformance"] = 0.0
        return features

    @staticmethod
    def _default_pressing_features() -> dict:
        features = {
            "ppda": 12.0,
            "high_press_pct": 30.0,
            "counterpress_rate": 0.35,
            "defensive_line_height": 45.0,
        }
        for w in ROLLING_WINDOWS:
            features[f"rolling_ppda_{w}"] = 12.0
            features[f"rolling_high_press_pct_{w}"] = 30.0
        return features

    @staticmethod
    def _default_form_features() -> dict:
        features = {}
        for w in ROLLING_WINDOWS:
            features[f"recent_form_{w}"] = 1.0
            features[f"clean_sheet_rate_{w}"] = 0.25
        features.update({
            "win_streak": 0,
            "unbeaten_streak": 0,
            "goals_scored_trend": 0.0,
            "days_since_last_match": 30,
            "total_matches_played": 0,
            "overall_win_rate": 0.33,
            "overall_draw_rate": 0.33,
            "avg_goals_scored": 1.2,
            "avg_goals_conceded": 1.2,
        })
        return features

    @staticmethod
    def _default_encoding_features() -> dict:
        return {
            "conf_freq": 0.1,
            "team_freq": 0.02,
            "conf_size": 5,
        }

    @staticmethod
    def _ewm_latest(values: list[float], window: int, default: float) -> float:
        recent = values[-window:]
        if not recent:
            return default
        return round(float(pd.Series(recent).ewm(span=min(window, len(recent)), adjust=False).mean().iloc[-1]), 3)

    @staticmethod
    def _current_streak(results: list[str], unbeaten: bool = False) -> int:
        streak = 0
        for result in reversed(results):
            if result == "W" or (unbeaten and result == "D"):
                streak += 1
            else:
                break
        return streak

    @staticmethod
    def _goals_trend(goals: list[float], window: int = 10) -> float:
        recent = goals[-window:] if len(goals) >= window else goals
        if len(recent) < 2:
            return 0.0
        x = np.arange(len(recent))
        return round(float(np.polyfit(x, np.array(recent, dtype=float), 1)[0]), 4)

    def _form_snapshot_from_state(self, state: dict, as_of_date: pd.Timestamp) -> dict:
        if not state["points"]:
            return self._default_form_features()

        features = {}
        for w in ROLLING_WINDOWS:
            features[f"recent_form_{w}"] = self._ewm_latest(state["points"], w, 1.0)
            features[f"clean_sheet_rate_{w}"] = self._ewm_latest(state["clean_sheets"], w, 0.25)

        total_matches = len(state["points"])
        features["win_streak"] = self._current_streak(state["results"], unbeaten=False)
        features["unbeaten_streak"] = self._current_streak(state["results"], unbeaten=True)
        features["goals_scored_trend"] = self._goals_trend(state["goals_for"])
        features["days_since_last_match"] = (
            max((as_of_date - state["last_date"]).days, 0)
            if state["last_date"] is not None
            else 30
        )
        features["total_matches_played"] = total_matches
        features["overall_win_rate"] = round(state["results"].count("W") / total_matches, 3)
        features["overall_draw_rate"] = round(state["results"].count("D") / total_matches, 3)
        features["avg_goals_scored"] = round(float(np.mean(state["goals_for"])), 2)
        features["avg_goals_conceded"] = round(float(np.mean(state["goals_against"])), 2)
        return features

    @staticmethod
    def _update_form_state(state: dict, goals_for: int, goals_against: int, match_date: pd.Timestamp) -> None:
        if goals_for > goals_against:
            result = "W"
            points = 3
        elif goals_for == goals_against:
            result = "D"
            points = 1
        else:
            result = "L"
            points = 0

        state["points"].append(points)
        state["clean_sheets"].append(1 if goals_against == 0 else 0)
        state["results"].append(result)
        state["goals_for"].append(goals_for)
        state["goals_against"].append(goals_against)
        state["last_date"] = match_date

    def _compute_pre_match_form_snapshots(self, finished_matches: pd.DataFrame) -> dict:
        """Return shifted form features keyed by (team, match_idx)."""
        states: dict[str, dict] = {}
        snapshots = {}

        def state_for(team: str) -> dict:
            if team not in states:
                states[team] = {
                    "points": [],
                    "clean_sheets": [],
                    "results": [],
                    "goals_for": [],
                    "goals_against": [],
                    "last_date": None,
                }
            return states[team]

        for match_idx, row in finished_matches.sort_values("date").iterrows():
            match_date = pd.Timestamp(row["date"])
            home = row["home_team"]
            away = row["away_team"]

            snapshots[(home, match_idx)] = self._form_snapshot_from_state(state_for(home), match_date)
            snapshots[(away, match_idx)] = self._form_snapshot_from_state(state_for(away), match_date)

            hs = int(row["home_score"])
            away_score = int(row["away_score"])
            self._update_form_state(state_for(home), hs, away_score, match_date)
            self._update_form_state(state_for(away), away_score, hs, match_date)

        return snapshots

    def _build_xg_snapshot_lookup(self, finished_matches: pd.DataFrame) -> dict:
        xg_features = compute_rolling_xg_features(finished_matches)
        lookup = {}
        defaults = self._default_xg_features()

        for team, team_df in xg_features.items():
            for _, row in team_df.iterrows():
                values = {}
                for key in defaults:
                    value = row.get(key, defaults[key])
                    values[key] = defaults[key] if pd.isna(value) else float(value)
                lookup[(team, int(row["match_idx"]))] = values

        return lookup

    def _training_team_snapshot(
        self,
        team: str,
        row: pd.Series,
        side: str,
        xg_lookup: dict,
        form_lookup: dict,
        pressing_features: dict,
        encoding_features: dict,
    ) -> dict:
        match_idx = int(row.name)
        features = {
            "elo_rating": float(row[f"elo_{side}"]),
            "form_elo_rating": float(row[f"form_elo_{side}"]),
            "fifa_rating": FIFA_RANKINGS.get(team, 1500.0),
            # Current squad form is not historically available for old matches.
            # Keep it neutral in training; constant columns are removed below.
            "club_form_power": 0.0,
        }

        features.update(self._default_xg_features())
        features.update(xg_lookup.get((team, match_idx), {}))

        features.update(self._default_pressing_features())
        features.update(pressing_features.get(team, {}))

        features.update(self._default_form_features())
        features.update(form_lookup.get((team, match_idx), {}))

        features.update(self._default_encoding_features())
        features.update(encoding_features.get(team, {}))

        return features

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

        # Build training rows from pre-match snapshots. The inference path still
        # uses build_team_features() to produce the latest tournament snapshot.
        matches = matches.copy()
        matches["date"] = pd.to_datetime(matches["date"])
        matches_modern = matches[matches["date"] >= "2010-01-01"].copy()

        logger.info("  Computing shifted Elo snapshots for training...")
        elo_system = EloRatingSystem()
        matches_with_elo = elo_system.process_matches(matches_modern)
        matches_with_elo = matches_with_elo.sort_values("date").reset_index(drop=True)

        finished = matches_with_elo.dropna(subset=["home_score", "away_score"]).copy().reset_index(drop=True)

        logger.info("  Computing shifted rolling xG snapshots for training...")
        xg_lookup = self._build_xg_snapshot_lookup(finished)

        logger.info("  Computing shifted form snapshots for training...")
        form_lookup = self._compute_pre_match_form_snapshots(finished)

        logger.info("  Computing static priors for training...")
        pressing_features = compute_pressing_features(finished, pd.DataFrame())
        encoding_features = compute_team_level_encodings(team_stats)

        # --- Timeline Truncation ---
        # Elo has been running since 2010. We give it time to settle before
        # using the values to train the trees.
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
            
            home_features = self._training_team_snapshot(
                home,
                row,
                side="home",
                xg_lookup=xg_lookup,
                form_lookup=form_lookup,
                pressing_features=pressing_features,
                encoding_features=encoding_features,
            )
            away_features = self._training_team_snapshot(
                away,
                row,
                side="away",
                xg_lookup=xg_lookup,
                form_lookup=form_lookup,
                pressing_features=pressing_features,
                encoding_features=encoding_features,
            )
            vector = self._build_matchup_vector_from_features(home_features, away_features)
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

        constant_cols = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]
        if constant_cols:
            X = X.drop(columns=constant_cols)
            self.feature_columns = list(X.columns)
            logger.info(f"Dropped {len(constant_cols)} constant training features")
        
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

        logger.info("Building latest team feature snapshot for inference...")
        self.build_team_features(matches, team_stats)
        
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
