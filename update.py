"""
Fetches latest match results, re-runs predictions and simulation,.
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "update.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("update")


def daily_update():
    """
    Daily update workflow:
    1. Fetch latest results from football-data.org
    2. Update Elo ratings and features with new data
    3. Re-generate predictions for remaining matches
    4. Re-run Monte Carlo simulation
    5. Regenerate dashboard
    """
    start = time.time()
    logger.info(f"\n{'='*60}")
    logger.info(f"  DAILY UPDATE -- {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"{'='*60}")
    
    try:
        from src.data_ingestion.github_loader import GithubDataLoader
        from src.data_ingestion.data_merger import DataMerger
        from src.features.pipeline import FeaturePipeline
        from src.model.stacking import StackedEnsemble
        from src.model.predict import MatchPredictor
        from src.simulation.simulator import TournamentSimulator
        from src.output.dashboard import DashboardGenerator
        from config import MC_DEFAULT_ITERATIONS
        import pandas as pd
        
        # 1. Fetch latest data
        logger.info("\n>> Fetching latest match results from Github...")
        gh_loader = GithubDataLoader()
        gh_loader.fetch_data()
        
        # Define empty DataFrames for backwards compatibility in the simulation block
        finished = pd.DataFrame()
        upcoming = pd.DataFrame()
        
        # 2. Re-merge data
        logger.info("\n>> Merging data...")
        merger = DataMerger()
        matches_df, team_stats_df = merger.merge()
        
        if matches_df.empty:
            logger.warning("No match data available for update.")
            return
        
        # 3. Rebuild features
        logger.info("\n>> Rebuilding features...")
        pipeline = FeaturePipeline()
        pipeline.build_team_features(matches_df, team_stats_df)
        pipeline.save()
        
        # 4. Load model and predict
        logger.info("\n>> Generating predictions...")
        ensemble = StackedEnsemble.load()
        predictor = MatchPredictor(pipeline=pipeline, ensemble=ensemble)
        
        prob_matrix = predictor.generate_probability_matrix()
        match_preds = predictor.predict_group_matches()
        
        # Lock in finished match results
        simulator = TournamentSimulator()
        if not finished.empty:
            for _, match in finished.iterrows():
                if match.get("home_score") is not None:
                    simulator.structure.lock_result(
                        team_a=match["home_team"],
                        team_b=match["away_team"],
                        score_a=int(match["home_score"]),
                        score_b=int(match["away_score"]),
                        stage=match.get("stage", "GROUP_STAGE"),
                    )
        
        # 5. Simulate
        logger.info("\n>> Running Monte Carlo simulation...")
        simulator.load_predictions(prob_matrix)
        results = simulator.simulate(num_iterations=MC_DEFAULT_ITERATIONS)
        simulator.save_results()
        
        # Log top 5
        logger.info("\n-- Current Top 5:")
        for team, prob in simulator.get_winner_rankings()[:5]:
            logger.info(f"  {team:20s} {prob:6.2f}%")
        
        # 6. Dashboard
        logger.info("\n>> Regenerating dashboard...")
        dashboard = DashboardGenerator()
        path = dashboard.generate(
            simulation_results=results,
            match_predictions=match_preds,
            model_metrics=ensemble.training_metrics,
        )
        
        elapsed = time.time() - start
        logger.info(f"\n[OK] Update complete in {elapsed:.1f}s")
        logger.info(f"Dashboard: {path}")
        
    except Exception as e:
        logger.error(f"[ERROR] Update failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    daily_update()
