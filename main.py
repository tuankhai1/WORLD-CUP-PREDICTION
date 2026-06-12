"""
Main entry point for the World Cup prediction pipeline.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    GROUPS,
    MC_DEFAULT_ITERATIONS,
    MC_DEFAULT_SEED,
    OUTPUT_DIR,
    MODEL_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "pipeline.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


def run_full_pipeline(test_mode: bool = False, mode: str = "full"):
    """Run the complete pipeline."""
    total_start = time.time()
    
    logger.info("=" * 70)
    logger.info("  2026 FIFA WORLD CUP PREDICTION — FULL PIPELINE")
    logger.info("=" * 70)
    
    # Data Ingestion
    
    # GitHub
    try:
        from src.data_ingestion.github_loader import GithubDataLoader
        gh_loader = GithubDataLoader()
        gh_loader.fetch_data()
    except Exception as e:
        logger.warning(f"GitHub ingestion failed: {e}")
    
    # Data Merging
    
    from src.data_ingestion.data_merger import DataMerger
    
    merger = DataMerger()
    matches_df, team_stats_df = merger.merge()
    
    if matches_df.empty:
        logger.error("No match data available after merge! Exiting.")
        sys.exit(1)
    
    logger.info(f"Total matches: {len(matches_df)}")
    logger.info(f"Teams with stats: {len(team_stats_df)}")
    
    # Process Player Stats for Squad Power Ratings
    try:
        from src.data_ingestion.player_stats_loader import PlayerStatsLoader
        player_loader = PlayerStatsLoader()
        player_loader.load_and_aggregate()
    except Exception as e:
        logger.warning(f"Player stats loading failed: {e}")
    
    # Feature Engineering
    
    from src.features.pipeline import FeaturePipeline
    
    pipeline = FeaturePipeline()
    X, y_wdl, y_gd, sample_weights = pipeline.build_training_matrix(matches_df, team_stats_df)
    
    logger.info(f"Feature matrix: {X.shape}")
    
    # Hyperparameter Tuning
    
    from src.model.optuna_tuning import run_tuning
    
    if test_mode:
        n_trials = 2
    elif mode == "quick":
        n_trials = 20
    elif mode == "medium":
        n_trials = 50
    else:
        n_trials = 100
        
    best_params = run_tuning(X, y_wdl, y_gd, n_trials=n_trials)
    
    logger.info("Best parameters found:")
    for model_name, params in best_params.items():
        logger.info(f"  {model_name}: {params}")
    
    # Train Stacked Ensemble
    
    from src.model.stacking import StackedEnsemble
    
    ensemble = StackedEnsemble(base_params=best_params)
    ensemble.fit(X, y_wdl, y_gd, sample_weights=sample_weights)
    ensemble.save()
    pipeline.save()
    
    # Generate Predictions
    
    from src.model.predict import MatchPredictor
    
    predictor = MatchPredictor(pipeline=pipeline, ensemble=ensemble)
    prob_matrix = predictor.generate_probability_matrix()
    match_predictions = predictor.predict_group_matches()
    
    logger.info(f"Generated {len(prob_matrix)} matchup predictions")
    logger.info(f"Group match predictions: {len(match_predictions)}")
    
    logger.info("\n-- Key Match Predictions:")
    for pred in match_predictions[:6]:
        logger.info(
            f"  {pred['team_a']} vs {pred['team_b']}: "
            f"W={pred['win_prob']:.0%} D={pred['draw_prob']:.0%} L={pred['loss_prob']:.0%} "
            f"(Score: {pred['predicted_score']})"
        )
    
    # Monte Carlo Simulation
    
    from src.simulation.simulator import TournamentSimulator
    
    simulator = TournamentSimulator()
    simulator.load_predictions(prob_matrix)
    
    iters = 10_000 if test_mode else MC_DEFAULT_ITERATIONS
    results = simulator.simulate(num_iterations=iters, seed=MC_DEFAULT_SEED)
    simulator.save_results()
    
    logger.info("\n-- Tournament Winner Probabilities (Top 10):")
    for team, prob in simulator.get_winner_rankings()[:10]:
        logger.info(f"  {team:20s} {prob:6.2f}%")
    
    # Generate Dashboard
    
    from src.output.dashboard import DashboardGenerator
    
    dashboard = DashboardGenerator()
    dashboard_path = dashboard.generate(
        simulation_results=results,
        match_predictions=match_predictions,
        model_metrics=ensemble.training_metrics,
    )
    
    total_elapsed = time.time() - total_start
    
    logger.info("\n" + "=" * 70)
    logger.info(f"  [OK] PIPELINE COMPLETE -- Total time: {total_elapsed:.1f}s")
    logger.info(f"  Dashboard: {dashboard_path}")
    logger.info("=" * 70)
    
    return results


def run_update():
    """Update predictions with latest match results."""
    logger.info("[UPDATE] Updating predictions with latest results...")
    
    from src.data_ingestion.github_loader import GithubDataLoader
    from src.data_ingestion.data_merger import DataMerger
    from src.features.pipeline import FeaturePipeline
    from src.model.predict import MatchPredictor
    from src.model.stacking import StackedEnsemble
    from src.simulation.simulator import TournamentSimulator
    from src.output.dashboard import DashboardGenerator
    
    # Fetch latest data
    logger.info("\n>> Fetching latest match results from Github...")
    gh_loader = GithubDataLoader()
    gh_loader.fetch_data()
    
    # Re-merge
    merger = DataMerger()
    matches_df, team_stats_df = merger.merge()
    
    # Rebuild features with updated data
    pipeline = FeaturePipeline()
    pipeline.build_team_features(matches_df, team_stats_df)
    pipeline.save()
    
    # Load saved model and generate new predictions
    ensemble = StackedEnsemble.load()
    predictor = MatchPredictor(pipeline=pipeline, ensemble=ensemble)
    prob_matrix = predictor.generate_probability_matrix()
    match_predictions = predictor.predict_group_matches()
    
    # Re-simulate
    simulator = TournamentSimulator()
    simulator.load_predictions(prob_matrix)
    results = simulator.simulate()
    simulator.save_results()
    
    # Regenerate dashboard
    dashboard = DashboardGenerator()
    dashboard_path = dashboard.generate(
        simulation_results=results,
        match_predictions=match_predictions,
        model_metrics=ensemble.training_metrics,
    )
    
    logger.info(f"[OK] Update complete. Dashboard: {dashboard_path}")
    
    for team, prob in simulator.get_winner_rankings()[:5]:
        logger.info(f"  {team:20s} {prob:6.2f}%")


def run_predict(team_a: str, team_b: str):
    """Predict a specific matchup."""
    from src.model.predict import MatchPredictor
    
    predictor = MatchPredictor()
    predictor.load()
    
    result = predictor.predict_match(team_a, team_b)
    
    print(f"\n{'='*50}")
    print(f"  {team_a} vs {team_b}")
    print(f"{'='*50}")
    print(f"  {team_a} Win:  {result['win_prob']:.1%}")
    print(f"  Draw:         {result['draw_prob']:.1%}")
    print(f"  {team_b} Win:  {result['loss_prob']:.1%}")
    print(f"  Expected GD:  {result['expected_goal_diff']:+.2f}")
    print(f"  Pred. Score:  {result['predicted_score']}")
    print(f"{'='*50}\n")


def run_simulate_only():
    """Run simulation using saved model predictions."""
    from src.simulation.simulator import TournamentSimulator
    from src.model.predict import MatchPredictor
    from src.output.dashboard import DashboardGenerator
    from src.model.stacking import StackedEnsemble
    
    predictor = MatchPredictor()
    predictor.load()
    prob_matrix = predictor.generate_probability_matrix()
    
    simulator = TournamentSimulator()
    simulator.load_predictions(prob_matrix)
    results = simulator.simulate()
    simulator.save_results()
    
    # Regenerate dashboard
    ensemble = StackedEnsemble.load()
    dashboard = DashboardGenerator()
    dashboard.generate(
        simulation_results=results,
        match_predictions=predictor.predict_group_matches(),
        model_metrics=ensemble.training_metrics,
    )
    
    print("\n-- Tournament Winner Probabilities:")
    for team, prob in simulator.get_winner_rankings()[:10]:
        print(f"  {team:20s} {prob:6.2f}%")


def run_dashboard_only():
    """Regenerate dashboard from saved simulation results."""
    from src.simulation.simulator import TournamentSimulator
    from src.model.predict import MatchPredictor
    from src.model.stacking import StackedEnsemble
    from src.output.dashboard import DashboardGenerator
    
    simulator = TournamentSimulator()
    results = simulator.load_results()
    
    if not results:
        logger.error("No simulation results found. Run simulation first.")
        return
    
    predictor = MatchPredictor()
    predictor.load()
    
    try:
        ensemble = StackedEnsemble.load()
        metrics = ensemble.training_metrics
    except Exception:
        metrics = {}
    
    dashboard = DashboardGenerator()
    path = dashboard.generate(
        simulation_results=results,
        match_predictions=predictor.predict_group_matches(),
        model_metrics=metrics,
    )
    print(f"[OK] Dashboard generated: {path}")



def main():
    parser = argparse.ArgumentParser(
        description="2026 FIFA World Cup Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode full              Run complete pipeline
  python main.py --mode full --test       Quick test with limited data
  python main.py --mode update            Update with latest results
  python main.py --mode predict --teams "Spain" "Brazil"
  python main.py --mode simulate          Re-run simulation only
  python main.py --mode dashboard         Regenerate dashboard
        """,
    )
    
    parser.add_argument(
        "--mode",
        choices=["full", "quick", "medium", "update", "predict", "simulate", "dashboard", "test"],
        default="full",
        help="Pipeline mode to run",
    )
    parser.add_argument(
        "--teams",
        nargs=2,
        metavar=("TEAM_A", "TEAM_B"),
        help="Teams for prediction mode (e.g., --teams Spain Brazil)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (faster, limited data)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=MC_DEFAULT_ITERATIONS,
        help=f"Number of Monte Carlo iterations (default: {MC_DEFAULT_ITERATIONS:,})",
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode in ["full", "test", "quick", "medium"]:
            run_full_pipeline(test_mode=(args.mode == "test" or args.test), mode=args.mode)
        elif args.mode == "update":
            run_update()
        elif args.mode == "predict":
            if not args.teams:
                parser.error("--teams required for predict mode")
            run_predict(args.teams[0], args.teams[1])
        elif args.mode == "simulate":
            run_simulate_only()
        elif args.mode == "dashboard":
            run_dashboard_only()
    except KeyboardInterrupt:
        logger.info("\n[WARN] Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
