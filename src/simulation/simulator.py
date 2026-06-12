"""
Python orchestrator that:.
"""

import json
import logging
import time
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    MC_DEFAULT_ITERATIONS,
    MC_DEFAULT_SEED,
    MC_NUM_THREADS,
    GROUPS,
    ALL_TEAMS,
    OUTPUT_DIR,
)
from src.simulation.tournament import TournamentStructure

logger = logging.getLogger(__name__)


class TournamentSimulator:
    """
    Orchestrates Monte Carlo tournament simulation.
    
    Connects the ML model's match predictions to the simulation engine,
    handles the C++/Python backend selection, and post-processes results.
    """

    def __init__(self):
        """Initialize the simulator."""
        self.structure = TournamentStructure()
        self.prob_matrix: dict = {}
        self.results: dict = {}
        self._cpp_available = self._check_cpp_engine()

    def _check_cpp_engine(self) -> bool:
        """Check if the C++ simulation module is available and functional."""
        try:
            import mc_simulation
            # Verify the compiled function actually exists (not just a directory import)
            if not hasattr(mc_simulation, 'simulate'):
                logger.warning(
                    "mc_simulation module found but not compiled. "
                    "Using Python fallback. "
                    "Build C++ with: python setup.py build_ext --inplace"
                )
                return False
            logger.info("C++ Monte Carlo engine loaded successfully")
            return True
        except ImportError:
            logger.warning(
                "C++ Monte Carlo engine not available. "
                "Using Python fallback (slower). "
                "Build with: python setup.py build_ext --inplace"
            )
            return False

    def load_predictions(self, prediction_matrix: dict):
        """
        Load match predictions from the ML model.
        
        Args:
            prediction_matrix: Dict mapping "teamA|teamB" -> {win, draw, loss, xgd}
        """
        self.prob_matrix = prediction_matrix
        logger.info(f"Loaded {len(self.prob_matrix)} matchup predictions")

    def _build_cpp_prob_map(self) -> dict:
        """
        Convert prediction matrix to the format expected by the C++ engine.
        Keys: "id1|id2" (integer IDs), Values: [win, draw, loss, xgd]
        """
        cpp_map = {}
        
        for key, probs in self.prob_matrix.items():
            parts = key.split("|")
            if len(parts) != 2:
                continue
            
            team_a, team_b = parts
            id_a = self.structure.get_team_id(team_a)
            id_b = self.structure.get_team_id(team_b)
            
            if id_a < 0 or id_b < 0:
                continue
            
            cpp_key = f"{id_a}|{id_b}"
            cpp_map[cpp_key] = [
                probs["win"],
                probs["draw"],
                probs["loss"],
                probs["xgd"],
            ]

        return cpp_map

    def simulate_cpp(self, num_iterations: int = MC_DEFAULT_ITERATIONS,
                     num_threads: int = MC_NUM_THREADS,
                     seed: int = MC_DEFAULT_SEED) -> dict:
        """
        Run simulation using the C++ engine.
        
        Args:
            num_iterations: Number of tournament simulations
            num_threads: OpenMP thread count
            seed: Random seed for reproducibility
            
        Returns:
            Dict mapping team_name -> {round_name: probability %}
        """
        import mc_simulation
        
        logger.info(f"Running C++ simulation: {num_iterations:,} iterations, "
                    f"{num_threads} threads...")
        
        start_time = time.time()
        
        raw_results = mc_simulation.simulate(
            groups=self.structure.groups_as_ids,
            prob_map=self._build_cpp_prob_map(),
            num_iterations=num_iterations,
            num_threads=num_threads,
            seed=seed,
        )
        
        elapsed = time.time() - start_time
        logger.info(f"C++ simulation completed in {elapsed:.2f}s "
                    f"({num_iterations / elapsed:,.0f} iterations/sec)")
        
        # Convert to human-readable format
        return self.structure.format_bracket_results(raw_results)

    def simulate_python(self, num_iterations: int = 100_000,
                        seed: int = MC_DEFAULT_SEED) -> dict:
        """
        Pure Python/NumPy fallback simulation.
        
        Slower than C++ but works without compilation.
        Recommended: 100K iterations (~30 seconds).
        
        Args:
            num_iterations: Number of tournament simulations
            seed: Random seed
            
        Returns:
            Dict mapping team_name -> {round_name: probability %}
        """
        logger.info(f"Running Python simulation: {num_iterations:,} iterations...")
        
        np.random.seed(seed)
        start_time = time.time()
        
        # Initialize counters
        teams = sorted(ALL_TEAMS)
        counters = {
            team: {
                "group_exit": 0, "r32": 0, "r16": 0,
                "qf": 0, "sf": 0, "final": 0, "winner": 0,
            }
            for team in teams
        }

        for iteration in range(num_iterations):
            # Simulate group stage
            all_qualifiers = []
            all_third = []
            
            for group_name in sorted(GROUPS.keys()):
                group_teams = GROUPS[group_name]
                standings = {t: {"pts": 0, "gd": 0, "gf": 0} for t in group_teams}
                
                # Round-robin
                for i in range(len(group_teams)):
                    for j in range(i + 1, len(group_teams)):
                        a, b = group_teams[i], group_teams[j]
                        probs = self._get_match_probs(a, b)
                        
                        # Simulate goals
                        avg_total = 2.5
                        lam_a = max(0.2, (avg_total + probs["xgd"]) / 2)
                        lam_b = max(0.2, (avg_total - probs["xgd"]) / 2)
                        
                        ga = np.random.poisson(lam_a)
                        gb = np.random.poisson(lam_b)
                        
                        standings[a]["gf"] += ga
                        standings[a]["gd"] += ga - gb
                        standings[b]["gf"] += gb
                        standings[b]["gd"] += gb - ga
                        
                        if ga > gb:
                            standings[a]["pts"] += 3
                        elif ga == gb:
                            standings[a]["pts"] += 1
                            standings[b]["pts"] += 1
                        else:
                            standings[b]["pts"] += 3
                
                # Sort standings
                sorted_teams = sorted(
                    group_teams,
                    key=lambda t: (standings[t]["pts"], standings[t]["gd"], standings[t]["gf"]),
                    reverse=True,
                )
                
                # Top 2 qualify directly
                all_qualifiers.extend(sorted_teams[:2])
                # Third-placed team
                if len(sorted_teams) >= 3:
                    all_third.append((sorted_teams[2], standings[sorted_teams[2]]))
                # Fourth-placed team exits
                if len(sorted_teams) >= 4:
                    counters[sorted_teams[3]]["group_exit"] += 1
            
            # Select 8 best third-placed teams
            all_third.sort(key=lambda x: (x[1]["pts"], x[1]["gd"], x[1]["gf"]), reverse=True)
            advancing_thirds = [t[0] for t in all_third[:8]]
            for t, _ in all_third[8:]:
                counters[t]["group_exit"] += 1
            
            all_qualifiers.extend(advancing_thirds)
            
            # Record R32 qualification
            for t in all_qualifiers:
                counters[t]["r32"] += 1
            
            # Knockout rounds
            current_round = all_qualifiers[:32]  # Ensure exactly 32
            if len(current_round) < 32:
                current_round.extend(advancing_thirds[:32 - len(current_round)])
            
            round_names = ["r16", "qf", "sf", "final"]
            
            for rnd_idx, rnd_name in enumerate(round_names):
                if len(current_round) < 2:
                    break
                    
                next_round = []
                for i in range(0, len(current_round) - 1, 2):
                    a, b = current_round[i], current_round[i + 1]
                    probs = self._get_match_probs(a, b)
                    
                    avg_total = 2.5
                    lam_a = max(0.2, (avg_total + probs["xgd"]) / 2)
                    lam_b = max(0.2, (avg_total - probs["xgd"]) / 2)
                    
                    ga = np.random.poisson(lam_a)
                    gb = np.random.poisson(lam_b)
                    
                    # Knockout: extra time + penalties if drawn
                    if ga == gb:
                        et_a = np.random.poisson(max(0.1, lam_a / 3))
                        et_b = np.random.poisson(max(0.1, lam_b / 3))
                        ga += et_a
                        gb += et_b
                    
                    if ga == gb:
                        # Penalties
                        pk_adv = 0.5 + (probs["win"] - probs["loss"]) * 0.1
                        if np.random.random() < pk_adv:
                            ga += 1
                        else:
                            gb += 1
                    
                    winner = a if ga > gb else b
                    next_round.append(winner)
                    counters[winner][rnd_name] += 1
                
                current_round = next_round
            
            # Winner
            if current_round:
                counters[current_round[0]]["winner"] += 1

        # Convert counts to percentages
        n = float(num_iterations)
        results = {}
        for team, counts in counters.items():
            results[team] = {
                k: round(v / n * 100, 2) for k, v in counts.items()
            }

        elapsed = time.time() - start_time
        logger.info(f"Python simulation completed in {elapsed:.1f}s")
        
        return results

    def _get_match_probs(self, team_a: str, team_b: str) -> dict:
        """Get match probabilities, with fallback to Elo-based estimates."""
        key = f"{team_a}|{team_b}"
        if key in self.prob_matrix:
            return self.prob_matrix[key]
        
        # Try reverse
        rev_key = f"{team_b}|{team_a}"
        if rev_key in self.prob_matrix:
            p = self.prob_matrix[rev_key]
            return {"win": p["loss"], "draw": p["draw"], "loss": p["win"], "xgd": -p["xgd"]}
        
        # Elo-based fallback
        from src.features.elo_rating import EloRatingSystem
        elo = EloRatingSystem()
        result = elo.predict_match(team_a, team_b, is_neutral=True)
        return {
            "win": result["win_a"],
            "draw": result["draw"],
            "loss": result["win_b"],
            "xgd": result["elo_diff"] / 400,
        }

    def simulate(self, num_iterations: int = MC_DEFAULT_ITERATIONS,
                 seed: int = MC_DEFAULT_SEED) -> dict:
        """
        Run simulation using the best available backend.
        
        Tries C++ first, falls back to Python.
        
        Args:
            num_iterations: Number of iterations
            seed: Random seed
            
        Returns:
            Dict mapping team_name -> {round_name: probability %}
        """
        if self._cpp_available:
            self.results = self.simulate_cpp(
                num_iterations=num_iterations,
                num_threads=MC_NUM_THREADS,
                seed=seed,
            )
        else:
            # Use fewer iterations for Python fallback
            py_iters = min(num_iterations, 100_000)
            if py_iters < num_iterations:
                logger.info(f"Python fallback: reducing to {py_iters:,} iterations")
            self.results = self.simulate_python(
                num_iterations=py_iters,
                seed=seed,
            )
        
        return self.results

    def get_winner_rankings(self) -> list[tuple[str, float]]:
        """
        Get teams ranked by probability of winning the tournament.
        
        Returns:
            List of (team_name, win_probability_%) sorted descending
        """
        if not self.results:
            return []
        
        rankings = [
            (team, probs.get("winner", 0))
            for team, probs in self.results.items()
        ]
        return sorted(rankings, key=lambda x: -x[1])

    def get_group_advancement_probs(self) -> dict:
        """
        Get probability of advancing from group stage for each team,
        organized by group.
        
        Returns:
            Dict mapping group_name -> [(team, advance_prob), ...]
        """
        if not self.results:
            return {}

        group_probs = {}
        for group_name, teams in GROUPS.items():
            team_probs = []
            for team in teams:
                probs = self.results.get(team, {})
                advance = 100.0 - probs.get("group_exit", 0)
                team_probs.append((team, round(advance, 2)))
            team_probs.sort(key=lambda x: -x[1])
            group_probs[group_name] = team_probs

        return group_probs

    def save_results(self, path: Optional[Path] = None):
        """Save simulation results to JSON."""
        path = path or OUTPUT_DIR / "simulation_results.json"
        
        output = {
            "results": self.results,
            "winner_rankings": self.get_winner_rankings()[:10],
            "group_advancement": self.get_group_advancement_probs(),
            "iterations": MC_DEFAULT_ITERATIONS,
        }
        
        path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        logger.info(f"Simulation results saved to {path}")

    def load_results(self, path: Optional[Path] = None) -> dict:
        """Load previously saved simulation results."""
        path = path or OUTPUT_DIR / "simulation_results.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.results = data.get("results", {})
            return self.results
        return {}
