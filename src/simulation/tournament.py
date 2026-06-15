"""
Encodes the 2026 FIFA World Cup bracket structure and rules.
"""

import logging
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import GROUPS, ALL_TEAMS, NUM_GROUPS

logger = logging.getLogger(__name__)


class TournamentStructure:
    """
    Manages the 2026 FIFA World Cup bracket structure.
    
    48 teams in 12 groups of 4.
    Top 2 per group (24) + 8 best third-placed teams = 32 advance.
    Then: R32 -> R16 -> QF -> SF -> Final.
    
    This class handles:
    - Team name <-> ID mapping (C++ engine uses integer IDs)
    - Group structure encoding
    - Bracket path definitions
    - "Lock-in" of actual results as the tournament progresses
    """

    def __init__(self):
        """Initialize the tournament structure from config."""
        # Build name <-> id mapping
        self.team_to_id: dict[str, int] = {}
        self.id_to_team: dict[int, str] = {}
        
        for i, team in enumerate(sorted(ALL_TEAMS)):
            self.team_to_id[team] = i
            self.id_to_team[i] = team

        # Build group structure as integer IDs
        self.groups_as_ids: list[list[int]] = []
        self.group_names: list[str] = []
        
        for group_name in sorted(GROUPS.keys()):
            teams = GROUPS[group_name]
            ids = [self.team_to_id[t] for t in teams]
            self.groups_as_ids.append(ids)
            self.group_names.append(group_name)

        # Locked-in results (actual match results from the tournament)
        self.locked_results: dict[str, dict] = {}

    def get_team_id(self, name: str) -> int:
        """Get integer ID for a team name."""
        return self.team_to_id.get(name, -1)

    def get_team_name(self, team_id: int) -> str:
        """Get team name from integer ID."""
        return self.id_to_team.get(team_id, f"Unknown({team_id})")

    def lock_result(self, team_a: str, team_b: str, 
                    score_a: int, score_b: int, stage: str = "GROUP_STAGE"):
        """
        Lock in an actual match result. The simulator will use this
        result instead of generating a random one.
        
        Args:
            team_a: First team
            team_b: Second team
            score_a: Goals scored by team_a
            score_b: Goals scored by team_b
            stage: Tournament stage
        """
        stage = str(stage).upper().replace(" ", "_")
        key = f"{team_a}|{team_b}"
        self.locked_results[key] = {
            "team_a": team_a,
            "team_b": team_b,
            "score_a": score_a,
            "score_b": score_b,
            "stage": stage,
        }
        logger.info(f"Locked result: {team_a} {score_a}-{score_b} {team_b}")

    def format_bracket_results(self, raw_results: dict) -> dict:
        """
        Convert raw simulation results (team_id -> probabilities)
        to human-readable format (team_name -> probabilities).
        
        Args:
            raw_results: Dict from C++ engine {team_id: {round: probability}}
            
        Returns:
            Dict with team names as keys
        """
        formatted = {}
        for team_id, probs in raw_results.items():
            team_id_int = int(team_id)
            team_name = self.get_team_name(team_id_int)
            
            formatted[team_name] = {
                "group_exit": round(probs.get("group_exit", 0) * 100, 2),
                "r32": round(probs.get("r32", 0) * 100, 2),
                "r16": round(probs.get("r16", 0) * 100, 2),
                "qf": round(probs.get("qf", 0) * 100, 2),
                "sf": round(probs.get("sf", 0) * 100, 2),
                "final": round(probs.get("final", 0) * 100, 2),
                "winner": round(probs.get("winner", 0) * 100, 2),
            }

        return formatted

