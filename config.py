"""
Central configuration for the 2026 FIFA World Cup prediction system.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Create directories on import
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, MODEL_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


TOURNAMENT_NAME = "2026 FIFA World Cup"
TOURNAMENT_YEAR = 2026
NUM_TEAMS = 48
NUM_GROUPS = 12
TEAMS_PER_GROUP = 4
THIRD_PLACE_ADVANCE = 8  # 8 best 3rd-placed teams advance

# Full group assignments from the December 2025 draw
# Note: Some spots are TBD (European/Intercontinental playoffs).
# We use the most likely qualifiers as placeholders.
GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# All participating teams (flattened)
ALL_TEAMS = sorted(set(team for group in GROUPS.values() for team in group))

# Team-to-group mapping
TEAM_TO_GROUP = {}
for group_name, teams in GROUPS.items():
    for team in teams:
        TEAM_TO_GROUP[team] = group_name

FIFA_RANKINGS = {
    "Argentina": 1877.27,
    "Spain": 1874.71,
    "France": 1870.70,
    "England": 1828.02,
    "Portugal": 1767.85,
    "Brazil": 1765.86,
    "Morocco": 1755.10,
    "Netherlands": 1753.57,
    "Belgium": 1742.24,
    "Germany": 1735.77,
    "Italy": 1720.00,
    "Colombia": 1710.00,
    "Uruguay": 1700.00,
    "Croatia": 1695.00,
    "Japan": 1690.00,
    "United States": 1680.00,
    "Mexico": 1670.00,
    "Switzerland": 1660.00,
    "Denmark": 1655.00,
    "Turkey": 1650.00,
    "Ecuador": 1640.00,
    "Senegal": 1635.00,
    "Austria": 1630.00,
    "Ukraine": 1625.00,
    "South Korea": 1620.00,
    "Australia": 1610.00,
    "Norway": 1605.00,
    "Egypt": 1600.00,
    "Iran": 1595.00,
    "Algeria": 1590.00,
    "Tunisia": 1580.00,
    "Paraguay": 1575.00,
    "Scotland": 1570.00,
    "Canada": 1565.00,
    "Ivory Coast": 1560.00,
    "Saudi Arabia": 1555.00,
    "Ghana": 1550.00,
    "Panama": 1540.00,
    "Uzbekistan": 1530.00,
    "Qatar": 1520.00,
    "Jordan": 1510.00,
    "DR Congo": 1505.00,
    "Iraq": 1500.00,
    "South Africa": 1495.00,
    "Cape Verde": 1480.00,
    "Haiti": 1460.00,
    "Curacao": 1440.00,
    "New Zealand": 1430.00,
}

ROLLING_WINDOWS = [5, 10]           # Match windows for rolling features
ELO_K_FACTORS = {
    "friendly": 20,
    "qualifier": 30,
    "continental": 40,
    "world_cup": 50,
}
ELO_HOME_ADVANTAGE = 100            # Elo points for home advantage
ELO_NEUTRAL_VENUE_FACTOR = 0.0      # No home advantage at neutral venues
ELO_MOV_CAP = 3                     # Margin-of-victory cap (goals)

OPTUNA_N_TRIALS = 100               # Trials per base model
OPTUNA_CV_FOLDS = 5                 # TimeSeriesSplit folds
RANDOM_SEED = 42
TEST_SIZE = 0.2                     # Holdout for final evaluation
RECENCY_HALF_LIFE_DAYS = 270        # Aggressive exponential sample-weight half-life for recent form
TRAINING_CUTOFF_DATE = "2022-01-01"  # Keep model focused on the current World Cup cycle
RECENCY_MIN_WEIGHT = 0.03          # Floor so older retained matches keep small context
MATCH_IMPORTANCE_WEIGHTS = {
    "world_cup": 1.50,
    "qualifier": 1.25,
    "continental": 1.15,
    "friendly": 0.65,
    "default": 1.00,
}
ELO_HALF_LIFE_DAYS = 1095          # Long Elo decays halfway to baseline every 3 years
FORM_ELO_HALF_LIFE_DAYS = 180      # Form Elo decays halfway to baseline every 6 months
FORM_ELO_K_MULTIPLIER = 3.0

# Target columns
TARGET_WDL = "result"               # 0=Loss, 1=Draw, 2=Win (from team_a perspective)
TARGET_GD = "goal_diff"             # goal_diff = team_a_goals - team_b_goals

MC_DEFAULT_ITERATIONS = 1_000_000
MC_DEFAULT_SEED = 42
MC_NUM_THREADS = os.cpu_count() or 4

DASHBOARD_TITLE = "2026 FIFA World Cup Predictions"
CONFEDERATIONS = {
    "UEFA": [
        "Spain", "France", "England", "Portugal", "Belgium", "Germany",
        "Netherlands", "Italy", "Croatia", "Denmark", "Switzerland",
        "Austria", "Turkey", "Ukraine", "Scotland", "Norway",
    ],
    "CONMEBOL": [
        "Argentina", "Brazil", "Colombia", "Uruguay", "Ecuador", "Paraguay",
    ],
    "CONCACAF": [
        "United States", "Mexico", "Canada", "Panama", "Haiti", "Curacao",
    ],
    "CAF": [
        "Morocco", "Senegal", "Ivory Coast", "Egypt", "Algeria", "Tunisia",
        "Ghana", "South Africa", "Cape Verde", "DR Congo",
    ],
    "AFC": [
        "Japan", "South Korea", "Australia", "Iran", "Saudi Arabia",
        "Qatar", "Uzbekistan", "Jordan", "Iraq",
    ],
    "OFC": [
        "New Zealand",
    ],
}

TEAM_TO_CONFEDERATION = {}
for conf, teams in CONFEDERATIONS.items():
    for team in teams:
        TEAM_TO_CONFEDERATION[team] = conf

TEAM_ALIASES = {
    "USA": "United States",
    "US": "United States",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Korea Republic": "South Korea",
    "Korea, Republic of": "South Korea",
    "Korea DPR": "North Korea",
    "Czechia": "Czech Republic",
    "IR Iran": "Iran",
    "Congo DR": "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
    "Cabo Verde": "Cape Verde",
    "Curaçao": "Curacao",
    "Rep. of Ireland": "Ireland",
    "Republic of Ireland": "Ireland",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "North Macedonia": "North Macedonia",
    "Chinese Taipei": "Taiwan",
    "China PR": "China",
}


def resolve_team_name(name: str) -> str:
    """Resolve a team name to its canonical form using aliases."""
    return TEAM_ALIASES.get(name, name)
