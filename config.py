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

FOOTBALL_DATA_API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"
FOOTBALL_DATA_RATE_LIMIT = 10  # calls per minute (free tier)

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

# After group stage: 24 teams qualify (12 × top-2) + 8 best 3rd-placed = 32
# Round of 32 matchups based on FIFA's official bracket paths
# Format: (group_position_1, group_position_2)
# Positions encoded as "A1" = Group A winner, "A2" = Group A runner-up, etc.
# "3rd_X" = best 3rd-place from a set of groups

R32_MATCHUPS = [
    ("A1", "3rd_CDE"),    # Match 49
    ("B1", "3rd_ADF"),    # Match 50
    ("C1", "3rd_ABF"),    # Match 51
    ("D1", "3rd_BEF"),    # Match 52
    ("E1", "3rd_ACD"),    # Match 53
    ("F1", "3rd_BCD"),    # Match 54
    ("G1", "H2"),         # Match 55
    ("H1", "G2"),         # Match 56
    ("I1", "J2"),         # Match 57
    ("J1", "I2"),         # Match 58
    ("K1", "L2"),         # Match 59
    ("L1", "K2"),         # Match 60
    ("A2", "B2"),         # Match 61
    ("C2", "D2"),         # Match 62
    ("E2", "F2"),         # Match 63
    ("G2", "3rd_IJL"),    # Match 64
    ("H2", "3rd_GKL"),    # Match 65
    ("I2", "3rd_GHK"),    # Match 66
    ("J2", "3rd_GIJ"),    # Match 67
    ("K2", "3rd_HIJ"),    # Match 68
    ("L2", "3rd_GHL"),    # Match 69
]

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

# Target columns
TARGET_WDL = "result"               # 0=Loss, 1=Draw, 2=Win (from team_a perspective)
TARGET_GD = "goal_diff"             # goal_diff = team_a_goals - team_b_goals

MC_DEFAULT_ITERATIONS = 1_000_000
MC_DEFAULT_SEED = 42
MC_NUM_THREADS = os.cpu_count() or 4

DASHBOARD_TITLE = "⚽ 2026 FIFA World Cup Predictions"
DASHBOARD_THEME = "dark"
COLOR_PALETTE = {
    "bg_primary": "#0a0a1a",
    "bg_secondary": "#12122a",
    "bg_card": "rgba(255, 255, 255, 0.05)",
    "text_primary": "#e0e0ff",
    "text_secondary": "#8888aa",
    "accent_gold": "#ffd700",
    "accent_blue": "#4a9eff",
    "accent_green": "#00e676",
    "accent_red": "#ff5252",
    "gradient_start": "#1a1a3e",
    "gradient_end": "#0a0a1a",
    "win_color": "#00e676",
    "draw_color": "#ffab40",
    "loss_color": "#ff5252",
}

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