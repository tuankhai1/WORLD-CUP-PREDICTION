# 2026 FIFA World Cup Prediction Model

## Introduction

This project is an advanced machine learning and Monte Carlo simulation pipeline built to forecast the upcoming 2026 FIFA World Cup. It utilizes historical match data to train an ensemble of gradient-boosting models (XGBoost, CatBoost, LightGBM) to predict match outcomes (Win, Draw, Loss) and expected goal differences. The predictions power a high-performance C++ Monte Carlo simulator to run millions of tournament permutations, culminating in a sleek, glassmorphic UI dashboard that visually maps out the most probable knockout bracket.

## Current Model Results

Based on our most recent end-to-end Monte Carlo simulation using the **Modern Era Reset** and **Official 2026 Squad Lists**, the tournament probabilities and intrinsic team ratings are as follows:

### Top 10 Tournament Winner Probabilities

| Rank | Team | Winner Probability | Note |
|:---:|:---|:---:|:---|
| 1 | Argentina | **18.65%** | Highest Official Squad Power |
| 2 | Brazil | **13.94%** | Still highly ranked due to modern form |
| 3 | Spain | **13.32%** | Highest 2010+ Modern Era Elo (2130) |
| 4 | France | **12.92%** | Second Highest Modern Era Elo (2053) |
| 5 | Morocco | **5.67%** | Incredible Modern Overperformance |
| 6 | Japan | **5.55%** | Consistently high Rolling xG Metrics |
| 7 | England | **4.60%** | Massive Squad Power (1969.0) |
| 8 | Belgium | **4.06%** | Excellent form and legacy depth |
| 9 | Portugal | **3.78%** | Elite depth, historically strong in Europe |
| 10 | Netherlands | **2.40%** | Deep runs in recent tournaments |

### Top 10 Club Form Power Ratings

Calculated by aggregating the 2025/2026 season statistics of each nation's top 15 players across the Top 5 European Leagues. Metrics weighted include Goals, Assists, Shots on Target, Tackles, Interceptions, and Points Per Match, removing legacy bias like Total Caps and International Goals.

| Rank | Team | Club Form Power | Note |
|:---:|:---|:---:|:---|
| 1 | France | **1919.60** | Elite depth across all positions |
| 2 | Spain | **1892.78** | Dominant midfield playmaking stats |
| 3 | Germany | **1712.98** | Strong attacking and defensive balance |
| 4 | England | **1540.12** | High output forwards |
| 5 | Brazil | **1321.60** | Reduced due to legacy transitions |
| 6 | Netherlands | **1299.01** | Solid core |
| 7 | Argentina | **1274.70** | Star-heavy but top-heavy |
| 8 | Portugal | **1228.85** | Strong elite contributions |
| 9 | Austria | **988.88** | Overperforming pressing metrics |
| 10 | Senegal | **926.30** | Best African representation |

---

## Data Source & Architecture

The system is divided into four main layers:

### 1. Data Ingestion & Processing

The foundation of the prediction model relies on a highly comprehensive historical football dataset.

- **Primary Source**: The model draws exclusively from the `martj42/international_results` GitHub repository. This robust, open-source database contains over 40,000 international football results dating back to the 19th century, ensuring massive historical depth and a fast, reliable ingestion process without the bottlenecks of fetching from multiple live endpoints.
- **Processing (Timeline Truncation)**: The ingestion pipeline fetches the full dataset to establish stable Elo rating baselines, but uses matches from **2010 onwards** to initialize modern-era Elo, then explicitly truncates the machine learning training matrix to matches played from **2022 onwards**. This strictly limits the model to predicting on modern football trends, eliminating the bias of "historical ghost prestige" from decades past.
- **Official Squad Lists (PDF Extraction)**: The system utilizes `pdfplumber` to scrape the official FIFA 2026 World Cup 26-man squad list PDF. This gives the model the absolute ground-truth roster of players traveling to the tournament.

### 2. Feature Engineering

A robust, custom feature pipeline transforms raw match results into powerful predictive indicators:

- **Dual Elo Ratings**: A dual-system tracking both long-term historical prestige and a highly reactive 'Form Elo' (using an amplified K-factor) to mathematically measure a team's current trajectory.
- **Club Form Power (2025/2026)**: The pipeline computes a weighted index of squad dominance based on the accumulated stats of each nation's players in the Top 5 European Leagues over the most recent season (Goals, Assists, Tackles, Interceptions). Legacy metrics like Total Caps have been explicitly removed to prevent bias.
- **Exponential Sample Weighting**: During model training, an exponential decay function (configurable via `RECENCY_HALF_LIFE_DAYS`, currently a 270-day half-life) mathematically scales the `sample_weight` of historical matches. Recent matches aggressively dictate the gradient descent, while older matches provide only underlying context.
- **Head-to-Head Statistics**: Historical matchups between two specific nations.
- **Rolling xG & Pressing Intensity**: Advanced metrics (where available) modeling offensive output and defensive pressure.

### 3. Modeling

The predictive core uses a **Stacked Ensemble** approach to maximize accuracy and prevent overfitting:

- **Base Models**: Three gradient-boosting algorithms act as the base learners: `CatBoost`, `XGBoost`, and `LightGBM`. These were chosen for their exceptional performance on tabular data and differing algorithmic approaches to tree building.
- **Hyperparameter Tuning**: Optuna runs a Bayesian optimization search to find the mathematically optimal parameters (learning rate, depth, regularization) for each base model.
- **Meta-Model**: The predictions from the three base models are fed into a final meta-learner (Logistic Regression) to produce the absolute, calibrated probability of a Win, Draw, or Loss for any given matchup.

### 4. Simulation & Output

- **C++ Monte Carlo Engine**: A highly optimized C++ backend simulates the entire World Cup structure (group stages, tie-breakers, 32-team knockouts) over 1,000,000+ times to account for variance and upsets.
- **Dashboard**: Results are dumped into an interactive, glassmorphic HTML/CSS dashboard equipped with Plotly visualizations mapping out the most mathematically probable deterministic bracket.

## Project Structure

```text
.
├── config.py             # World Cup 2026 groups and tournament configurations
├── main.py               # CLI entry point for training, simulation, and dashboard generation
├── update.py             # Data refresh script
├── setup.py              # Build script for C++ extensions
├── requirements.txt      # Python dependencies
├── src/
│   ├── data_ingestion/   # Connectors for data fetching
│   ├── features/         # Feature engineering and preprocessing pipelines
│   ├── model/            # Optuna tuning, ensemble stacking, and inference logic
│   ├── simulation/       # C++ simulator bindings and Python simulation wrappers
│   └── output/           # HTML dashboard generation and Plotly chart logic
├── data/                 # Raw and processed datasets (ignored in Git)
├── models/               # Saved Joblib models and pipelines (ignored in Git)
└── output/               # Generated dashboard and JSON results (ignored in Git)
```

## Steps to Implement

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/world-cup-prediction.git
cd world-cup-prediction
pip install -r requirements.txt
```

### 2. Build the C++ Simulator

The Monte Carlo simulation relies on a highly optimized C++ backend. Compile the extension before running the pipeline:

```bash
python setup.py build_ext --inplace
```

### 3. Run the Pipeline

The project exposes a CLI to handle data fetching, model tuning, and dashboard generation.

To run the complete pipeline (Train -> Simulate -> Dashboard):

```bash
python main.py --mode full
```

To run a quicker version with fewer tuning trials:

```bash
python main.py --mode quick
```

To just regenerate the dashboard without retraining:

```bash
python main.py --mode dashboard
```


To control Monte Carlo simulation depth:

```bash
python main.py --mode full --iterations 1000000
```

The resulting `dashboard.html` will be generated in the `output/` directory. Simply open it in any web browser to explore the predictions!
