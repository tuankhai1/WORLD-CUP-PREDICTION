# 2026 FIFA World Cup Prediction Model

## Introduction

This project is an advanced machine learning and Monte Carlo simulation pipeline built to forecast the upcoming 2026 FIFA World Cup. It utilizes historical match data to train an ensemble of gradient-boosting models (XGBoost, CatBoost, LightGBM) to predict match outcomes (Win, Draw, Loss) and expected goal differences. The predictions power a high-performance C++ Monte Carlo simulator to run millions of tournament permutations, culminating in a sleek, glassmorphic UI dashboard that visually maps out the most probable knockout bracket.

## Data Source & Architecture

The system is divided into four main layers:

### 1. Data Ingestion & Processing

The foundation of the prediction model relies on a highly comprehensive historical football dataset.

- **Primary Source**: The model draws exclusively from the `martj42/international_results` GitHub repository. This robust, open-source database contains over 40,000 international football results dating back to the 19th century, ensuring massive historical depth without relying on external APIs that are prone to rate limits or blocking.
- **Processing**: The ingestion pipeline fetches the latest dataset, filters for relevant eras and tournament weights, and standardizes team names to fuel the dynamic Elo engine. Secondary data structures prepare the specific 2026 World Cup group matchups defined in `config.py`.

### 2. Feature Engineering

A robust, custom feature pipeline transforms raw match results into powerful predictive indicators:

- **Elo Ratings**: A dynamic, historical Elo ranking system that updates after every match to provide a mathematically sound representation of team strength.
- **Form & Momentum**: Calculates a team's recent performance (win/loss ratio, goals scored/conceded in the last N matches).
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

The resulting `dashboard.html` will be generated in the `output/` directory. Simply open it in any web browser to explore the predictions!
