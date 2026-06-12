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
| 10| Netherlands | **2.40%** | Deep runs in recent tournaments |

### Top 10 Official Squad Power Ratings

Calculated directly from the official 26-man rosters using Elite Depth (players in Top 5 European Leagues), International Experience (Total Caps), and Output (Total International Goals).

| Rank | Team | Squad Power Index | Elite Players (Top 5 Leagues) | Total Caps | Total Goals |
|:---:|:---|:---:|:---:|:---:|:---:|
| 1 | Argentina | **2071.5** | 20 | 1251 | 223 |
| 2 | Switzerland | **2014.0** | 24 | 1132 | 124 |
| 3 | Belgium | **1970.5** | 20 | 1085 | 214 |
| 4 | England | **1969.0** | 25 | 842 | 149 |
| 5 | Portugal | **1948.5** | 17 | 1161 | 259 |
| 6 | Spain | **1905.5** | 26 | 743 | 117 |
| 7 | Germany | **1882.0** | 25 | 844 | 105 |
| 8 | Netherlands | **1836.5** | 22 | 861 | 153 |
| 9 | France | **1801.5** | 24 | 803 | 100 |
| 10 | Croatia | **1786.0** | 18 | 1160 | 153 |

*Note: Brazil currently ranks outside the top 10 in pure Squad Power due to recent roster transitions, reflecting their reliance on legacy rather than current elite output.*

---

## Data Source & Architecture

The system is divided into four main layers:

### 1. Data Ingestion & Processing

The foundation of the prediction model relies on a highly comprehensive historical football dataset.

- **Primary Source**: The model draws exclusively from the `martj42/international_results` GitHub repository. This robust, open-source database contains over 40,000 international football results dating back to the 19th century, ensuring massive historical depth and a fast, reliable ingestion process without the bottlenecks of fetching from multiple live endpoints.
- **Processing (Timeline Truncation)**: The ingestion pipeline fetches the full dataset to establish stable Elo rating baselines, but explicitly truncates the machine learning training matrix to matches played from **2010 onwards**. This strictly limits the model to predicting on modern football trends, eliminating the bias of "historical ghost prestige" from decades past.
- **Official Squad Lists (PDF Extraction)**: The system utilizes `pdfplumber` to scrape the official FIFA 2026 World Cup 26-man squad list PDF. This gives the model the absolute ground-truth roster of players traveling to the tournament.

### 2. Feature Engineering

A robust, custom feature pipeline transforms raw match results into powerful predictive indicators:

- **Dual Elo Ratings**: A dual-system tracking both long-term historical prestige and a highly reactive 'Form Elo' (using an amplified K-factor) to mathematically measure a team's current trajectory.
- **Squad Power Rating (Official Squads)**: The pipeline dynamically parses the official tournament squad list to compute a weighted index of squad dominance based on three metrics: Number of players in Top 5 European Leagues (Elite Depth), Total International Caps (Experience), and Total International Goals (In-Form Output).
- **Exponential Sample Weighting**: During model training, an exponential decay function (with a 2-year half-life) mathematically scales the `sample_weight` of historical matches. Recent matches aggressively dictate the gradient descent, while older matches provide only underlying context.
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
