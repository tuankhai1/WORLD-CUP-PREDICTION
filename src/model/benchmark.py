"""
Benchmark tabular base models for W/D/L prediction.

The benchmark consumes the generated feature matrix and target files from
data/processed, evaluates several baseline/base classifiers, and writes a
GitHub-ready Markdown result table.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import PROCESSED_DATA_DIR, OUTPUT_DIR, PROJECT_ROOT, RANDOM_SEED

logger = logging.getLogger(__name__)

CLASSES = np.array([0, 1, 2])
README_START = "<!-- MODEL_BENCHMARK_START -->"
README_END = "<!-- MODEL_BENCHMARK_END -->"


@dataclass
class ModelSpec:
    name: str
    build: Callable[[], BaseEstimator]
    category: str
    display_order: int
    feature_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None


def _rating_features(X: pd.DataFrame) -> pd.DataFrame:
    """Use only rating-style features for a compact interpretable baseline."""
    cols = [
        col
        for col in X.columns
        if any(token in col for token in ["elo", "fifa_rating"])
    ]
    return X[cols]


def _build_model_specs(include_slow: bool = True) -> list[ModelSpec]:
    specs = [
        ModelSpec(
            "Class-prior baseline",
            lambda: DummyClassifier(strategy="prior"),
            category="Baseline",
            display_order=10,
        ),
        ModelSpec(
            "Elo/FIFA Logistic Regression",
            lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.5,
                    max_iter=2000,
                    multi_class="auto",
                    random_state=RANDOM_SEED,
                ),
            ),
            category="Baseline",
            display_order=20,
            feature_filter=_rating_features,
        ),
        ModelSpec(
            "Logistic Regression",
            lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.3,
                    max_iter=2000,
                    multi_class="auto",
                    random_state=RANDOM_SEED,
                ),
            ),
            category="Baseline",
            display_order=30,
        ),
        ModelSpec(
            "Neural Net (MLP)",
            lambda: make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    alpha=0.01,
                    learning_rate_init=0.001,
                    early_stopping=True,
                    max_iter=300,
                    random_state=RANDOM_SEED,
                ),
            ),
            category="Neural network",
            display_order=100,
        ),
        ModelSpec(
            "Decision Tree",
            lambda: DecisionTreeClassifier(
                max_depth=5,
                min_samples_leaf=30,
                random_state=RANDOM_SEED,
            ),
            category="Tree-based",
            display_order=200,
        ),
        ModelSpec(
            "Random Forest",
            lambda: RandomForestClassifier(
                n_estimators=250,
                max_depth=8,
                min_samples_leaf=15,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_SEED,
            ),
            category="Tree-based",
            display_order=210,
        ),
        ModelSpec(
            "Extra Trees",
            lambda: ExtraTreesClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=15,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_SEED,
            ),
            category="Tree-based",
            display_order=220,
        ),
        ModelSpec(
            "Gradient Boosting tree",
            lambda: GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.04,
                max_depth=3,
                min_samples_leaf=20,
                random_state=RANDOM_SEED,
            ),
            category="Tree-based",
            display_order=230,
        ),
        ModelSpec(
            "AdaBoost tree",
            lambda: AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=2, min_samples_leaf=25),
                n_estimators=200,
                learning_rate=0.05,
                random_state=RANDOM_SEED,
            ),
            category="Tree-based",
            display_order=240,
        ),
        ModelSpec(
            "HistGradientBoosting",
            lambda: HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.04,
                max_leaf_nodes=31,
                l2_regularization=0.1,
                random_state=RANDOM_SEED,
            ),
            category="Tree-based",
            display_order=250,
        ),
    ]

    if include_slow:
        specs.extend(_optional_boosting_specs())

    return specs


def _optional_boosting_specs() -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    catboost_builder = None
    xgboost_builder = None
    lightgbm_builder = None

    try:
        from catboost import CatBoostClassifier

        catboost_builder = lambda: CatBoostClassifier(
            iterations=300,
            learning_rate=0.04,
            depth=5,
            l2_leaf_reg=8.0,
            loss_function="MultiClass",
            classes_count=3,
            random_seed=RANDOM_SEED,
            verbose=0,
            thread_count=-1,
        )
        specs.append(
            ModelSpec(
                "CatBoost",
                catboost_builder,
                category="Tree-based",
                display_order=260,
            )
        )
    except ImportError:
        logger.warning("CatBoost is not installed; skipping CatBoost benchmark")

    try:
        from xgboost import XGBClassifier

        xgboost_builder = lambda: XGBClassifier(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=3,
            min_child_weight=6,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=8.0,
            reg_alpha=1.0,
            objective="multi:softprob",
            num_class=3,
            random_state=RANDOM_SEED,
            verbosity=0,
            n_jobs=-1,
        )
        specs.append(
            ModelSpec(
                "XGBoost",
                xgboost_builder,
                category="Tree-based",
                display_order=270,
            )
        )
    except ImportError:
        logger.warning("XGBoost is not installed; skipping XGBoost benchmark")

    try:
        from lightgbm import LGBMClassifier

        lightgbm_builder = lambda: LGBMClassifier(
            n_estimators=300,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=5,
            min_child_samples=60,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            reg_lambda=5.0,
            reg_alpha=0.5,
            objective="multiclass",
            num_class=3,
            random_state=RANDOM_SEED,
            verbosity=-1,
            n_jobs=-1,
        )
        specs.append(
            ModelSpec(
                "LightGBM",
                lightgbm_builder,
                category="Tree-based",
                display_order=280,
            )
        )
    except ImportError:
        logger.warning("LightGBM is not installed; skipping LightGBM benchmark")

    if catboost_builder and xgboost_builder and lightgbm_builder:
        specs.append(
            ModelSpec(
                "Ultimate Ensemble (weighted soft vote)",
                lambda: VotingClassifier(
                    estimators=[
                        ("catboost", catboost_builder()),
                        ("xgboost", xgboost_builder()),
                        ("lightgbm", lightgbm_builder()),
                    ],
                    voting="soft",
                    weights=[0.45, 0.35, 0.20],
                    n_jobs=1,
                ),
                category="Ultimate ensemble",
                display_order=900,
            )
        )

    return specs


def _load_data() -> tuple[pd.DataFrame, pd.Series]:
    feature_path = PROCESSED_DATA_DIR / "feature_matrix.parquet"
    target_path = PROCESSED_DATA_DIR / "targets.parquet"
    if not feature_path.exists() or not target_path.exists():
        raise FileNotFoundError(
            "Missing processed benchmark inputs. Run `python main.py --mode full` "
            "or `python main.py --mode test` first."
        )

    X = pd.read_parquet(feature_path)
    y = pd.read_parquet(target_path)["wdl"].astype(int)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, y


def _splitter(cv: str, folds: int, y: pd.Series):
    if cv == "kfold":
        return StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=RANDOM_SEED,
        ).split(np.zeros(len(y)), y)
    return TimeSeriesSplit(n_splits=folds).split(np.zeros(len(y)))


def _normalize_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    proba = np.clip(proba, 1e-15, 1.0)
    row_sums = proba.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return proba / row_sums


def _predict_proba_aligned(model: BaseEstimator, X_val: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(X_val)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "steps"):
        classes = model.steps[-1][1].classes_
    classes = np.asarray(classes)

    aligned = np.zeros((len(X_val), len(CLASSES)))
    for raw_idx, cls in enumerate(classes):
        class_matches = np.where(CLASSES == cls)[0]
        if len(class_matches):
            aligned[:, class_matches[0]] = raw[:, raw_idx]

    return _normalize_proba(aligned)


def _safe_auc(y_true: pd.Series, proba: np.ndarray) -> float:
    try:
        return float(
            roc_auc_score(
                y_true,
                proba,
                labels=CLASSES,
                multi_class="ovr",
                average="micro",
            )
        )
    except ValueError:
        return float("nan")


def evaluate_model(spec: ModelSpec, X: pd.DataFrame, y: pd.Series, cv: str, folds: int) -> dict:
    X_model = spec.feature_filter(X) if spec.feature_filter else X
    y_true_all = []
    y_pred_all = []
    proba_all = []
    losses = []

    for fold, (train_idx, val_idx) in enumerate(_splitter(cv, folds, y), start=1):
        model = clone(spec.build())
        X_train, X_val = X_model.iloc[train_idx], X_model.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            model.fit(X_train, y_train)

        proba = _predict_proba_aligned(model, X_val)
        pred = CLASSES[np.argmax(proba, axis=1)]

        y_true_all.append(y_val.to_numpy())
        y_pred_all.append(pred)
        proba_all.append(proba)
        losses.append(log_loss(y_val, proba, labels=CLASSES))
        logger.info("  %s fold %s/%s logloss %.4f", spec.name, fold, folds, losses[-1])

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    proba_full = np.vstack(proba_all)

    return {
        "category": spec.category,
        "display_order": spec.display_order,
        "model": spec.name,
        "accuracy_pct": accuracy_score(y_true, y_pred) * 100,
        "f1_micro_pct": f1_score(y_true, y_pred, average="micro") * 100,
        "auroc_micro": _safe_auc(pd.Series(y_true), proba_full),
        "logloss": float(np.mean(losses)),
    }


def _format_markdown(results: pd.DataFrame, cv: str, folds: int) -> str:
    cv_label = "Stratified K-fold" if cv == "kfold" else "TimeSeriesSplit"
    lines = [
        f"Latest benchmark: `{folds}`-fold `{cv_label}` on `data/processed/feature_matrix.parquet`.",
        "",
        "| Category | Model | CV accuracy (%) | F1 micro (%) | AUROC micro | Logloss |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in results.iterrows():
        auc = "n/a" if pd.isna(row["auroc_micro"]) else f"{row['auroc_micro']:.3f}"
        lines.append(
            f"| {row['category']} | {row['model']} | {row['accuracy_pct']:.2f} | "
            f"{row['f1_micro_pct']:.2f} | {auc} | {row['logloss']:.4f} |"
        )
    return "\n".join(lines)


def _update_readme(markdown: str, readme_path: Path | None = None) -> None:
    readme_path = readme_path or PROJECT_ROOT / "README.md"
    if not readme_path.exists():
        logger.warning("README not found at %s; skipping README update", readme_path)
        return

    readme = readme_path.read_text(encoding="utf-8")
    block = f"{README_START}\n{markdown}\n{README_END}"

    if README_START in readme and README_END in readme:
        before = readme.split(README_START)[0]
        after = readme.split(README_END, 1)[1]
        updated = before + block + after
    else:
        insert_after = "## Modeling Caveats"
        section = "## Model Benchmark Results\n\n" + block + "\n\n"
        if insert_after in readme:
            updated = readme.replace(insert_after, section + insert_after, 1)
        else:
            updated = readme.rstrip() + "\n\n" + section

    readme_path.write_text(updated, encoding="utf-8")
    logger.info("Updated README benchmark table at %s", readme_path)


def _safe_csv_write(results: pd.DataFrame, path: Path) -> Path:
    try:
        results.to_csv(path, index=False)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
        results.to_csv(fallback, index=False)
        logger.warning("Could not write %s because it is locked; wrote %s instead", path, fallback)
        return fallback


def _safe_text_write(text: str, path: Path) -> Path:
    try:
        path.write_text(text, encoding="utf-8")
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
        fallback.write_text(text, encoding="utf-8")
        logger.warning("Could not write %s because it is locked; wrote %s instead", path, fallback)
        return fallback


def run_benchmark(
    cv: str = "time",
    folds: int = 5,
    include_slow: bool = True,
    update_readme: bool = True,
) -> pd.DataFrame:
    X, y = _load_data()
    logger.info("Loaded benchmark data: %s samples x %s features", X.shape[0], X.shape[1])

    records = []
    for spec in _build_model_specs(include_slow=include_slow):
        logger.info("Benchmarking %s", spec.name)
        records.append(evaluate_model(spec, X, y, cv=cv, folds=folds))

    results = (
        pd.DataFrame(records)
        .sort_values(["display_order", "logloss"], ascending=[True, True])
        .reset_index(drop=True)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"model_benchmark_{cv}_{folds}fold"
    csv_path = OUTPUT_DIR / f"{stem}.csv"
    md_path = OUTPUT_DIR / f"{stem}.md"
    markdown = _format_markdown(results, cv=cv, folds=folds)

    written_csv = _safe_csv_write(results, csv_path)
    written_md = _safe_text_write(markdown + "\n", md_path)
    logger.info("Saved benchmark CSV to %s", written_csv)
    logger.info("Saved benchmark Markdown to %s", written_md)

    if cv == "time" and folds == 5:
        _safe_csv_write(results, OUTPUT_DIR / "model_benchmark.csv")
        _safe_text_write(markdown + "\n", OUTPUT_DIR / "model_benchmark.md")

    if update_readme:
        _update_readme(markdown)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark base W/D/L models.")
    parser.add_argument("--cv", choices=["time", "kfold"], default="time")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fast", action="store_true", help="Skip CatBoost/XGBoost/LightGBM.")
    parser.add_argument("--no-readme", action="store_true", help="Do not update README.md.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_benchmark(
        cv=args.cv,
        folds=args.folds,
        include_slow=not args.fast,
        update_readme=not args.no_readme,
    )


if __name__ == "__main__":
    main()
