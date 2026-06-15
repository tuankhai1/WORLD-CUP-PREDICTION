"""HTML dashboard generation for simulation and matchup predictions."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DASHBOARD_TITLE, OUTPUT_DIR


class DashboardGenerator:
    """Generate a self-contained HTML dashboard from pipeline outputs."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        simulation_results: dict[str, dict[str, float]],
        match_predictions: list[dict[str, Any]] | None = None,
        model_metrics: dict[str, Any] | None = None,
        prob_matrix: dict[str, dict[str, float]] | None = None,
        filename: str = "dashboard.html",
    ) -> Path:
        """Write the dashboard and return its path."""
        match_predictions = match_predictions or []
        model_metrics = model_metrics or {}
        prob_matrix = prob_matrix or {}

        results_df = self._simulation_frame(simulation_results)
        winner_chart = self._winner_chart(results_df)
        advancement_chart = self._advancement_chart(results_df)

        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape(DASHBOARD_TITLE)}</title>
  <style>
    body {{ margin: 0; font-family: Inter, system-ui, sans-serif; background: #0a0a1a; color: #e0e0ff; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px; }}
    .card {{ background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.12); border-radius: 18px; padding: 20px; margin: 18px 0; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px; border-bottom: 1px solid rgba(255,255,255,.1); text-align: left; }}
    th {{ color: #ffd700; }}
    code, pre {{ background: rgba(0,0,0,.35); border-radius: 8px; padding: 2px 6px; }}
    .muted {{ color: #aaaacc; }}
  </style>
</head>
<body>
<main>
  <h1>{escape(DASHBOARD_TITLE)}</h1>
  <p class="muted">Generated from saved model predictions and Monte Carlo simulation output.</p>
  <section class="card"><h2>Winner probabilities</h2>{winner_chart}</section>
  <section class="card"><h2>Round advancement</h2>{advancement_chart}</section>
  <section class="card"><h2>Top teams</h2>{self._results_table(results_df)}</section>
  <section class="card"><h2>Model metrics</h2>{self._metrics_table(model_metrics)}</section>
  <section class="card"><h2>Sample group predictions</h2>{self._match_table(match_predictions)}</section>
  <section class="card"><h2>Prediction matrix</h2><p>{len(prob_matrix):,} matchup probability entries loaded.</p></section>
</main>
</body>
</html>"""
        path = self.output_dir / filename
        path.write_text(html, encoding="utf-8")
        return path

    @staticmethod
    def _simulation_frame(simulation_results: dict[str, dict[str, float]]) -> pd.DataFrame:
        rows = [{"team": team, **probs} for team, probs in simulation_results.items()]
        if not rows:
            return pd.DataFrame(columns=["team", "r32", "r16", "qf", "sf", "final", "winner"])
        return pd.DataFrame(rows).fillna(0).sort_values("winner", ascending=False)

    @staticmethod
    def _winner_chart(df: pd.DataFrame) -> str:
        if df.empty:
            return "<p>No simulation results available.</p>"
        fig = px.bar(df.head(16), x="team", y="winner", title="Top winner probabilities (%)")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    @staticmethod
    def _advancement_chart(df: pd.DataFrame) -> str:
        if df.empty:
            return "<p>No simulation results available.</p>"
        cols = [c for c in ["r32", "r16", "qf", "sf", "final", "winner"] if c in df.columns]
        long_df = df.head(12).melt(id_vars="team", value_vars=cols, var_name="round", value_name="probability")
        fig = px.line(long_df, x="round", y="probability", color="team", markers=True, title="Advancement probabilities (%)")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig.to_html(full_html=False, include_plotlyjs=False)

    @staticmethod
    def _results_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "<p>No results available.</p>"
        cols = [c for c in ["team", "r32", "r16", "qf", "sf", "final", "winner"] if c in df.columns]
        return df[cols].head(20).to_html(index=False, classes="results", border=0, escape=True)

    @staticmethod
    def _metrics_table(metrics: dict[str, Any]) -> str:
        if not metrics:
            return "<p>No training metrics available.</p>"
        df = pd.DataFrame([{"metric": k, "value": v} for k, v in sorted(metrics.items())])
        return df.to_html(index=False, border=0, escape=True)

    @staticmethod
    def _match_table(match_predictions: list[dict[str, Any]]) -> str:
        if not match_predictions:
            return "<p>No match predictions available.</p>"
        df = pd.DataFrame(match_predictions[:24])
        return df.to_html(index=False, border=0, escape=True)
