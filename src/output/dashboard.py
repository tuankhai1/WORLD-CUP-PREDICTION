"""HTML dashboard generation for simulation and matchup predictions."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    ALL_TEAMS,
    DASHBOARD_TITLE,
    GROUPS,
    OUTPUT_DIR,
    PROCESSED_DATA_DIR,
    RECENCY_HALF_LIFE_DAYS,
    TRAINING_CUTOFF_DATE,
)


ROUND_COLUMNS = ["r32", "r16", "qf", "sf", "final", "winner"]
ROUND_LABELS = {
    "r32": "Round of 32",
    "r16": "Round of 16",
    "qf": "Quarterfinal",
    "sf": "Semifinal",
    "final": "Final",
    "winner": "Winner",
}


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
        squad_df = self._squad_frame()
        matchup_payload = self._matchup_payload(prob_matrix)
        team_names = self._team_names(results_df, matchup_payload)
        title = escape(DASHBOARD_TITLE)

        leader = results_df.iloc[0].to_dict() if not results_df.empty else {}
        leader_name = escape(str(leader.get("team", "No leader yet")))
        leader_prob = self._pct(leader.get("winner", 0), already_percent=True)
        avg_advance = self._pct(results_df["advance"].mean() if not results_df.empty else 0, already_percent=True)

        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  {self._style()}
</head>
<body>
  <header class="topbar">
    <div>
      <p class="eyebrow">Live tournament model</p>
      <h1>{title}</h1>
      <p class="subtitle">Monte Carlo probabilities, matchup explorer, projected knockout path, and squad-form signals in one matchday view.</p>
    </div>
    <div class="status-panel">
      <span>{len(results_df):,} teams</span>
      <span>{len(prob_matrix):,} matchups</span>
      <span>{len(match_predictions):,} group fixtures modeled</span>
    </div>
  </header>

  <main>
    <section class="kpi-grid" aria-label="Model summary">
      {self._kpi("Current favorite", leader_name, leader_prob)}
      {self._kpi("Average advance chance", avg_advance, "Across the tournament field")}
      {self._kpi("Training cutoff", escape(TRAINING_CUTOFF_DATE), "Current World Cup cycle only")}
      {self._kpi("Recency half-life", f"{RECENCY_HALF_LIFE_DAYS} days", "Recent results carry more weight")}
    </section>

    <section class="panel wide">
      <div class="section-head">
        <div>
          <p class="eyebrow">Head to head</p>
          <h2>Matchup Explorer</h2>
        </div>
        <p>Choose any two tournament teams to update win, draw, loss, expected goal differential, and likely score.</p>
      </div>
      {self._matchup_explorer(team_names)}
    </section>

    <section class="panel wide">
      <div class="section-head">
        <div>
          <p class="eyebrow">Projected path</p>
          <h2>Bracketology Road To The Final</h2>
        </div>
        <p>This deterministic bracket is seeded from projected group points and direct matchup probabilities. Replace this with the official FIFA third-place mapping when final bracket rules are confirmed.</p>
      </div>
      {self._bracketology(results_df, prob_matrix, match_predictions)}
    </section>

    <section class="panel wide">
      <div class="section-head">
        <div>
          <p class="eyebrow">Tournament outlook</p>
          <h2>Winner Probability</h2>
        </div>
        <p>Top contenders by simulated title probability.</p>
      </div>
      {self._winner_chart(results_df)}
    </section>

    <section class="grid-two">
      <section class="panel">
        <div class="section-head">
          <div>
            <p class="eyebrow">Path strength</p>
            <h2>Round Progression</h2>
          </div>
        </div>
        {self._advancement_chart(results_df)}
      </section>

      <section class="panel">
        <div class="section-head">
          <div>
            <p class="eyebrow">Model health</p>
            <h2>Training Metrics</h2>
          </div>
        </div>
        {self._metrics_grid(model_metrics)}
      </section>
    </section>

    <section class="panel wide">
      <div class="section-head">
        <div>
          <p class="eyebrow">Groups</p>
          <h2>Group Advancement</h2>
        </div>
        <p>Advancement probability is calculated as 100 minus simulated group-exit rate.</p>
      </div>
      {self._group_cards(results_df)}
    </section>

    <section class="grid-two">
      <section class="panel">
        <div class="section-head">
          <div>
            <p class="eyebrow">Ranking table</p>
            <h2>Top Teams</h2>
          </div>
        </div>
        {self._results_table(results_df)}
      </section>

      <section class="panel">
        <div class="section-head">
          <div>
            <p class="eyebrow">Squad signal</p>
            <h2>Club Form Power</h2>
          </div>
        </div>
        {self._squad_table(squad_df)}
      </section>
    </section>
  </main>
  {self._dashboard_script(matchup_payload)}
</body>
</html>"""

        path = self.output_dir / filename
        path.write_text(html, encoding="utf-8")
        return path

    @staticmethod
    def _style() -> str:
        return """<style>
    :root {
      color-scheme: light;
      --bg: #f7f9fb;
      --panel: #ffffff;
      --panel-alt: #eef5f0;
      --text: #17211b;
      --muted: #5f6f66;
      --line: #dbe5df;
      --green: #126b45;
      --green-soft: #dbeee4;
      --mint: #2f9d6a;
      --gold: #b58b12;
      --red: #b94848;
      --ink: #0a2b33;
      --shadow: 0 16px 36px rgba(24, 38, 30, 0.08);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .topbar {
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: flex-end;
      padding: 32px clamp(18px, 4vw, 56px) 24px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(135deg, #ffffff 0%, #edf6f1 100%);
    }
    h1, h2, h3, p { margin: 0; }
    h1 { font-size: clamp(2rem, 4vw, 3.7rem); line-height: 0.98; letter-spacing: 0; max-width: 860px; }
    h2 { font-size: 1.1rem; letter-spacing: 0; }
    h3 { font-size: 0.92rem; }
    .subtitle { color: var(--muted); margin-top: 12px; max-width: 720px; line-height: 1.55; }
    .eyebrow {
      color: var(--green);
      font-size: 0.72rem;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }
    .status-panel {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
      min-width: 240px;
    }
    .status-panel span, .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 11px;
      background: rgba(255, 255, 255, 0.7);
      color: var(--muted);
      font-size: 0.82rem;
      font-weight: 700;
      white-space: nowrap;
    }
    main { padding: 24px clamp(18px, 4vw, 56px) 48px; }
    .kpi-grid, .group-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }
    .grid-two {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr);
      gap: 18px;
      margin-bottom: 18px;
    }
    .panel, .kpi {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .panel { padding: 18px; overflow: hidden; }
    .wide { margin-bottom: 18px; }
    .kpi { padding: 18px; }
    .kpi .label { color: var(--muted); font-size: 0.78rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.06em; }
    .kpi .value { margin-top: 8px; font-size: 1.45rem; font-weight: 850; }
    .kpi .detail { margin-top: 6px; color: var(--muted); font-size: 0.88rem; }
    .section-head {
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: start;
      margin-bottom: 16px;
    }
    .section-head > p { color: var(--muted); max-width: 520px; line-height: 1.45; font-size: 0.9rem; }
    .control-grid {
      display: grid;
      grid-template-columns: minmax(180px, 1fr) minmax(180px, 1fr);
      gap: 12px;
      margin-bottom: 16px;
    }
    label { display: grid; gap: 7px; color: var(--muted); font-size: 0.78rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.06em; }
    select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 13px;
      background: #fbfcfd;
      color: var(--text);
      font: inherit;
      font-weight: 700;
    }
    .matchup-card {
      display: grid;
      grid-template-columns: minmax(220px, 0.8fr) minmax(0, 1.2fr);
      gap: 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      background: var(--panel-alt);
    }
    .match-title { font-size: 1.32rem; font-weight: 850; }
    .match-subtitle { color: var(--muted); margin-top: 8px; line-height: 1.45; }
    .prob-stack { display: grid; gap: 11px; }
    .prob-item { display: grid; grid-template-columns: 70px minmax(0, 1fr) 58px; gap: 10px; align-items: center; font-weight: 800; }
    .prob-track { height: 11px; border-radius: 999px; background: #dde8e1; overflow: hidden; }
    .prob-fill { display: block; height: 100%; border-radius: inherit; width: 0; transition: width 160ms ease; }
    .prob-fill.win { background: var(--green); }
    .prob-fill.draw { background: var(--gold); }
    .prob-fill.loss { background: var(--red); }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; min-width: 560px; }
    th, td { padding: 11px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: middle; }
    th { color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; }
    td { font-size: 0.92rem; }
    .numeric { text-align: right; font-variant-numeric: tabular-nums; }
    .bar {
      height: 8px;
      background: #e8eee9;
      border-radius: 999px;
      overflow: hidden;
      min-width: 84px;
    }
    .bar > span { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--mint), var(--green)); }
    .group-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: var(--panel-alt);
    }
    .group-card h3 { margin-bottom: 12px; }
    .team-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 54px;
      gap: 8px;
      align-items: center;
      margin: 10px 0;
      color: var(--muted);
      font-size: 0.88rem;
    }
    .team-row strong { color: var(--text); font-weight: 750; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfd;
    }
    .metric .name { color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; }
    .metric .number { margin-top: 6px; font-weight: 850; font-size: 1.12rem; }
    .empty {
      color: var(--muted);
      padding: 18px;
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: #fbfcfd;
    }
    .win { color: var(--green); font-weight: 800; }
    .draw { color: var(--gold); font-weight: 800; }
    .loss { color: var(--red); font-weight: 800; }
    .bracket-scroll { overflow-x: auto; padding-bottom: 4px; }
    .bracketology {
      min-width: 1120px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 210px minmax(0, 1fr);
      gap: 16px;
      align-items: center;
    }
    .bracket-half {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
      align-items: stretch;
    }
    .bracket-round { display: grid; gap: 9px; align-content: center; }
    .bracket-round h3 {
      color: var(--muted);
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 2px;
    }
    .bracket-card {
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 8px;
      padding: 8px;
      background: var(--ink);
      color: #f6fbff;
      min-height: 62px;
    }
    .bracket-card.compact { min-height: 48px; }
    .bracket-team {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      color: rgba(246,251,255,0.72);
      font-size: 0.78rem;
      line-height: 1.2;
    }
    .bracket-team strong { color: #fff; }
    .bracket-winner {
      margin-top: 7px;
      display: inline-flex;
      border-radius: 999px;
      padding: 3px 7px;
      background: rgba(47, 157, 106, 0.28);
      color: #dcffed;
      font-size: 0.73rem;
      font-weight: 850;
    }
    .final-column { display: grid; gap: 14px; align-content: center; }
    .champion-card {
      border-radius: 8px;
      padding: 18px;
      text-align: center;
      color: #fff;
      background: linear-gradient(160deg, var(--ink), #126b45);
      box-shadow: var(--shadow);
    }
    .champion-card .label { color: rgba(255,255,255,0.72); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; }
    .champion-card .name { margin-top: 8px; font-size: 1.45rem; font-weight: 900; }
    @media (max-width: 900px) {
      .topbar { align-items: flex-start; flex-direction: column; }
      .status-panel { justify-content: flex-start; }
      .grid-two, .matchup-card { grid-template-columns: 1fr; }
      .control-grid { grid-template-columns: 1fr; }
      .section-head { flex-direction: column; }
      table { min-width: 680px; }
    }
  </style>"""

    @staticmethod
    def _simulation_frame(simulation_results: dict[str, dict[str, float]]) -> pd.DataFrame:
        rows = [{"team": team, **probs} for team, probs in simulation_results.items()]
        cols = ["team", "group_exit", *ROUND_COLUMNS, "advance"]
        if not rows:
            return pd.DataFrame(columns=cols)

        df = pd.DataFrame(rows).fillna(0)
        for col in ["group_exit", *ROUND_COLUMNS]:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df["advance"] = (100.0 - df["group_exit"]).clip(lower=0, upper=100)
        return df.sort_values("winner", ascending=False).reset_index(drop=True)

    @staticmethod
    def _squad_frame() -> pd.DataFrame:
        path = PROCESSED_DATA_DIR / "squad_ratings.parquet"
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()
        if "club_form_power" not in df.columns:
            return pd.DataFrame()
        df["club_form_power"] = pd.to_numeric(df["club_form_power"], errors="coerce").fillna(0)
        return df.sort_values("club_form_power", ascending=False).reset_index(drop=True)

    @staticmethod
    def _pct(value: Any, already_percent: bool = False) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        if not already_percent and number <= 1.0:
            number *= 100
        return f"{number:.1f}%"

    @staticmethod
    def _bar(value: Any) -> str:
        try:
            width = max(0.0, min(100.0, float(value)))
        except (TypeError, ValueError):
            width = 0.0
        return f'<div class="bar"><span style="width:{width:.2f}%"></span></div>'

    @staticmethod
    def _kpi(label: str, value: str, detail: str = "") -> str:
        detail_html = f'<div class="detail">{escape(detail)}</div>' if detail else ""
        return (
            '<article class="kpi">'
            f'<div class="label">{escape(label)}</div>'
            f'<div class="value">{value}</div>'
            f"{detail_html}"
            "</article>"
        )

    @staticmethod
    def _matchup_payload(prob_matrix: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        payload = {}
        for key, probs in prob_matrix.items():
            try:
                payload[key] = {
                    "win": float(probs.get("win", 0)),
                    "draw": float(probs.get("draw", 0)),
                    "loss": float(probs.get("loss", 0)),
                    "xgd": float(probs.get("xgd", 0)),
                }
            except (TypeError, ValueError):
                continue
        return payload

    @staticmethod
    def _team_names(results_df: pd.DataFrame, matchup_payload: dict[str, dict[str, float]]) -> list[str]:
        teams = set(ALL_TEAMS)
        if not results_df.empty:
            teams.update(results_df["team"].dropna().astype(str))
        for key in matchup_payload:
            parts = key.split("|")
            if len(parts) == 2:
                teams.update(parts)
        return sorted(teams)

    def _matchup_explorer(self, team_names: list[str]) -> str:
        if len(team_names) < 2:
            return '<p class="empty">No matchup matrix available.</p>'

        default_a = "Spain" if "Spain" in team_names else team_names[0]
        default_b = "France" if "France" in team_names and "France" != default_a else team_names[1]
        options_a = self._select_options(team_names, default_a)
        options_b = self._select_options(team_names, default_b)
        return f"""
      <div class="control-grid">
        <label>Team A
          <select id="team-a">{options_a}</select>
        </label>
        <label>Team B
          <select id="team-b">{options_b}</select>
        </label>
      </div>
      <div class="matchup-card" aria-live="polite">
        <div>
          <div class="match-title" id="match-title">Select a matchup</div>
          <p class="match-subtitle" id="match-subtitle">Probabilities update instantly from the generated matchup matrix.</p>
        </div>
        <div class="prob-stack">
          {self._prob_row("win", "Team A win")}
          {self._prob_row("draw", "Draw")}
          {self._prob_row("loss", "Team B win")}
        </div>
      </div>"""

    @staticmethod
    def _select_options(team_names: list[str], selected: str) -> str:
        options = []
        for team in team_names:
            selected_attr = " selected" if team == selected else ""
            options.append(f'<option value="{escape(team)}"{selected_attr}>{escape(team)}</option>')
        return "".join(options)

    @staticmethod
    def _prob_row(kind: str, label: str) -> str:
        return (
            f'<div class="prob-item">'
            f'<span id="{kind}-name">{escape(label)}</span>'
            f'<div class="prob-track"><span id="{kind}-fill" class="prob-fill {kind}"></span></div>'
            f'<span id="{kind}-value" class="numeric">0.0%</span>'
            "</div>"
        )

    @staticmethod
    def _winner_chart(df: pd.DataFrame) -> str:
        if df.empty:
            return '<p class="empty">No simulation results available.</p>'

        chart_df = df.head(16).sort_values("winner", ascending=True).copy()
        chart_df["label"] = chart_df["winner"].map(lambda value: f"{value:.1f}%")
        fig = px.bar(
            chart_df,
            x="winner",
            y="team",
            orientation="h",
            text="label",
            color="winner",
            color_continuous_scale=["#dbeee4", "#2f9d6a", "#126b45"],
        )
        fig.update_traces(textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:.2f}%<extra></extra>")
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            xaxis_title="Title probability",
            yaxis_title=None,
            margin=dict(l=8, r=42, t=8, b=36),
            height=max(420, len(chart_df) * 32),
            font=dict(family="Inter, sans-serif", color="#17211b"),
        )
        return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False, "responsive": True})

    @staticmethod
    def _advancement_chart(df: pd.DataFrame) -> str:
        if df.empty:
            return '<p class="empty">No advancement results available.</p>'

        cols = [col for col in ROUND_COLUMNS if col in df.columns]
        chart_df = df.head(10).melt(id_vars="team", value_vars=cols, var_name="round", value_name="probability")
        chart_df["round"] = chart_df["round"].map(ROUND_LABELS)
        fig = px.line(chart_df, x="round", y="probability", color="team", markers=True)
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title=None,
            yaxis_title="Probability",
            yaxis_ticksuffix="%",
            legend_title=None,
            margin=dict(l=8, r=8, t=8, b=38),
            height=360,
            font=dict(family="Inter, sans-serif", color="#17211b"),
        )
        return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False, "responsive": True})

    def _group_cards(self, df: pd.DataFrame) -> str:
        if df.empty:
            return '<p class="empty">No group probabilities available.</p>'

        by_team = df.set_index("team").to_dict("index")
        cards = []
        for group_name, teams in GROUPS.items():
            team_rows = []
            ranked = sorted(
                teams,
                key=lambda team: by_team.get(team, {}).get("advance", 0),
                reverse=True,
            )
            for team in ranked:
                advance = by_team.get(team, {}).get("advance", 0)
                team_rows.append(
                    '<div class="team-row">'
                    f'<strong>{escape(team)}</strong>'
                    f'<span class="numeric">{self._pct(advance, already_percent=True)}</span>'
                    f'<div style="grid-column:1 / -1">{self._bar(advance)}</div>'
                    "</div>"
                )
            cards.append(
                '<article class="group-card">'
                f"<h3>Group {escape(group_name)}</h3>"
                f"{''.join(team_rows)}"
                "</article>"
            )
        return f'<div class="group-grid">{"".join(cards)}</div>'

    def _results_table(self, df: pd.DataFrame) -> str:
        if df.empty:
            return '<p class="empty">No results available.</p>'

        rows = []
        for idx, row in df.head(18).iterrows():
            rows.append(
                "<tr>"
                f'<td>{idx + 1}</td>'
                f'<td><strong>{escape(str(row["team"]))}</strong></td>'
                f'<td class="numeric">{self._pct(row["advance"], already_percent=True)}</td>'
                f'<td>{self._bar(row["winner"])}</td>'
                f'<td class="numeric win">{self._pct(row["winner"], already_percent=True)}</td>'
                "</tr>"
            )

        return (
            '<div class="table-wrap"><table>'
            "<thead><tr><th>Rank</th><th>Team</th><th>Advance</th><th>Winner bar</th><th>Winner</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table></div>"
        )

    @staticmethod
    def _format_metric(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"
        return escape(str(value))

    def _metrics_grid(self, metrics: dict[str, Any]) -> str:
        if not metrics:
            return '<p class="empty">No training metrics available for this run.</p>'

        items = []
        for key, value in sorted(metrics.items()):
            label = key.replace("_", " ").title()
            items.append(
                '<article class="metric">'
                f'<div class="name">{escape(label)}</div>'
                f'<div class="number">{self._format_metric(value)}</div>'
                "</article>"
            )
        return f'<div class="metric-grid">{"".join(items)}</div>'

    def _squad_table(self, df: pd.DataFrame) -> str:
        if df.empty:
            return '<p class="empty">No squad form table found. Run the player stats loader to generate squad ratings.</p>'

        rows = []
        for idx, row in df.head(14).iterrows():
            power = float(row.get("club_form_power", 0))
            rows.append(
                "<tr>"
                f'<td>{idx + 1}</td>'
                f'<td><strong>{escape(str(row.get("team", "")))}</strong></td>'
                f'<td class="numeric">{power:.1f}</td>'
                "</tr>"
            )
        return (
            '<div class="table-wrap"><table>'
            "<thead><tr><th>Rank</th><th>Team</th><th>Power</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table></div>"
        )

    def _bracketology(
        self,
        results_df: pd.DataFrame,
        prob_matrix: dict[str, dict[str, float]],
        match_predictions: list[dict[str, Any]],
    ) -> str:
        seeds = self._projected_seeds(results_df, match_predictions)
        if len(seeds) < 32:
            return '<p class="empty">Not enough projected qualifiers to build a bracket.</p>'

        r32_pairs = [(seeds[i], seeds[31 - i]) for i in range(16)]
        r32 = self._play_round(r32_pairs, prob_matrix, results_df)
        r16 = self._play_round(self._pair_winners(r32), prob_matrix, results_df)
        qf = self._play_round(self._pair_winners(r16), prob_matrix, results_df)
        sf = self._play_round(self._pair_winners(qf), prob_matrix, results_df)
        final = self._play_round(self._pair_winners(sf), prob_matrix, results_df)
        champion = final[0]["winner"] if final else {"team": "TBD", "label": ""}

        return (
            '<div class="bracket-scroll"><div class="bracketology">'
            f'<div class="bracket-half">{self._round_column("R32", r32[:8])}{self._round_column("R16", r16[:4])}{self._round_column("QF", qf[:2])}{self._round_column("SF", sf[:1])}</div>'
            '<div class="final-column">'
            f'{self._round_column("Final", final, compact=False)}'
            f'<div class="champion-card"><div class="label">Projected winner</div><div class="name">{escape(champion["team"])}</div></div>'
            '</div>'
            f'<div class="bracket-half">{self._round_column("SF", sf[1:])}{self._round_column("QF", qf[2:])}{self._round_column("R16", r16[4:])}{self._round_column("R32", r32[8:])}</div>'
            "</div></div>"
        )

    def _projected_seeds(self, results_df: pd.DataFrame, match_predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result_lookup = results_df.set_index("team").to_dict("index") if not results_df.empty else {}
        projected_points = {group: {team: 0.0 for team in teams} for group, teams in GROUPS.items()}

        for pred in match_predictions:
            group = str(pred.get("group", ""))
            if group not in projected_points:
                continue
            team_a = str(pred.get("team_a", ""))
            team_b = str(pred.get("team_b", ""))
            if team_a not in projected_points[group] or team_b not in projected_points[group]:
                continue
            win = float(pred.get("win_prob", 0))
            draw = float(pred.get("draw_prob", 0))
            loss = float(pred.get("loss_prob", 0))
            projected_points[group][team_a] += (3 * win) + draw
            projected_points[group][team_b] += (3 * loss) + draw

        top_two = []
        thirds = []
        for group, teams in GROUPS.items():
            standings = []
            for team in teams:
                sim = result_lookup.get(team, {})
                points = projected_points[group].get(team, 0.0)
                standings.append({
                    "team": team,
                    "group": group,
                    "points": points,
                    "advance": float(sim.get("advance", 0)),
                    "winner": float(sim.get("winner", 0)),
                })
            standings.sort(key=lambda row: (row["points"], row["advance"], row["winner"]), reverse=True)
            for pos, row in enumerate(standings, start=1):
                row["label"] = f"{pos}{group}"
                row["seed_score"] = (row["points"] * 10) + row["advance"] + (row["winner"] * 0.25)
            top_two.extend(standings[:2])
            thirds.append(standings[2])

        best_thirds = sorted(thirds, key=lambda row: (row["seed_score"], row["advance"]), reverse=True)[:8]
        qualifiers = top_two + best_thirds
        seen = {row["team"] for row in qualifiers}
        for _, row in results_df.sort_values("advance", ascending=False).iterrows():
            team = str(row["team"])
            if len(qualifiers) >= 32:
                break
            if team in seen:
                continue
            qualifiers.append({
                "team": team,
                "group": "",
                "points": 0.0,
                "advance": float(row.get("advance", 0)),
                "winner": float(row.get("winner", 0)),
                "label": "WC",
                "seed_score": float(row.get("advance", 0)),
            })
            seen.add(team)

        qualifiers.sort(key=lambda row: (row["seed_score"], row["winner"], row["advance"]), reverse=True)
        return qualifiers[:32]

    def _play_round(
        self,
        pairs: list[tuple[dict[str, Any], dict[str, Any]]],
        prob_matrix: dict[str, dict[str, float]],
        results_df: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        played = []
        for team_a, team_b in pairs:
            winner, chance = self._pick_winner(team_a["team"], team_b["team"], prob_matrix, results_df)
            winner_source = team_a if winner == team_a["team"] else team_b
            played.append({
                "team_a": team_a,
                "team_b": team_b,
                "winner": {"team": winner, "label": winner_source.get("label", "")},
                "chance": chance,
            })
        return played

    @staticmethod
    def _pair_winners(matches: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        winners = []
        for match in matches:
            winners.append(match["winner"])
        return [(winners[i], winners[i + 1]) for i in range(0, len(winners) - 1, 2)]

    @staticmethod
    def _pick_winner(
        team_a: str,
        team_b: str,
        prob_matrix: dict[str, dict[str, float]],
        results_df: pd.DataFrame,
    ) -> tuple[str, float]:
        direct = prob_matrix.get(f"{team_a}|{team_b}")
        if direct:
            win = float(direct.get("win", 0))
            loss = float(direct.get("loss", 0))
            return (team_a, win) if win >= loss else (team_b, loss)

        reverse = prob_matrix.get(f"{team_b}|{team_a}")
        if reverse:
            win = float(reverse.get("loss", 0))
            loss = float(reverse.get("win", 0))
            return (team_a, win) if win >= loss else (team_b, loss)

        lookup = results_df.set_index("team").to_dict("index") if not results_df.empty else {}
        score_a = float(lookup.get(team_a, {}).get("winner", 0))
        score_b = float(lookup.get(team_b, {}).get("winner", 0))
        total = score_a + score_b
        chance = (max(score_a, score_b) / total) if total else 0.5
        return (team_a, chance) if score_a >= score_b else (team_b, chance)

    def _round_column(self, title: str, matches: list[dict[str, Any]], compact: bool = True) -> str:
        cards = "".join(self._bracket_card(match, compact=compact) for match in matches)
        return f'<div class="bracket-round"><h3>{escape(title)}</h3>{cards}</div>'

    def _bracket_card(self, match: dict[str, Any], compact: bool = True) -> str:
        team_a = match["team_a"]
        team_b = match["team_b"]
        winner = match["winner"]
        compact_class = " compact" if compact else ""
        return (
            f'<div class="bracket-card{compact_class}">'
            f'{self._bracket_team(team_a)}'
            f'{self._bracket_team(team_b)}'
            f'<span class="bracket-winner">{escape(winner["team"])} {self._pct(match.get("chance", 0))}</span>'
            "</div>"
        )

    @staticmethod
    def _bracket_team(team: dict[str, Any]) -> str:
        return (
            '<div class="bracket-team">'
            f'<strong>{escape(str(team.get("team", "TBD")))}</strong>'
            f'<span>{escape(str(team.get("label", "")))}</span>'
            "</div>"
        )

    @staticmethod
    def _dashboard_script(matchup_payload: dict[str, dict[str, float]]) -> str:
        data = json.dumps(matchup_payload, ensure_ascii=True).replace("</", "<\\/")
        return f"""<script>
    const MATCHUP_MATRIX = {data};

    function pct(value) {{
      return `${{(Number(value || 0) * 100).toFixed(1)}}%`;
    }}

    function lookupMatchup(teamA, teamB) {{
      const direct = MATCHUP_MATRIX[`${{teamA}}|${{teamB}}`];
      if (direct) return direct;
      const reverse = MATCHUP_MATRIX[`${{teamB}}|${{teamA}}`];
      if (!reverse) return null;
      return {{
        win: reverse.loss,
        draw: reverse.draw,
        loss: reverse.win,
        xgd: -reverse.xgd
      }};
    }}

    function scoreline(xgd, win, draw, loss) {{
      if (draw > win && draw > loss) return Math.abs(xgd) < 0.8 ? "1-1" : "2-2";
      if (win >= loss) {{
        if (xgd < 0.5) return "1-0";
        if (xgd < 1.5) return "2-1";
        if (xgd < 2.5) return "2-0";
        return "3-1";
      }}
      if (xgd > -0.5) return "0-1";
      if (xgd > -1.5) return "1-2";
      if (xgd > -2.5) return "0-2";
      return "1-3";
    }}

    function setProb(kind, value) {{
      const fill = document.getElementById(`${{kind}}-fill`);
      const label = document.getElementById(`${{kind}}-value`);
      if (fill) fill.style.width = pct(value);
      if (label) label.textContent = pct(value);
    }}

    function renderMatchup() {{
      const teamA = document.getElementById("team-a")?.value;
      const teamB = document.getElementById("team-b")?.value;
      const title = document.getElementById("match-title");
      const subtitle = document.getElementById("match-subtitle");
      const winName = document.getElementById("win-name");
      const lossName = document.getElementById("loss-name");
      if (!teamA || !teamB || !title || !subtitle) return;

      if (winName) winName.textContent = `${{teamA}} win`;
      if (lossName) lossName.textContent = `${{teamB}} win`;

      if (teamA === teamB) {{
        title.textContent = `${{teamA}} vs ${{teamB}}`;
        subtitle.textContent = "Pick two different teams.";
        setProb("win", 0);
        setProb("draw", 0);
        setProb("loss", 0);
        return;
      }}

      const probs = lookupMatchup(teamA, teamB);
      title.textContent = `${{teamA}} vs ${{teamB}}`;
      if (!probs) {{
        subtitle.textContent = "No matchup entry found in the generated probability matrix.";
        setProb("win", 0);
        setProb("draw", 0);
        setProb("loss", 0);
        return;
      }}

      setProb("win", probs.win);
      setProb("draw", probs.draw);
      setProb("loss", probs.loss);
      subtitle.textContent = `Expected GD: ${{Number(probs.xgd || 0).toFixed(2)}} for ${{teamA}}. Likely score: ${{scoreline(probs.xgd, probs.win, probs.draw, probs.loss)}}.`;
    }}

    document.addEventListener("DOMContentLoaded", () => {{
      document.getElementById("team-a")?.addEventListener("change", renderMatchup);
      document.getElementById("team-b")?.addEventListener("change", renderMatchup);
      renderMatchup();
    }});
  </script>"""
