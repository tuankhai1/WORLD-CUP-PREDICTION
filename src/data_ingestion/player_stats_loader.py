import pandas as pd
import logging
from pathlib import Path
import sys
import pdfplumber
import re
from difflib import get_close_matches

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RAW_DATA_DIR, resolve_team_name

logger = logging.getLogger(__name__)

class PlayerStatsLoader:
    """
    Parses the official FIFA World Cup 26 Squad Lists (PDF) and the Top 5 Leagues CSV
    to build a highly accurate "Squad Power Rating" and "Club Form Power" for each nation.
    """
    def __init__(self):
        self.raw_path = RAW_DATA_DIR / "SquadLists-English.pdf"
        self.club_stats_path = RAW_DATA_DIR / "players_data-2025_2026.csv"
        self.club_stats_dir = RAW_DATA_DIR / "player_stats"
        self.squad_players_path = RAW_DATA_DIR / "squad_players.csv"
        self._club_stats_cache: pd.DataFrame | None = None
        

    @staticmethod
    def _norm_name(name: str) -> str:
        """Normalize player names for deterministic/fuzzy roster joins."""
        return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()

    @staticmethod
    def _canonicalize_stats_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize common provider column names into the internal schema."""
        aliases = {
            "Player": ["player", "player_name", "name", "Name"],
            "Nation": ["nationality", "Nationality", "country", "Country"],
            "Pos": ["position", "Position"],
            "Squad": ["team", "Team", "club", "Club"],
            "Comp": ["league", "League", "competition", "Competition"],
            "Gls": ["goals", "Goals", "G"],
            "Ast": ["assists", "Assists", "A"],
            "SoT": ["shots_on_target", "Shots On Target", "Shots on Target"],
            "+/-": ["plus_minus", "PlusMinus", "Plus/Minus"],
            "Int": ["interceptions", "Interceptions"],
            "TklW": ["tackles_won", "Tackles Won", "TacklesWon"],
            "PPM": ["points_per_match", "Points Per Match", "Pts/MP"],
            "Saves": ["saves", "Save", "Saves_stats_keeper"],
            "+/-90": ["plus_minus_per90", "PlusMinus90", "+/-_per90"],
            "Min": ["minutes", "Minutes"],
            "90s": ["nineties", "Nineties"],
        }

        rename_map = {}
        lower_to_original = {str(col).lower(): col for col in df.columns}
        for target, candidates in aliases.items():
            if target in df.columns:
                continue
            for candidate in candidates:
                source = lower_to_original.get(str(candidate).lower())
                if source is not None:
                    rename_map[source] = target
                    break

        if rename_map:
            df = df.rename(columns=rename_map)
        return df

    def _club_stats_paths(self) -> list[Path]:
        """Return all configured player-stat CSV paths without duplicates."""
        paths: list[Path] = []
        if self.club_stats_path.exists():
            paths.append(self.club_stats_path)
        if self.club_stats_dir.exists():
            paths.extend(sorted(self.club_stats_dir.glob("*.csv")))

        seen = set()
        unique_paths = []
        for path in paths:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            unique_paths.append(path)
        return unique_paths

    def _load_club_stats(self) -> pd.DataFrame:
        """Load and merge one or more current-season player-stat exports."""
        if self._club_stats_cache is not None:
            return self._club_stats_cache.copy()

        frames = []
        for path in self._club_stats_paths():
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                logger.warning(f"Could not read player stats CSV {path}: {exc}")
                continue
            df = self._canonicalize_stats_columns(df)
            df["_source_file"] = path.name
            frames.append(df)
            logger.info(f"Loaded {len(df)} player stat rows from {path}")

        if not frames:
            logger.warning(
                f"No player stat CSVs found at {self.club_stats_path} "
                f"or in {self.club_stats_dir}"
            )
            return pd.DataFrame()

        stats = pd.concat(frames, ignore_index=True, sort=False)
        dedupe_cols = [c for c in ["Player", "Nation", "Squad", "Comp", "Min"] if c in stats.columns]
        if dedupe_cols:
            before = len(stats)
            stats = stats.drop_duplicates(subset=dedupe_cols, keep="last")
            if len(stats) != before:
                logger.info(f"Dropped {before - len(stats)} duplicate player stat rows")
        self._club_stats_cache = stats
        return stats.copy()

    @staticmethod
    def _nation_keys(value: str) -> list[str]:
        """Return possible country-code/name keys for a provider nationality value."""
        raw = str(value).strip()
        if not raw or raw.lower() == "nan":
            return []

        keys = {resolve_team_name(raw)}
        parts = raw.split()
        if parts:
            last = parts[-1].upper()
            if len(last) == 3:
                keys.add(last)
                keys.add(resolve_team_name(last))
        return [key for key in keys if key]

    def _calc_player_form_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a position-aware form index to a player-season stats frame."""
        for col in ['Gls', 'Ast', 'SoT', '+/-', 'Int', 'TklW', 'PPM', 'Saves', '+/-90', 'Min', '90s']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0

        def calc_index(row):
            pos = str(row.get('Pos', ''))
            ppm = row['PPM']
            idx = 0
            if 'FW' in pos:
                idx = (row['Gls'] * 3) + (row['Ast'] * 2) + (row['SoT'] * 0.5) + (row['+/-'] * 0.5)
            elif 'MF' in pos:
                idx = (row['Ast'] * 3) + (row['Int'] * 0.5) + (row['TklW'] * 0.5) + (ppm * 2)
            elif 'DF' in pos:
                idx = (row['TklW'] * 1.5) + (row['Int'] * 1.5) + (ppm * 3) + (row['+/-'] * 1)
            elif 'GK' in pos:
                idx = (row['Saves'] * 0.5) + (ppm * 3) + (row['+/-90'] * 5)
            else:
                idx = (row['Gls'] * 1.5) + (row['Ast'] * 1.5) + (row['TklW'] * 0.5) + (ppm * 2)

            minutes = row['Min'] if row['Min'] else row['90s'] * 90
            minutes_factor = min(1.0, max(0.15, minutes / 900)) if minutes else 0.15
            return max(0, idx) * minutes_factor

        df['FormIndex'] = df.apply(calc_index, axis=1)
        return df

    def load_roster_aware_club_form_stats(self) -> dict:
        """Compute squad form by joining official squad players to season stats.

        Expected optional file: data/raw/squad_players.csv with columns such as
        team/team_code, player_name/name/player, club, and position/pos. If this
        roster table is absent, the loader falls back to nationality-pool form.
        """
        stats = self._load_club_stats()
        if not self.squad_players_path.exists() or stats.empty:
            return {}

        roster = pd.read_csv(self.squad_players_path)
        roster_player_col = next((c for c in ['player_name', 'Player', 'player', 'Name', 'name'] if c in roster.columns), None)
        stats_player_col = next((c for c in ['Player', 'player_name', 'player', 'Name', 'name'] if c in stats.columns), None)
        team_col = next((c for c in ['team', 'Team', 'nation', 'Nation'] if c in roster.columns), None)
        if not roster_player_col or not stats_player_col or not team_col:
            logger.warning("squad_players.csv missing required player/team columns; using nationality-pool form.")
            return {}

        stats = self._calc_player_form_index(stats)
        stats['_norm_name'] = stats[stats_player_col].apply(self._norm_name)
        roster['_norm_name'] = roster[roster_player_col].apply(self._norm_name)
        stat_names = stats['_norm_name'].dropna().unique().tolist()

        matched_rows = []
        for _, squad_player in roster.iterrows():
            norm = squad_player['_norm_name']
            candidates = stats[stats['_norm_name'] == norm]
            if candidates.empty:
                matches = get_close_matches(norm, stat_names, n=1, cutoff=0.88)
                candidates = stats[stats['_norm_name'] == matches[0]] if matches else pd.DataFrame()
            if candidates.empty:
                matched_rows.append({
                    'team': resolve_team_name(str(squad_player[team_col])),
                    'player_name': squad_player[roster_player_col],
                    'player_form': 0.0,
                    'matched': False,
                })
                continue
            best = candidates.sort_values('FormIndex', ascending=False).iloc[0]
            matched_rows.append({
                'team': resolve_team_name(str(squad_player[team_col])),
                'player_name': squad_player[roster_player_col],
                'player_form': float(best['FormIndex']),
                'matched': True,
            })

        matched = pd.DataFrame(matched_rows)
        out_path = RAW_DATA_DIR.parent / "processed" / "squad_player_form.parquet"
        matched.to_parquet(out_path, index=False)

        squad_form = {}
        for team, group in matched.groupby('team'):
            ranked = group.sort_values('player_form', ascending=False)
            top_xi = ranked.head(11)['player_form'].mean() if not ranked.empty else 0.0
            depth = ranked.head(23).tail(12)['player_form'].mean() if len(ranked) > 11 else 0.0
            squad_form[team] = (0.65 * top_xi) + (0.35 * depth)
        logger.info(f"Computed roster-aware squad form for {len(squad_form)} nations.")
        return squad_form

    def load_club_form_stats(self) -> dict:
        df = self._load_club_stats()
        if df.empty:
            return {}

        if "Nation" not in df.columns:
            logger.warning("Player stats are missing a Nation/Nationality column.")
            return {}
        
        # Extract 3-letter code from "us USA" -> "USA"
        df['NationKey'] = df['Nation'].apply(self._nation_keys)
        
        df = self._calc_player_form_index(df)

        df = df.explode('NationKey')
        df = df[df['NationKey'].notna() & (df['NationKey'] != "")]

        nation_power = {}
        for nation, group in df.groupby('NationKey'):
            top_15 = group.nlargest(15, 'FormIndex')
            nation_power[nation] = top_15['FormIndex'].sum()
            
        return nation_power

    def load_and_aggregate(self) -> pd.DataFrame:
        if not self.raw_path.exists():
            logger.warning(f"Official Squad PDF not found at {self.raw_path}")
            return pd.DataFrame()
            
        logger.info(f"Loading official squad lists from {self.raw_path}")
        
        club_form_map = self.load_roster_aware_club_form_stats() or self.load_club_form_stats()
        teams_data = []
        top_leagues = ['(ENG)', '(ESP)', '(ITA)', '(GER)', '(FRA)']
        
        try:
            with pdfplumber.open(self.raw_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if not text: continue
                    
                    lines = text.split('\n')
                    team_name = None
                    team_code = None
                    
                    # Extract the country name
                    for line in lines:
                        m = re.match(r"^(.*?)\s+\([A-Z]{3}\)$", line.strip())
                        if m:
                            team_name = m.group(1).strip()
                            team_code = line.strip()[-4:-1] # Extract the 3-letter code
                            
                            aliases = {
                                "IR Iran": "Iran",
                                "Korea Republic": "South Korea",
                                "USA": "United States",
                                "Cabo Verde": "Cape Verde",
                                "Cote D'Ivoire": "Ivory Coast",
                                "Côte D'Ivoire": "Ivory Coast",
                                "Turkiye": "Turkey",
                                "Türkiye": "Turkey",
                                "Congo DR": "DR Congo",
                                "Czechia": "Czech Republic",
                            }
                            team_name = aliases.get(team_name, team_name)
                            
                            team_name = resolve_team_name(team_name)
                            break
                            
                    if not team_name:
                        continue
                        
                    caps = 0
                    goals = 0
                    elite = 0
                    squad_size = 0
                    
                    for line in lines:
                        m = re.search(r"(\d+)\s+(\d+)\s+(\d+)$", line)
                        if m:
                            try:
                                c = int(m.group(2))
                                g = int(m.group(3))
                                caps += c
                                goals += g
                                squad_size += 1
                                
                                if any(l in line for l in top_leagues):
                                    elite += 1
                            except:
                                pass
                                
                    if squad_size > 0:
                        teams_data.append({
                            'team': team_name,
                            'squad_size': squad_size,
                            'total_caps': caps,
                            'total_international_goals': goals,
                            'elite_players': elite,
                            'team_code': team_code
                        })

            agg_df = pd.DataFrame(teams_data)
            
            # Note: Legacy squad_power_rating using total_caps and total_international_goals 
            # has been completely removed to prevent historical bias favoring older players.
            
            # Add Club Form Power. Stats providers may key by FIFA code (ARG)
            # or full country name (Argentina), so try both.
            agg_df['club_form_power'] = agg_df.apply(
                lambda row: club_form_map.get(
                    row.get('team_code'),
                    club_form_map.get(row.get('team'), 0.0),
                ),
                axis=1,
            )
            
            logger.info(f"Generated OFFICIAL squad power ratings for {len(agg_df)} nations.")
            
            out_path = RAW_DATA_DIR.parent / "processed" / "squad_ratings.parquet"
            agg_df.to_parquet(out_path, index=False)
            logger.info(f"Saved squad power ratings to {out_path}")
            
            return agg_df
            
        except Exception as e:
            logger.error(f"Failed to process squad lists: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    loader = PlayerStatsLoader()
    df = loader.load_and_aggregate()
    if not df.empty:
        top_teams = df.sort_values("club_form_power", ascending=False).head(10)
        logger.info(
            "Top 10 nations by club form power:\n%s",
            top_teams[["team", "club_form_power"]].to_string(index=False),
        )
