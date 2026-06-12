import pandas as pd
import logging
from pathlib import Path
import sys
import pdfplumber
import re

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
        
    def load_club_form_stats(self) -> dict:
        if not self.club_stats_path.exists():
            logger.warning(f"Club stats CSV not found at {self.club_stats_path}")
            return {}
            
        logger.info(f"Loading club stats from {self.club_stats_path}")
        df = pd.read_csv(self.club_stats_path)
        
        # Extract 3-letter code from "us USA" -> "USA"
        df['NationCode'] = df['Nation'].apply(lambda x: str(x).split(' ')[-1] if pd.notna(x) else None)
        
        # Fill NaNs with 0 for numeric columns
        for col in ['Gls', 'Ast', 'SoT', '+/-', 'Int', 'TklW', 'PPM', 'Saves', '+/-90']:
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
            
            return max(0, idx)
            
        df['FormIndex'] = df.apply(calc_index, axis=1)
        
        nation_power = {}
        for nation, group in df.groupby('NationCode'):
            top_15 = group.nlargest(15, 'FormIndex')
            nation_power[nation] = top_15['FormIndex'].sum()
            
        return nation_power

    def load_and_aggregate(self) -> pd.DataFrame:
        if not self.raw_path.exists():
            logger.warning(f"Official Squad PDF not found at {self.raw_path}")
            return pd.DataFrame()
            
        logger.info(f"Loading official squad lists from {self.raw_path}")
        
        club_form_map = self.load_club_form_stats()
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
                            
                            if team_name == "IR Iran": team_name = "Iran"
                            if team_name == "Korea Republic": team_name = "South Korea"
                            if team_name == "USA": team_name = "United States"
                            if team_name == "Cabo Verde": team_name = "Cape Verde"
                            if team_name == "Côte D'Ivoire": team_name = "Ivory Coast"
                            if team_name == "Türkiye": team_name = "Turkey"
                            if team_name == "Congo DR": team_name = "DR Congo"
                            if team_name == "Czechia": team_name = "Czech Republic"
                            
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
            
            # Add Squad Power Rating
            agg_df['squad_power_rating'] = (
                (agg_df['elite_players'] * 50.0) +
                (agg_df['total_caps'] * 0.5) +
                (agg_df['total_international_goals'] * 2.0)
            )
            
            # Add Club Form Power
            agg_df['club_form_power'] = agg_df['team_code'].map(club_form_map).fillna(0)
            
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
        print("\nTop 10 Nations by Club Form Power:")
        print(df.sort_values('club_form_power', ascending=False).head(10)[['team', 'club_form_power']].to_string(index=False))
