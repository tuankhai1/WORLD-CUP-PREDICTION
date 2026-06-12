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
    Parses the official FIFA World Cup 26 Squad Lists (PDF) to build
    a highly accurate "Squad Power Rating" for each nation.
    """
    def __init__(self):
        self.raw_path = RAW_DATA_DIR / "SquadLists-English.pdf"
        
    def load_and_aggregate(self) -> pd.DataFrame:
        if not self.raw_path.exists():
            logger.warning(f"Official Squad PDF not found at {self.raw_path}")
            return pd.DataFrame()
            
        logger.info(f"Loading official squad lists from {self.raw_path}")
        
        teams_data = []
        top_leagues = ['(ENG)', '(ESP)', '(ITA)', '(GER)', '(FRA)']
        
        try:
            with pdfplumber.open(self.raw_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if not text: continue
                    
                    lines = text.split('\n')
                    team_name = None
                    
                    # Extract the country name
                    for line in lines:
                        m = re.match(r"^(.*?)\s+\([A-Z]{3}\)$", line.strip())
                        if m:
                            team_name = m.group(1).strip()
                            # Resolve names like "Côte D'Ivoire" -> "Ivory Coast"
                            # "IR Iran" -> "Iran", "Korea Republic" -> "South Korea"
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
                    
                    # Parse the player rows
                    for line in lines:
                        # Matches lines ending with Height, Caps, Goals
                        m = re.search(r"(\d+)\s+(\d+)\s+(\d+)$", line)
                        if m:
                            try:
                                c = int(m.group(2))
                                g = int(m.group(3))
                                caps += c
                                goals += g
                                squad_size += 1
                                
                                # Check if playing in top 5 leagues
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
                            'elite_players': elite
                        })

            agg_df = pd.DataFrame(teams_data)
            
            # The new Squad Power Rating math!
            # Heavy weighting on Elite players + International Experience
            agg_df['squad_power_rating'] = (
                (agg_df['elite_players'] * 50.0) +
                (agg_df['total_caps'] * 0.5) +
                (agg_df['total_international_goals'] * 2.0)
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
        print("\nTop 10 Nations by OFFICIAL Squad Power Rating:")
        print(df.sort_values('squad_power_rating', ascending=False).head(10).to_string(index=False))
