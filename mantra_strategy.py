"""
Mantra Fantacalcio Strategy Engine
Side Event Mantra 25/26 Season

Enhanced version of the original Fantacalcio.py with:
- Module-based analysis (4-2-3-1, 4-3-3, 4-4-2)
- Tier-based scarcity analysis
- Polyvalent player bonuses
- Auction strategy optimization
"""

import ast
import os
import yaml
from datetime import datetime
from random import randint
import time
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import pandas as pd
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

ruoli = ["Portieri", "Difensori", "Centrocampisti", "Trequartisti", "Attaccanti"]
role_mapping = {
    "Portieri": "P",
    "Difensori": "D", 
    "Centrocampisti": "C",
    "Trequartisti": "T",
    "Attaccanti": "A"
}

skills = {
    "Fuoriclasse": 1,
    "Titolare": 3,
    "Buona Media": 2,
    "Goleador": 4,
    "Assistman": 2,
    "Piazzati": 2,
    "Rigorista": 5,
    "Giovane talento": 2,
    "Panchinaro": -4,
    "Falloso": -2,
    "Outsider": 2,
}

class MantraStrategyEngine:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.df = None
        self.tier_data = {}
        
    def load_data(self, csv_path='giocatori.csv'):
        """Load player data from CSV"""
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(self.df)} players from {csv_path}")
        else:
            logger.error(f"Data file {csv_path} not found. Run data collection first.")
            return False
        return True
    
    def create_tier_system(self):
        """Create S/A/B/C tier system based on Quotazione quantiles"""
        if self.df is None:
            logger.error("No data loaded")
            return
            
        # Clean quotazione data
        quotazioni = pd.to_numeric(self.df['Quotazione'], errors='coerce')
        quotazioni = quotazioni.dropna()
        
        # Calculate quantiles
        q90 = quotazioni.quantile(0.90)  # S tier (top 10%)
        q65 = quotazioni.quantile(0.65)  # A tier (65-90%)
        q25 = quotazioni.quantile(0.25)  # B tier (25-65%)
        # C tier is bottom 25%
        
        # Assign tiers
        def assign_tier(quotazione):
            if pd.isna(quotazione):
                return 'C'
            if quotazione >= q90:
                return 'S'
            elif quotazione >= q65:
                return 'A'
            elif quotazione >= q25:
                return 'B'
            else:
                return 'C'
        
        self.df['Tier'] = self.df['Quotazione'].apply(assign_tier)
        
        # Store tier statistics
        self.tier_data = {
            'S_threshold': q90,
            'A_threshold': q65,
            'B_threshold': q25,
            'tier_counts': self.df['Tier'].value_counts()
        }
        
        logger.info(f"Tier system created: S≥{q90}, A≥{q65}, B≥{q25}")
        
    def analyze_module_scarcity(self, module_name="4-2-3-1"):
        """Analyze player scarcity for each role in the specified module"""
        if self.df is None:
            logger.error("No data loaded")
            return None
            
        module_config = None
        if module_name == self.config['modules']['primary']['name']:
            module_config = self.config['modules']['primary']
        else:
            for backup in self.config['modules']['backups']:
                if backup['name'] == module_name:
                    module_config = backup
                    break
                    
        if not module_config:
            logger.error(f"Module {module_name} not found in config")
            return None
            
        scarcity_analysis = {}
        
        for role_name, role_code in role_mapping.items():
            role_players = self.df[self.df['Ruolo'] == role_name].copy()
            required_count = module_config['roles'].get(role_code, 0)
            
            if len(role_players) == 0:
                continue
                
            # Analyze by tier
            tier_analysis = {}
            for tier in ['S', 'A', 'B', 'C']:
                tier_players = role_players[role_players['Tier'] == tier]
                tier_analysis[tier] = {
                    'count': len(tier_players),
                    'avg_quotazione': tier_players['Quotazione'].mean() if len(tier_players) > 0 else 0,
                    'top_players': tier_players.nlargest(3, 'Convenienza')['Nome'].tolist() if len(tier_players) > 0 else []
                }
            
            scarcity_analysis[role_code] = {
                'total_available': len(role_players),
                'required': required_count,
                'scarcity_ratio': required_count / len(role_players) if len(role_players) > 0 else float('inf'),
                'by_tier': tier_analysis,
                'avg_convenience': role_players['Convenienza'].mean()
            }
            
        return scarcity_analysis
    
    def calculate_polyvalence_bonus(self):
        """Calculate polyvalence bonuses for players"""
        if self.df is None:
            return
            
        # Define polyvalent mappings (players who can play multiple roles)
        polyvalent_mappings = {
            'Centrocampisti': ['T'],  # Some midfielders can play as attacking midfielders
            'Trequartisti': ['C', 'A'],  # Attacking midfielders can play central or as forwards
            'Difensori': ['C'],  # Some defenders can play defensive midfield
        }
        
        def calculate_bonus(row):
            base_role = row['Ruolo']
            bonus = 0
            
            # Check if player can fill multiple roles in our modules
            possible_roles = [role_mapping[base_role]]
            if base_role in polyvalent_mappings:
                possible_roles.extend(polyvalent_mappings[base_role])
            
            # Multi-role bonus
            if len(possible_roles) > 1:
                bonus += self.config['polyvalence']['multi_role_bonus']
            
            # Check utility in backup modules
            primary_module = self.config['modules']['primary']
            backup_modules = self.config['modules']['backups']
            
            useful_modules = 0
            for module in [primary_module] + backup_modules:
                for role in possible_roles:
                    if role in module['roles'] and module['roles'][role] > 0:
                        useful_modules += 1
                        break
            
            if useful_modules > 1:
                bonus += self.config['polyvalence']['flexibility_bonus']
            
            return bonus
        
        self.df['Polyvalence_Bonus'] = self.df.apply(calculate_bonus, axis=1)
        
    def calculate_enhanced_value(self):
        """Calculate enhanced player value including polyvalence and tier adjustments"""
        if self.df is None:
            return
            
        # Base convenience score
        base_convenience = self.df['Convenienza'].copy()
        
        # Apply polyvalence bonus
        enhanced_value = base_convenience * (1 + self.df['Polyvalence_Bonus'])
        
        # Tier-based adjustments
        tier_multipliers = {'S': 1.2, 'A': 1.1, 'B': 1.0, 'C': 0.9}
        tier_bonus = self.df['Tier'].map(tier_multipliers)
        enhanced_value *= tier_bonus
        
        self.df['Enhanced_Value'] = enhanced_value
        
    def generate_budget_allocation(self):
        """Generate optimal budget allocation strategy"""
        total_budget = self.config['auction']['total_budget']
        allocation = self.config['budget_allocation']
        
        budget_plan = {}
        for role, percentage in allocation.items():
            budget_plan[role] = {
                'budget': total_budget * percentage,
                'percentage': percentage * 100
            }
            
        return budget_plan
    
    def suggest_bidding_strategy(self, player_name, estimated_value):
        """Suggest bidding strategy with anti-parity rule"""
        max_bid = estimated_value * (1 + self.config['bidding']['max_overbid_percent'])
        
        # Anti-parity rule: +1 on round numbers
        if self.config['bidding']['anti_parity_rule']:
            if estimated_value == int(estimated_value):
                estimated_value += 1
                
        return {
            'suggested_bid': estimated_value,
            'max_bid': max_bid,
            'strategy': 'aggressive' if estimated_value > max_bid * 0.8 else 'conservative'
        }
    
    def create_auction_target_list(self, tier_preference=['S', 'A'], max_players=50):
        """Create prioritized target list for auction"""
        if self.df is None:
            logger.error("No data loaded")
            return None
            
        # Filter by preferred tiers
        filtered_df = self.df[self.df['Tier'].isin(tier_preference)].copy()
        
        # Sort by enhanced value
        target_list = filtered_df.nlargest(max_players, 'Enhanced_Value')
        
        # Add bidding recommendations
        target_list['Bidding_Strategy'] = target_list.apply(
            lambda row: self.suggest_bidding_strategy(row['Nome'], row['Enhanced_Value']), 
            axis=1
        )
        
        return target_list[['Nome', 'Ruolo', 'Tier', 'Quotazione', 'Enhanced_Value', 
                          'Polyvalence_Bonus', 'Bidding_Strategy']]
    
    def run_full_analysis(self, data_file='giocatori.csv'):
        """Run complete Mantra strategy analysis"""
        logger.info("Starting Mantra strategy analysis...")
        
        # Load data
        if not self.load_data(data_file):
            return False
        
        # Create tier system
        self.create_tier_system()
        
        # Calculate polyvalence bonuses
        self.calculate_polyvalence_bonus()
        
        # Calculate enhanced values
        self.calculate_enhanced_value()
        
        # Generate reports
        report_dir = self.generate_reports()
        
        logger.info(f"Analysis complete! Reports saved to {report_dir}")
        return report_dir

    def generate_reports(self, output_dir=None):
        """Generate comprehensive analysis reports"""
        if output_dir is None:
            output_dir = f"reports/{datetime.now().strftime('%Y-%m-%d')}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate summary CSV
        if self.df is not None:
            summary_data = self.df[['Nome', 'Ruolo', 'Squadra', 'Tier', 'Quotazione', 
                                   'Enhanced_Value', 'Polyvalence_Bonus', 'Convenienza']].copy()
            summary_data.to_csv(os.path.join(output_dir, "enhanced_player_data.csv"), index=False)
            
            # Create target list
            target_list = self.create_auction_target_list()
            if target_list is not None:
                target_list.to_csv(os.path.join(output_dir, "auction_target_list.csv"), index=False)
        
        # Generate module analysis
        scarcity_report = self.analyze_module_scarcity()
        if scarcity_report:
            scarcity_df = pd.DataFrame.from_dict(scarcity_report, orient='index')
            scarcity_df.to_csv(os.path.join(output_dir, "module_scarcity_analysis.csv"))
        
        # Generate budget allocation
        budget_plan = self.generate_budget_allocation()
        budget_df = pd.DataFrame.from_dict(budget_plan, orient='index')
        budget_df.to_csv(os.path.join(output_dir, "budget_allocation.csv"))
        
        logger.info(f"Reports generated in {output_dir}")
        return output_dir

# Original scraping functions (unchanged)
def get_giocatori(ruolo: str) -> list:
    """Scrape player URLs from fantacalciopedia"""
    html = requests.get(
        "https://www.fantacalciopedia.com/lista-calciatori-serie-a/"
        + ruolo.lower()
        + "/"
    )
    soup = BeautifulSoup(html.content, "html.parser")
    calciatori = []
    giocatori = soup.find_all("article")
    for giocatore in giocatori:
        calciatore = giocatore.find("a").get("href")
        calciatori.append(calciatore)
    return calciatori

def get_attributi(url: str) -> dict:
    """Scrape player attributes from fantacalciopedia"""
    time.sleep(randint(0, 2000) / 1000)
    attributi = dict()
    html = requests.get(url.strip())
    soup = BeautifulSoup(html.content, "html.parser")
    attributi["Nome"] = soup.select_one("h1").get_text().strip()

    selettore = "div.col_one_fourth:nth-of-type(1) span.stickdan"
    attributi["Punteggio"] = soup.select_one(selettore).text.strip().replace("/100", "")

    selettore = "\tdiv.col_one_fourth:nth-of-type(n+2) div"
    medie = [el.find("span").text.strip() for el in soup.select(selettore)]
    anni = [
        el.find("strong").text.split(" ")[-1].strip() for el in soup.select(selettore)
    ]
    i = 0
    for anno in anni:
        attributi[f"Fantamedia anno {anno}"] = medie[i]
        i += 1

    selettore = "div.col_one_third:nth-of-type(2) div"
    stats_ultimo_anno = soup.select_one(selettore)
    parametri = [
        el.text.strip().replace(":", "") for el in stats_ultimo_anno.find_all("strong")
    ]
    valori = [el.text.strip() for el in stats_ultimo_anno.find_all("span")]
    attributi.update(dict(zip(parametri, valori)))

    selettore = ".col_one_third.col_last div"
    stats_previste = soup.select_one(selettore)
    parametri = [
        el.text.strip().replace(":", "") for el in stats_previste.find_all("strong")
    ]
    valori = [el.text.strip() for el in stats_previste.find_all("span")]
    attributi.update(dict(zip(parametri, valori)))

    selettore = ".label12 span.label"
    ruolo = soup.select_one(selettore)
    attributi["Ruolo"] = ruolo.get_text().strip()

    selettore = "span.stickdanpic"
    skills_list = [el.text for el in soup.select(selettore)]
    attributi["Skills"] = skills_list

    selettore = "div.progress-percent"
    investimento = soup.select(selettore)[2]
    attributi["Buon investimento"] = investimento.text.replace("%", "")

    selettore = "div.progress-percent"
    investimento = soup.select(selettore)[3]
    attributi["Resistenza infortuni"] = investimento.text.replace("%", "")

    selettore = "img.inf_calc"
    try:
        consigliato = soup.select_one(selettore).get("title")
        if "Consigliato per la giornata" in consigliato:
            attributi["Consigliato prossima giornata"] = True
        else:
            attributi["Consigliato prossima giornata"] = False
    except:
        attributi["Consigliato prossima giornata"] = False

    selettore = "span.new_calc"
    nuovo = soup.select_one(selettore)
    if not nuovo == None:
        attributi["Nuovo acquisto"] = True
    else:
        attributi["Nuovo acquisto"] = False

    selettore = "img.inf_calc"
    try:
        infortunato = soup.select_one(selettore).get("title")
        if "Infortunato" in infortunato:
            attributi["Infortunato"] = True
        else:
            attributi["Infortunato"] = False
    except:
        attributi["Infortunato"] = False

    selettore = "#content > div > div.section.nobg.nomargin > div > div > div:nth-child(2) > div.col_three_fifth > div.promo.promo-border.promo-light.row > div:nth-child(3) > div:nth-child(1) > div > img"
    squadra = soup.select_one(selettore).get("title").split(":")[1].strip()
    attributi["Squadra"] = squadra

    selettore = "\tdiv.col_one_fourth:nth-of-type(n+2) div"
    try:
        trend = soup.select(selettore)[0].find("i").get("class")[1]
        if trend == "icon-arrow-up":
            attributi["Trend"] = "UP"
        else:
            attributi["Trend"] = "DOWN"
    except:
        attributi["Trend"] = "STABLE"

    selettore = "div.col_one_fourth:nth-of-type(2) span.rouge"
    presenze_attuali = soup.select_one(selettore).text
    attributi["Presenze campionato corrente"] = presenze_attuali

    return attributi

def appetibilita(df: pd.DataFrame) -> float:
    """Calculate appetibility score (unchanged from original)"""
    # cleaning
    for col in df.columns:
        df.loc[df[col] == "nd", col] = 0

    res = []
    giocatemax = 1

    for index, row in df.iterrows():
        if int(row[-1]) > int(giocatemax):
            giocatemax = int(row[-1])

    for index, row in df.iterrows():
        appetibilita = 0

        # media pesata fantamedia
        if int(row[5]) > 0:
            appetibilita += float(row[7]) * int(row[5]) / 38 * 20/100
        
        if not (
            df.columns[2].split(" ")[-1] == df.columns[6].split(" ")[-1]
            and int(row[-1]) > 5
        ):
            appetibilita = (
                appetibilita * float(row[6]) * int(row[-1]) / giocatemax * 80/100
            )  
        else: 
            appetibilita = float(row[7]) * int(row[5]) / 38 

        # media pesata fantamedia * convenienza rispetto alla quotazione * media scorso anno
        appetibilita = appetibilita * float(row['Punteggio']) * 30/100
        if float(row[1]) == 0: 
            pt = 1
        else: 
            pt = float(row[1])
        appetibilita = (appetibilita / pt * 100 / 40)

        # skills
        try:
            valori = ast.literal_eval(row[-9])
            plus = 0
            for skill in valori:
                plus += skills[skill]
            appetibilita += plus
        except:
            pass

        if row["Nuovo acquisto"]:
            appetibilita -= 2
        if row["Buon investimento"] == 60:
            appetibilita += 3
        if row["Consigliato prossima giornata"]:
            appetibilita += 1
        if row["Trend"] == "UP":
            appetibilita += 2
        if row["Infortunato"]:
            appetibilita -= 1
        if row["Resistenza infortuni"] > 60:
            appetibilita += 4
        if row["Resistenza infortuni"] == 60:
            appetibilita += 2

        res.append(appetibilita)

    return res

if __name__ == "__main__":
    # Command line interface
    import argparse
    
    parser = argparse.ArgumentParser(description='Mantra Fantacalcio Strategy Engine')
    parser.add_argument('--collect-data', action='store_true', help='Collect fresh data from fantacalciopedia')
    parser.add_argument('--analyze', action='store_true', help='Run strategy analysis')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    
    args = parser.parse_args()
    
    if args.collect_data:
        # Data collection (original functionality)
        giocatori_urls = []
        if not os.path.exists("giocatori_urls.txt"):
            for i in tqdm(range(0, len(ruoli), 1)):
                lista = get_giocatori(ruoli[i])
                [giocatori_urls.append(el) for el in lista]
            with open(r"giocatori_urls.txt", "w") as fp:
                for item in giocatori_urls:
                    fp.write("%s\n" % item)
                logger.debug("URL scritti")
        else:
            logger.debug("Leggo la lista giocatori")
            with open("giocatori_urls.txt", "r") as fp:
                giocatori_urls = fp.readlines()

        if not os.path.exists("giocatori.csv"):
            giocatori = []
            for i in tqdm(range(0, len(giocatori_urls), 1)):
                giocatore = get_attributi(giocatori_urls[i])
                giocatori.append(giocatore)
            df = pd.DataFrame.from_dict(giocatori)
            df.to_csv("giocatori.csv", index=False)
            logger.debug("CSV scritto")
        else:
            logger.debug("Leggo il dataset giocatori")
            df = pd.read_csv("giocatori.csv")

        df["Convenienza"] = appetibilita(df)
        df.to_csv("giocatori.csv", index=False)
        logger.debug("Data collection complete!")
    
    if args.analyze:
        # Strategy analysis
        engine = MantraStrategyEngine(args.config)
        report_dir = engine.run_full_analysis()
        print(f"Analysis complete! Reports available in: {report_dir}")
        
    if not args.collect_data and not args.analyze:
        print("Use --collect-data to gather fresh data or --analyze to run strategy analysis")
        print("Example: python mantra_strategy.py --collect-data --analyze")
