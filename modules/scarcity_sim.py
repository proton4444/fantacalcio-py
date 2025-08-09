#!/usr/bin/env python3
"""
Scarcity Simulation Module

Analyzes player scarcity and calculates Expected Value (EV) scores for fantasy football.
Includes position scarcity analysis, expected points calculation, and risk assessment.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Dict, Any, List, NamedTuple
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PlayerEV:
    """Data class for player Expected Value analysis."""
    name: str
    role: str
    team: str
    price: float
    expected_points: float
    position_scarcity: float
    risk_factor: float
    ev_score: float
    percentile_rank: float


class ScarcitySimulation:
    """Analyzes player scarcity and calculates Expected Value scores."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_path = Path(config.get('data_path', 'data/Quotazioni_Fantacalcio_Stagione_2025_26.xlsx'))
        self.players_df = None
        self.position_scarcity = {}
        self.player_evs: List[PlayerEV] = []
        
        # Analysis parameters
        self.scarcity_weights = config.get('scarcity_weights', {
            'Por': 0.8, 'Dc': 0.6, 'Dd': 0.7, 'Ds': 0.7,
            'E': 0.8, 'M': 0.6, 'C': 0.7, 'T': 0.7,
            'W': 0.7, 'A': 0.9, 'Pc': 1.0
        })
        
        self.risk_factors = config.get('risk_factors', {
            'injury_risk': 0.1,
            'form_variance': 0.15,
            'rotation_risk': 0.2
        })
    
    def load_data(self):
        """Load player data from Excel file."""
        try:
            logger.info(f"Loading player data from {self.data_path}")
            
            if not self.data_path.exists():
                logger.error(f"Data file not found: {self.data_path}")
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            # Read Excel file - skip first row (title) and use row 1 as header
            self.players_df = pd.read_excel(self.data_path, header=1)
            
            # Standardize column names based on actual file structure
            column_mapping = {
                'Nome': 'name',
                'Squadra': 'team', 
                'R': 'role',
                'FVM': 'price',  # Using FVM (Fantavoto Medio) as price
                'Qt.A': 'avg_score',  # Using Qt.A as average score
                'Id': 'player_id'
            }
            
            # Rename columns if they exist
            for old_col, new_col in column_mapping.items():
                if old_col in self.players_df.columns:
                    self.players_df = self.players_df.rename(columns={old_col: new_col})
            
            # Ensure required columns exist
            required_cols = ['name', 'role', 'team', 'price']
            missing_cols = [col for col in required_cols if col not in self.players_df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Clean data
            self.players_df = self.players_df.dropna(subset=required_cols)
            self.players_df['price'] = pd.to_numeric(self.players_df['price'], errors='coerce')
            self.players_df = self.players_df[self.players_df['price'] > 0]
            
            # Add avg_score if missing (use price as proxy)
            if 'avg_score' not in self.players_df.columns:
                self.players_df['avg_score'] = self.players_df['price'] / 4.0
            
            logger.success(f"Loaded {len(self.players_df)} players")
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def calculate_position_scarcity(self):
        """Calculate scarcity metrics for each position."""
        logger.info("Calculating position scarcity...")
        
        for role in self.players_df['role'].unique():
            role_players = self.players_df[self.players_df['role'] == role]
            
            # Calculate scarcity metrics
            total_players = len(role_players)
            avg_price = role_players['price'].mean()
            price_std = role_players['price'].std()
            top_tier_threshold = role_players['price'].quantile(0.8)
            top_tier_count = len(role_players[role_players['price'] >= top_tier_threshold])
            
            # Scarcity score (lower = more scarce)
            scarcity_score = (top_tier_count / total_players) * self.scarcity_weights.get(role, 0.7)
            
            self.position_scarcity[role] = {
                'total_players': total_players,
                'avg_price': avg_price,
                'price_std': price_std,
                'top_tier_count': top_tier_count,
                'scarcity_score': scarcity_score,
                'scarcity_multiplier': 1.0 / (scarcity_score + 0.1)  # Avoid division by zero
            }
        
        logger.info(f"Calculated scarcity for {len(self.position_scarcity)} positions")
    
    def calculate_expected_points(self):
        """Calculate expected points for each player."""
        logger.info("Calculating expected points...")
        
        # Use avg_score as base, adjust for price and position
        self.players_df['base_expected'] = self.players_df['avg_score']
        
        # Add position-based adjustments
        for role in self.position_scarcity.keys():
            role_mask = self.players_df['role'] == role
            role_multiplier = self.position_scarcity[role]['scarcity_multiplier']
            
            # Adjust expected points based on position scarcity
            self.players_df.loc[role_mask, 'expected_points'] = (
                self.players_df.loc[role_mask, 'base_expected'] * 
                (0.8 + 0.4 * role_multiplier)  # Scale between 0.8 and 1.2
            )
        
        # Ensure no negative expected points
        self.players_df['expected_points'] = self.players_df['expected_points'].clip(lower=0)
    
    def calculate_risk_factors(self):
        """Calculate risk factors for each player."""
        logger.info("Calculating risk factors...")
        
        # Base risk calculation
        self.players_df['base_risk'] = 0.1  # 10% base risk
        
        # Price-based risk (higher price = lower risk for established players)
        price_percentiles = self.players_df.groupby('role')['price'].rank(pct=True)
        self.players_df['price_risk'] = (1 - price_percentiles) * 0.2
        
        # Position-based risk
        position_risk_map = {
            'Por': 0.05,  # Goalkeepers - low risk
            'Dc': 0.08, 'Dd': 0.10, 'Ds': 0.10,  # Defenders
            'E': 0.12, 'M': 0.10, 'C': 0.12,  # Midfielders
            'T': 0.15, 'W': 0.18,  # Wingers/Trequartisti
            'A': 0.20, 'Pc': 0.25  # Attackers - higher risk/reward
        }
        
        self.players_df['position_risk'] = self.players_df['role'].map(position_risk_map).fillna(0.15)
        
        # Combined risk factor
        self.players_df['risk_factor'] = (
            self.players_df['base_risk'] + 
            self.players_df['price_risk'] + 
            self.players_df['position_risk']
        ).clip(upper=0.5)  # Cap at 50% risk
    
    def calculate_ev_scores(self):
        """Calculate Expected Value scores for all players."""
        logger.info("Calculating EV scores...")
        
        # EV = (Expected Points / Price) * (1 - Risk Factor) * Scarcity Multiplier
        ev_scores = []
        
        for _, player in self.players_df.iterrows():
            role = player['role']
            scarcity_mult = self.position_scarcity[role]['scarcity_multiplier']
            
            # Calculate EV score
            points_per_credit = player['expected_points'] / max(player['price'], 1)
            risk_adjusted = points_per_credit * (1 - player['risk_factor'])
            ev_score = risk_adjusted * scarcity_mult
            
            ev_scores.append(ev_score)
        
        self.players_df['ev_score'] = ev_scores
        
        # Calculate percentile ranks
        self.players_df['ev_percentile'] = self.players_df['ev_score'].rank(pct=True)
        
        # Create PlayerEV objects
        self.player_evs = []
        for _, player in self.players_df.iterrows():
            player_ev = PlayerEV(
                name=player['name'],
                role=player['role'],
                team=player['team'],
                price=player['price'],
                expected_points=player['expected_points'],
                position_scarcity=self.position_scarcity[player['role']]['scarcity_score'],
                risk_factor=player['risk_factor'],
                ev_score=player['ev_score'],
                percentile_rank=player['ev_percentile']
            )
            self.player_evs.append(player_ev)
        
        logger.success(f"Calculated EV scores for {len(self.player_evs)} players")
    
    def create_visualizations(self, output_dir: Path) -> Path:
        """Create visualization plots for scarcity analysis."""
        logger.info("Creating scarcity analysis visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Scarcity Analysis - Expected Value Study', fontsize=16, fontweight='bold')
        
        # 1. EV Score distribution by position
        sns.boxplot(data=self.players_df, x='role', y='ev_score', ax=axes[0, 0])
        axes[0, 0].set_title('EV Score Distribution by Position')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Price vs Expected Points scatter
        scatter = axes[0, 1].scatter(self.players_df['price'], self.players_df['expected_points'], 
                                   c=self.players_df['ev_score'], cmap='viridis', alpha=0.6)
        axes[0, 1].set_xlabel('Price')
        axes[0, 1].set_ylabel('Expected Points')
        axes[0, 1].set_title('Price vs Expected Points (colored by EV)')
        plt.colorbar(scatter, ax=axes[0, 1], label='EV Score')
        
        # 3. Position scarcity bar chart
        positions = list(self.position_scarcity.keys())
        scarcity_scores = [self.position_scarcity[pos]['scarcity_score'] for pos in positions]
        
        axes[1, 0].bar(positions, scarcity_scores, color='skyblue')
        axes[1, 0].set_title('Position Scarcity Scores')
        axes[1, 0].set_ylabel('Scarcity Score (lower = more scarce)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Top EV players table
        axes[1, 1].axis('off')
        top_players = sorted(self.player_evs, key=lambda x: x.ev_score, reverse=True)[:10]
        
        table_data = []
        for player in top_players:
            table_data.append([
                player.name[:15] + "..." if len(player.name) > 15 else player.name,
                player.role,
                f"{player.price:.0f}",
                f"{player.ev_score:.3f}"
            ])
        
        table = axes[1, 1].table(cellText=table_data,
                               colLabels=['Player', 'Role', 'Price', 'EV Score'],
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        axes[1, 1].set_title('Top 10 EV Players')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"scarcity_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.success(f"Scarcity visualization saved to {plot_path}")
        return plot_path
    
    def export_results(self, output_dir: Path) -> Path:
        """Export analysis results to CSV."""
        logger.info("Exporting scarcity analysis results...")
        
        # Prepare export data
        export_data = []
        for player_ev in self.player_evs:
            export_data.append({
                'name': player_ev.name,
                'role': player_ev.role,
                'team': player_ev.team,
                'price': player_ev.price,
                'expected_points': player_ev.expected_points,
                'position_scarcity': player_ev.position_scarcity,
                'risk_factor': player_ev.risk_factor,
                'ev_score': player_ev.ev_score,
                'ev_percentile': player_ev.percentile_rank
            })
        
        # Create DataFrame and sort by EV score
        results_df = pd.DataFrame(export_data)
        results_df = results_df.sort_values('ev_score', ascending=False)
        
        # Export to CSV
        csv_filename = f"scarcity_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path = output_dir / csv_filename
        results_df.to_csv(csv_path, index=False)
        
        logger.success(f"Results exported to {csv_path}")
        return csv_path
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the analysis."""
        if not self.player_evs:
            return {}
        
        ev_scores = [pev.ev_score for pev in self.player_evs]
        
        return {
            'total_players': len(self.player_evs),
            'avg_ev_score': np.mean(ev_scores),
            'median_ev_score': np.median(ev_scores),
            'top_ev_score': np.max(ev_scores),
            'positions_analyzed': len(self.position_scarcity),
            'high_value_players': len([pev for pev in self.player_evs if pev.percentile_rank >= 0.9])
        }
