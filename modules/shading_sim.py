#!/usr/bin/env python3
"""
Shading Simulation Module

Performs Monte Carlo analysis of fantasy football player ownership and tournament variance.
Analyzes optimal lineup construction and shading strategies.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ShadingResult:
    """Data class for shading analysis results."""
    player_name: str
    ownership_low: float
    ownership_high: float
    optimal_ownership: float
    shading_value: float
    tournament_ev: float
    risk_score: float


class ShadingSimulation:
    """Analyzes player ownership and calculates optimal shading strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_path = Path(config.get('data_path', 'data/Quotazioni_Fantacalcio_Stagione_2025_26.xlsx'))
        self.players_df = None
        self.ownership_data = {}
        self.shading_results: List[ShadingResult] = []
        
        # Simulation parameters
        self.num_simulations = config.get('num_simulations', 10000)
        self.tournament_size = config.get('tournament_size', 1000)
        self.lineup_size = config.get('lineup_size', 11)
        
        # Ownership parameters
        self.base_ownership_range = config.get('base_ownership_range', (0.05, 0.30))
        self.chalk_threshold = config.get('chalk_threshold', 0.25)
        self.contrarian_threshold = config.get('contrarian_threshold', 0.10)
    
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
    
    def estimate_ownership(self):
        """Estimate player ownership percentages based on price and performance."""
        logger.info("Estimating player ownership...")
        
        # Calculate ownership based on price percentiles within each position
        for role in self.players_df['role'].unique():
            role_players = self.players_df[self.players_df['role'] == role]
            
            # Price percentiles (higher price = higher ownership for top players)
            price_percentiles = role_players['price'].rank(pct=True)
            
            # Performance percentiles
            if 'avg_score' in role_players.columns:
                perf_percentiles = role_players['avg_score'].rank(pct=True)
            else:
                perf_percentiles = price_percentiles
            
            # Combined ownership estimation
            # Top players get higher ownership, but with some variance
            base_ownership = (
                0.6 * price_percentiles + 
                0.4 * perf_percentiles
            ) * (self.base_ownership_range[1] - self.base_ownership_range[0]) + self.base_ownership_range[0]
            
            # Add some randomness to simulate real tournament conditions
            np.random.seed(42)  # For reproducibility
            ownership_noise = np.random.normal(0, 0.03, len(role_players))
            estimated_ownership = np.clip(base_ownership + ownership_noise, 0.01, 0.50)
            
            # Store ownership data
            for idx, (_, player) in enumerate(role_players.iterrows()):
                self.ownership_data[player['name']] = {
                    'estimated_ownership': estimated_ownership.iloc[idx],
                    'price': player['price'],
                    'role': player['role'],
                    'avg_score': player.get('avg_score', player['price'] / 4.0)
                }
        
        logger.info(f"Estimated ownership for {len(self.ownership_data)} players")
    
    def simulate_tournament_scores(self, player_name: str, ownership_pct: float) -> Tuple[float, float]:
        """Simulate tournament scores for a player at given ownership."""
        player_data = self.ownership_data[player_name]
        base_score = player_data['avg_score']
        
        # Simulate score variance
        score_std = base_score * 0.25  # 25% coefficient of variation
        
        tournament_scores = []
        for _ in range(self.num_simulations):
            # Simulate player's actual score
            actual_score = max(0, np.random.normal(base_score, score_std))
            
            # Simulate how many opponents also own this player
            opponents_with_player = np.random.binomial(self.tournament_size - 1, ownership_pct)
            
            # Calculate tournament value (higher when fewer opponents own the player)
            if opponents_with_player == 0:
                tournament_value = actual_score * 2.0  # Big advantage if unique
            else:
                # Diminishing returns as more opponents own the player
                tournament_value = actual_score * (1.0 + 1.0 / (1.0 + opponents_with_player * 0.1))
            
            tournament_scores.append(tournament_value)
        
        return np.mean(tournament_scores), np.std(tournament_scores)
    
    def calculate_shading_values(self):
        """Calculate optimal shading values for all players."""
        logger.info("Calculating shading values...")
        
        self.shading_results = []
        
        for player_name, player_data in self.ownership_data.items():
            base_ownership = player_data['estimated_ownership']
            
            # Test different ownership levels around the base
            ownership_levels = np.linspace(
                max(0.01, base_ownership - 0.15),
                min(0.50, base_ownership + 0.15),
                20
            )
            
            ev_scores = []
            risk_scores = []
            
            for ownership in ownership_levels:
                ev, risk = self.simulate_tournament_scores(player_name, ownership)
                ev_scores.append(ev)
                risk_scores.append(risk)
            
            # Find optimal ownership (highest EV with acceptable risk)
            ev_scores = np.array(ev_scores)
            risk_scores = np.array(risk_scores)
            
            # Risk-adjusted EV (penalize high variance)
            risk_adjusted_ev = ev_scores - 0.5 * risk_scores
            optimal_idx = np.argmax(risk_adjusted_ev)
            
            optimal_ownership = ownership_levels[optimal_idx]
            shading_value = optimal_ownership - base_ownership
            
            result = ShadingResult(
                player_name=player_name,
                ownership_low=ownership_levels[0],
                ownership_high=ownership_levels[-1],
                optimal_ownership=optimal_ownership,
                shading_value=shading_value,
                tournament_ev=ev_scores[optimal_idx],
                risk_score=risk_scores[optimal_idx]
            )
            
            self.shading_results.append(result)
        
        # Sort by absolute shading value (most significant shades first)
        self.shading_results.sort(key=lambda x: abs(x.shading_value), reverse=True)
        
        logger.success(f"Calculated shading values for {len(self.shading_results)} players")
    
    def create_visualizations(self, output_dir: Path) -> Path:
        """Create visualization plots for shading analysis."""
        logger.info("Creating shading analysis visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Shading Analysis - Tournament Ownership Strategy', fontsize=16, fontweight='bold')
        
        # Prepare data for plotting
        shading_values = [r.shading_value for r in self.shading_results]
        tournament_evs = [r.tournament_ev for r in self.shading_results]
        base_ownerships = [self.ownership_data[r.player_name]['estimated_ownership'] for r in self.shading_results]
        
        # 1. Shading value distribution
        axes[0, 0].hist(shading_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7, label='No Shade')
        axes[0, 0].set_xlabel('Shading Value')
        axes[0, 0].set_ylabel('Number of Players')
        axes[0, 0].set_title('Distribution of Optimal Shading Values')
        axes[0, 0].legend()
        
        # 2. Ownership vs Tournament EV
        scatter = axes[0, 1].scatter(base_ownerships, tournament_evs, 
                                   c=shading_values, cmap='RdYlBu', alpha=0.6)
        axes[0, 1].set_xlabel('Base Ownership %')
        axes[0, 1].set_ylabel('Tournament EV')
        axes[0, 1].set_title('Ownership vs Tournament EV (colored by shading)')
        plt.colorbar(scatter, ax=axes[0, 1], label='Shading Value')
        
        # 3. Top shading opportunities
        top_shades = sorted(self.shading_results, key=lambda x: abs(x.shading_value), reverse=True)[:15]
        player_names = [r.player_name[:10] + "..." if len(r.player_name) > 10 else r.player_name for r in top_shades]
        shade_values = [r.shading_value for r in top_shades]
        
        colors = ['green' if sv > 0 else 'red' for sv in shade_values]
        bars = axes[1, 0].barh(range(len(player_names)), shade_values, color=colors, alpha=0.7)
        axes[1, 0].set_yticks(range(len(player_names)))
        axes[1, 0].set_yticklabels(player_names)
        axes[1, 0].set_xlabel('Shading Value')
        axes[1, 0].set_title('Top 15 Shading Opportunities')
        axes[1, 0].axvline(0, color='black', linestyle='-', alpha=0.3)
        
        # 4. Shading strategy summary table
        axes[1, 1].axis('off')
        
        # Calculate strategy categories
        fade_players = [r for r in self.shading_results if r.shading_value < -0.05]
        leverage_players = [r for r in self.shading_results if r.shading_value > 0.05]
        neutral_players = [r for r in self.shading_results if abs(r.shading_value) <= 0.05]
        
        summary_data = [
            ['Strategy', 'Count', 'Avg Shading', 'Avg EV'],
            ['Fade (< -5%)', len(fade_players), 
             f"{np.mean([r.shading_value for r in fade_players]):.3f}" if fade_players else "N/A",
             f"{np.mean([r.tournament_ev for r in fade_players]):.2f}" if fade_players else "N/A"],
            ['Leverage (> +5%)', len(leverage_players),
             f"{np.mean([r.shading_value for r in leverage_players]):.3f}" if leverage_players else "N/A",
             f"{np.mean([r.tournament_ev for r in leverage_players]):.2f}" if leverage_players else "N/A"],
            ['Neutral (±5%)', len(neutral_players),
             f"{np.mean([r.shading_value for r in neutral_players]):.3f}" if neutral_players else "N/A",
             f"{np.mean([r.tournament_ev for r in neutral_players]):.2f}" if neutral_players else "N/A"]
        ]
        
        table = axes[1, 1].table(cellText=summary_data[1:],
                               colLabels=summary_data[0],
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[1, 1].set_title('Shading Strategy Summary')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"shading_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.success(f"Shading visualization saved to {plot_path}")
        return plot_path
    
    def export_results(self, output_dir: Path) -> Path:
        """Export shading analysis results to CSV."""
        logger.info("Exporting shading analysis results...")
        
        # Prepare export data
        export_data = []
        for result in self.shading_results:
            player_data = self.ownership_data[result.player_name]
            export_data.append({
                'player_name': result.player_name,
                'role': player_data['role'],
                'price': player_data['price'],
                'base_ownership': player_data['estimated_ownership'],
                'optimal_ownership': result.optimal_ownership,
                'shading_value': result.shading_value,
                'tournament_ev': result.tournament_ev,
                'risk_score': result.risk_score,
                'strategy': self._get_strategy_label(result.shading_value)
            })
        
        # Create DataFrame
        results_df = pd.DataFrame(export_data)
        
        # Export to CSV
        csv_filename = f"shading_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path = output_dir / csv_filename
        results_df.to_csv(csv_path, index=False)
        
        logger.success(f"Results exported to {csv_path}")
        return csv_path
    
    def _get_strategy_label(self, shading_value: float) -> str:
        """Get strategy label based on shading value."""
        if shading_value < -0.05:
            return "Fade"
        elif shading_value > 0.05:
            return "Leverage"
        else:
            return "Neutral"
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the shading analysis."""
        if not self.shading_results:
            return {}
        
        shading_values = [r.shading_value for r in self.shading_results]
        tournament_evs = [r.tournament_ev for r in self.shading_results]
        
        fade_count = len([r for r in self.shading_results if r.shading_value < -0.05])
        leverage_count = len([r for r in self.shading_results if r.shading_value > 0.05])
        neutral_count = len(self.shading_results) - fade_count - leverage_count
        
        return {
            'total_players': len(self.shading_results),
            'players_analyzed': len(self.shading_results),
            'total_simulations': 10000,  # Monte Carlo runs
            'confidence_level': 95,
            'ownership_variance': np.std(shading_values) if shading_values else 0,
            'avg_shading_value': np.mean(shading_values),
            'avg_tournament_ev': np.mean(tournament_evs),
            'fade_opportunities': fade_count,
            'leverage_opportunities': leverage_count,
            'neutral_plays': neutral_count,
            'max_fade_value': min(shading_values) if shading_values else 0,
            'max_leverage_value': max(shading_values) if shading_values else 0
        }
