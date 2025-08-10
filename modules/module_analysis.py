#!/usr/bin/env python3
"""
Module Performance Analysis

Analyzes player performance across different modules and formations.
Provides insights into player effectiveness in various tactical setups.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


@dataclass
class ModulePerformance:
    """Container for module performance metrics."""
    module_name: str
    formation: str
    avg_points: float
    consistency: float
    risk_factor: float
    player_count: int
    top_performers: List[str]


@dataclass
class PlayerModuleStats:
    """Player statistics across different modules."""
    player_name: str
    role: str
    module_performances: Dict[str, float]
    best_module: str
    worst_module: str
    versatility_score: float


class ModulePerformanceAnalyzer:
    """Analyzes player performance across different tactical modules."""
    
    def __init__(self, config: dict):
        self.config = config
        self.data = None
        self.module_stats = {}
        self.player_stats = {}
        self.formation_analysis = {}
        
    def load_data(self) -> None:
        """Load player data from Excel file."""
        try:
            excel_path = Path(self.config['excel_path'])
            logger.info(f"Loading data from {excel_path}")
            
            self.data = pd.read_excel(excel_path, header=1)  # Use header=1 based on Excel structure
            logger.info(f"Loaded {len(self.data)} players")
            
            # Map actual Excel columns to expected names
            column_mapping = {
                'Nome': 'Nome',
                'R': 'Ruolo', 
                'Qt.A': 'Quotazione',
                'FVM': 'FantaMedia'
            }
            
            # Rename columns to match expected names
            self.data = self.data.rename(columns=column_mapping)
            
            # Ensure required columns exist after mapping
            required_cols = ['Nome', 'Ruolo', 'Quotazione', 'FantaMedia']
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns after mapping: {missing_cols}")
                
            # Clean data - remove rows with missing essential data
            self.data = self.data.dropna(subset=['Nome', 'Ruolo', 'Quotazione', 'FantaMedia'])
            logger.info(f"After cleaning: {len(self.data)} players")
                
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def analyze_module_performance(self) -> Dict[str, ModulePerformance]:
        """Analyze performance across different tactical modules."""
        logger.info("Analyzing module performance...")
        
        modules = self.config.get('modules', {})
        results = {}
        
        for module_name, module_config in modules.items():
            formation = module_config.get('formation', 'Unknown')
            role_mapping = module_config.get('role_mapping', {})
            
            # Filter players for this module
            module_players = self._get_module_players(role_mapping)
            
            if module_players.empty:
                logger.warning(f"No players found for module {module_name}")
                continue
            
            # Calculate performance metrics
            avg_points = module_players['FantaMedia'].mean()
            consistency = 1 / (module_players['FantaMedia'].std() + 0.1)  # Higher is better
            risk_factor = module_players['FantaMedia'].std() / avg_points if avg_points > 0 else 0
            player_count = len(module_players)
            
            # Get top performers
            top_performers = module_players.nlargest(5, 'FantaMedia')['Nome'].tolist()
            
            results[module_name] = ModulePerformance(
                module_name=module_name,
                formation=formation,
                avg_points=avg_points,
                consistency=consistency,
                risk_factor=risk_factor,
                player_count=player_count,
                top_performers=top_performers
            )
            
        self.module_stats = results
        logger.success(f"Analyzed {len(results)} modules")
        return results
    
    def analyze_player_versatility(self) -> Dict[str, PlayerModuleStats]:
        """Analyze how well players perform across different modules."""
        logger.info("Analyzing player versatility...")
        
        modules = self.config.get('modules', {})
        player_stats = {}
        
        for _, player in self.data.iterrows():
            player_name = player['Nome']
            role = player['Ruolo']
            base_performance = player['FantaMedia']
            
            module_performances = {}
            
            # Check performance in each module
            for module_name, module_config in modules.items():
                role_mapping = module_config.get('role_mapping', {})
                
                # Check if player fits in this module
                if self._player_fits_module(player, role_mapping):
                    # Calculate adjusted performance based on role importance
                    role_weight = self._get_role_weight(role, role_mapping)
                    adjusted_performance = base_performance * role_weight
                    module_performances[module_name] = adjusted_performance
            
            if module_performances:
                # Calculate versatility metrics
                performances = list(module_performances.values())
                best_module = max(module_performances, key=module_performances.get)
                worst_module = min(module_performances, key=module_performances.get)
                versatility_score = 1 - (np.std(performances) / np.mean(performances)) if performances else 0
                
                player_stats[player_name] = PlayerModuleStats(
                    player_name=player_name,
                    role=role,
                    module_performances=module_performances,
                    best_module=best_module,
                    worst_module=worst_module,
                    versatility_score=max(0, versatility_score)
                )
        
        self.player_stats = player_stats
        logger.success(f"Analyzed versatility for {len(player_stats)} players")
        return player_stats
    
    def _get_module_players(self, role_mapping: Dict[str, int]) -> pd.DataFrame:
        """Get players that fit in a specific module."""
        if not role_mapping:
            return pd.DataFrame()
        
        # Filter players by roles needed in this module
        valid_roles = list(role_mapping.keys())
        return self.data[self.data['Ruolo'].isin(valid_roles)].copy()
    
    def _player_fits_module(self, player: pd.Series, role_mapping: Dict[str, int]) -> bool:
        """Check if a player fits in a specific module."""
        return player['Ruolo'] in role_mapping
    
    def _get_role_weight(self, role: str, role_mapping: Dict[str, int]) -> float:
        """Get the importance weight of a role in a module."""
        total_players = sum(role_mapping.values())
        role_count = role_mapping.get(role, 0)
        return role_count / total_players if total_players > 0 else 1.0
    
    def create_visualizations(self, output_dir: Path) -> Path:
        """Create module analysis visualizations."""
        logger.info("Creating module analysis visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Module Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Module Performance Comparison
        if self.module_stats:
            modules = list(self.module_stats.keys())
            avg_points = [stats.avg_points for stats in self.module_stats.values()]
            
            axes[0, 0].bar(modules, avg_points, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Average Points by Module')
            axes[0, 0].set_ylabel('Average FantaMedia')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Risk vs Reward
        if self.module_stats:
            risk_factors = [stats.risk_factor for stats in self.module_stats.values()]
            
            axes[0, 1].scatter(avg_points, risk_factors, s=100, alpha=0.7, color='coral')
            for i, module in enumerate(modules):
                axes[0, 1].annotate(module, (avg_points[i], risk_factors[i]), 
                                   xytext=(5, 5), textcoords='offset points')
            axes[0, 1].set_xlabel('Average Points')
            axes[0, 1].set_ylabel('Risk Factor')
            axes[0, 1].set_title('Risk vs Reward by Module')
        
        # 3. Player Versatility Distribution
        if self.player_stats:
            versatility_scores = [stats.versatility_score for stats in self.player_stats.values()]
            
            axes[1, 0].hist(versatility_scores, bins=20, alpha=0.7, color='lightgreen')
            axes[1, 0].set_xlabel('Versatility Score')
            axes[1, 0].set_ylabel('Number of Players')
            axes[1, 0].set_title('Player Versatility Distribution')
        
        # 4. Top Versatile Players
        if self.player_stats:
            top_versatile = sorted(self.player_stats.values(), 
                                 key=lambda x: x.versatility_score, reverse=True)[:10]
            
            names = [p.player_name[:15] for p in top_versatile]  # Truncate long names
            scores = [p.versatility_score for p in top_versatile]
            
            axes[1, 1].barh(names, scores, color='gold', alpha=0.7)
            axes[1, 1].set_xlabel('Versatility Score')
            axes[1, 1].set_title('Top 10 Most Versatile Players')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        plot_path = output_dir / f'module_analysis_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.success(f"Visualizations saved to {plot_path}")
        return plot_path
    
    def export_results(self, output_dir: Path) -> Path:
        """Export analysis results to CSV."""
        logger.info("Exporting module analysis results...")
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        csv_path = output_dir / f'module_analysis_results_{timestamp}.csv'
        
        # Prepare data for export
        export_data = []
        
        # Module performance data
        for module_name, stats in self.module_stats.items():
            export_data.append({
                'Type': 'Module Performance',
                'Name': module_name,
                'Formation': stats.formation,
                'Avg_Points': stats.avg_points,
                'Consistency': stats.consistency,
                'Risk_Factor': stats.risk_factor,
                'Player_Count': stats.player_count,
                'Top_Performers': '; '.join(stats.top_performers)
            })
        
        # Player versatility data
        for player_name, stats in self.player_stats.items():
            export_data.append({
                'Type': 'Player Versatility',
                'Name': player_name,
                'Role': stats.role,
                'Best_Module': stats.best_module,
                'Worst_Module': stats.worst_module,
                'Versatility_Score': stats.versatility_score,
                'Module_Count': len(stats.module_performances)
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(export_data)
        df.to_csv(csv_path, index=False)
        
        logger.success(f"Results exported to {csv_path}")
        return csv_path
    
    def get_summary_stats(self) -> Dict[str, any]:
        """Get summary statistics for the analysis."""
        return {
            'total_modules_analyzed': len(self.module_stats),
            'total_players_analyzed': len(self.player_stats),
            'best_performing_module': max(self.module_stats.keys(), 
                                        key=lambda x: self.module_stats[x].avg_points) if self.module_stats else None,
            'most_versatile_player': max(self.player_stats.keys(), 
                                       key=lambda x: self.player_stats[x].versatility_score) if self.player_stats else None,
            'avg_versatility_score': np.mean([stats.versatility_score for stats in self.player_stats.values()]) if self.player_stats else 0
        }


def analyze_module_performance(config: dict, output_dir: Path) -> dict:
    """Main function to run module performance analysis."""
    analyzer = ModulePerformanceAnalyzer(config)
    analyzer.load_data()
    
    # Run analyses
    module_stats = analyzer.analyze_module_performance()
    player_stats = analyzer.analyze_player_versatility()
    
    # Generate outputs
    plot_path = analyzer.create_visualizations(output_dir)
    csv_path = analyzer.export_results(output_dir)
    
    return {
        'module_stats': module_stats,
        'player_stats': player_stats,
        'summary_stats': analyzer.get_summary_stats(),
        'plot_path': plot_path,
        'csv_path': csv_path
    }