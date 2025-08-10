#!/usr/bin/env python3
"""
Tier System Module

Classifies players into tiers based on performance metrics and provides
tier-based analysis for auction strategy and team building.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class TierLevel(Enum):
    """Enumeration of player tier levels."""
    ELITE = "Elite"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    BENCH = "Bench"


@dataclass
class TierThresholds:
    """Thresholds for tier classification."""
    elite_min: float
    high_min: float
    medium_min: float
    low_min: float
    # bench_min is implicitly 0


@dataclass
class PlayerTierInfo:
    """Information about a player's tier classification."""
    player_name: str
    role: str
    tier: TierLevel
    score: float
    percentile: float
    tier_rank: int  # Rank within tier
    role_rank: int  # Rank within role


@dataclass
class TierAnalysis:
    """Analysis results for tier system."""
    tier_distribution: Dict[TierLevel, int]
    role_tier_matrix: Dict[str, Dict[TierLevel, int]]
    tier_stats: Dict[TierLevel, Dict[str, float]]
    thresholds: TierThresholds


class TierClassifier:
    """Classifies players into performance tiers."""
    
    def __init__(self, config: dict):
        self.config = config
        self.data = None
        self.player_tiers = {}
        self.tier_analysis = None
        self.thresholds = None
        
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
    
    def calculate_tier_thresholds(self, score_column: str = 'FantaMedia') -> TierThresholds:
        """Calculate tier thresholds based on score distribution."""
        logger.info(f"Calculating tier thresholds based on {score_column}")
        
        # Get quantiles from config or use defaults
        quantiles = self.config.get('tier_quantiles', {
            'elite': 0.90,
            'high': 0.75,
            'medium': 0.50,
            'low': 0.25
        })
        
        scores = self.data[score_column].dropna()
        
        thresholds = TierThresholds(
            elite_min=scores.quantile(quantiles['elite']),
            high_min=scores.quantile(quantiles['high']),
            medium_min=scores.quantile(quantiles['medium']),
            low_min=scores.quantile(quantiles['low'])
        )
        
        self.thresholds = thresholds
        logger.info(f"Tier thresholds: Elite≥{thresholds.elite_min:.2f}, "
                   f"High≥{thresholds.high_min:.2f}, Medium≥{thresholds.medium_min:.2f}, "
                   f"Low≥{thresholds.low_min:.2f}")
        
        return thresholds
    
    def classify_players_by_tier(self, score_column: str = 'FantaMedia') -> Dict[str, PlayerTierInfo]:
        """Classify all players into tiers."""
        logger.info("Classifying players by tier...")
        
        if self.thresholds is None:
            self.calculate_tier_thresholds(score_column)
        
        player_tiers = {}
        
        # Calculate percentiles for all players
        scores = self.data[score_column].dropna()
        
        for _, player in self.data.iterrows():
            if pd.isna(player[score_column]):
                continue
                
            player_name = player['Nome']
            role = player['Ruolo']
            score = player[score_column]
            
            # Determine tier
            if score >= self.thresholds.elite_min:
                tier = TierLevel.ELITE
            elif score >= self.thresholds.high_min:
                tier = TierLevel.HIGH
            elif score >= self.thresholds.medium_min:
                tier = TierLevel.MEDIUM
            elif score >= self.thresholds.low_min:
                tier = TierLevel.LOW
            else:
                tier = TierLevel.BENCH
            
            # Calculate percentile
            percentile = (scores < score).mean() * 100
            
            player_tiers[player_name] = PlayerTierInfo(
                player_name=player_name,
                role=role,
                tier=tier,
                score=score,
                percentile=percentile,
                tier_rank=0,  # Will be calculated later
                role_rank=0   # Will be calculated later
            )
        
        # Calculate ranks within tiers and roles
        self._calculate_ranks(player_tiers, score_column)
        
        self.player_tiers = player_tiers
        logger.success(f"Classified {len(player_tiers)} players into tiers")
        
        return player_tiers
    
    def _calculate_ranks(self, player_tiers: Dict[str, PlayerTierInfo], score_column: str) -> None:
        """Calculate ranks within tiers and roles."""
        # Group players by tier and role
        tier_groups = {}
        role_groups = {}
        
        for player_info in player_tiers.values():
            # Group by tier
            if player_info.tier not in tier_groups:
                tier_groups[player_info.tier] = []
            tier_groups[player_info.tier].append(player_info)
            
            # Group by role
            if player_info.role not in role_groups:
                role_groups[player_info.role] = []
            role_groups[player_info.role].append(player_info)
        
        # Calculate tier ranks
        for tier, players in tier_groups.items():
            sorted_players = sorted(players, key=lambda x: x.score, reverse=True)
            for rank, player in enumerate(sorted_players, 1):
                player.tier_rank = rank
        
        # Calculate role ranks
        for role, players in role_groups.items():
            sorted_players = sorted(players, key=lambda x: x.score, reverse=True)
            for rank, player in enumerate(sorted_players, 1):
                player.role_rank = rank
    
    def analyze_tier_distribution(self) -> TierAnalysis:
        """Analyze the distribution of players across tiers."""
        logger.info("Analyzing tier distribution...")
        
        if not self.player_tiers:
            raise ValueError("Players must be classified before analysis")
        
        # Tier distribution
        tier_distribution = {tier: 0 for tier in TierLevel}
        for player_info in self.player_tiers.values():
            tier_distribution[player_info.tier] += 1
        
        # Role-tier matrix
        roles = list(set(player_info.role for player_info in self.player_tiers.values()))
        role_tier_matrix = {}
        
        for role in roles:
            role_tier_matrix[role] = {tier: 0 for tier in TierLevel}
            for player_info in self.player_tiers.values():
                if player_info.role == role:
                    role_tier_matrix[role][player_info.tier] += 1
        
        # Tier statistics
        tier_stats = {}
        for tier in TierLevel:
            tier_players = [p for p in self.player_tiers.values() if p.tier == tier]
            if tier_players:
                scores = [p.score for p in tier_players]
                tier_stats[tier] = {
                    'count': len(tier_players),
                    'avg_score': np.mean(scores),
                    'min_score': np.min(scores),
                    'max_score': np.max(scores),
                    'std_score': np.std(scores)
                }
            else:
                tier_stats[tier] = {
                    'count': 0,
                    'avg_score': 0,
                    'min_score': 0,
                    'max_score': 0,
                    'std_score': 0
                }
        
        self.tier_analysis = TierAnalysis(
            tier_distribution=tier_distribution,
            role_tier_matrix=role_tier_matrix,
            tier_stats=tier_stats,
            thresholds=self.thresholds
        )
        
        logger.success("Tier distribution analysis completed")
        return self.tier_analysis
    
    def get_tier_recommendations(self, budget: float, formation: str = "3-5-2") -> Dict[str, List[str]]:
        """Get player recommendations by tier for a given budget and formation."""
        logger.info(f"Getting tier recommendations for budget {budget} and formation {formation}")
        
        # Parse formation
        formation_parts = formation.split('-')
        if len(formation_parts) != 3:
            raise ValueError(f"Invalid formation format: {formation}")
        
        defenders = int(formation_parts[0])
        midfielders = int(formation_parts[1])
        forwards = int(formation_parts[2])
        
        recommendations = {
            'Portiere': [],
            'Difensore': [],
            'Centrocampista': [],
            'Attaccante': []
        }
        
        # Get top players by role and tier
        for role in recommendations.keys():
            role_players = [p for p in self.player_tiers.values() if p.role == role]
            role_players.sort(key=lambda x: x.score, reverse=True)
            
            # Determine how many players to recommend based on formation
            if role == 'Portiere':
                target_count = 1
            elif role == 'Difensore':
                target_count = defenders
            elif role == 'Centrocampista':
                target_count = midfielders
            elif role == 'Attaccante':
                target_count = forwards
            else:
                target_count = 1
            
            # Add top players from each tier
            for tier in [TierLevel.ELITE, TierLevel.HIGH, TierLevel.MEDIUM]:
                tier_players = [p for p in role_players if p.tier == tier]
                recommendations[role].extend([p.player_name for p in tier_players[:target_count]])
        
        return recommendations
    
    def create_visualizations(self, output_dir: Path) -> Path:
        """Create tier analysis visualizations."""
        logger.info("Creating tier analysis visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Player Tier Analysis', fontsize=16, fontweight='bold')
        
        # 1. Tier Distribution
        if self.tier_analysis:
            tiers = [tier.value for tier in TierLevel]
            counts = [self.tier_analysis.tier_distribution[tier] for tier in TierLevel]
            
            axes[0, 0].pie(counts, labels=tiers, autopct='%1.1f%%', startangle=90)
            axes[0, 0].set_title('Player Distribution by Tier')
        
        # 2. Role-Tier Heatmap
        if self.tier_analysis:
            roles = list(self.tier_analysis.role_tier_matrix.keys())
            tier_names = [tier.value for tier in TierLevel]
            
            heatmap_data = []
            for role in roles:
                row = [self.tier_analysis.role_tier_matrix[role][tier] for tier in TierLevel]
                heatmap_data.append(row)
            
            im = axes[0, 1].imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
            axes[0, 1].set_xticks(range(len(tier_names)))
            axes[0, 1].set_xticklabels(tier_names)
            axes[0, 1].set_yticks(range(len(roles)))
            axes[0, 1].set_yticklabels(roles)
            axes[0, 1].set_title('Players by Role and Tier')
            
            # Add text annotations
            for i in range(len(roles)):
                for j in range(len(tier_names)):
                    text = axes[0, 1].text(j, i, heatmap_data[i][j],
                                         ha="center", va="center", color="black")
        
        # 3. Score Distribution by Tier
        if self.player_tiers:
            tier_scores = {tier.value: [] for tier in TierLevel}
            for player_info in self.player_tiers.values():
                tier_scores[player_info.tier.value].append(player_info.score)
            
            # Create box plot
            data_for_boxplot = [scores for scores in tier_scores.values() if scores]
            labels_for_boxplot = [tier for tier, scores in tier_scores.items() if scores]
            
            axes[1, 0].boxplot(data_for_boxplot, labels=labels_for_boxplot)
            axes[1, 0].set_title('Score Distribution by Tier')
            axes[1, 0].set_ylabel('FantaMedia Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Top Players by Tier
        if self.player_tiers:
            # Get top 3 players from each tier
            tier_top_players = {}
            for tier in TierLevel:
                tier_players = [p for p in self.player_tiers.values() if p.tier == tier]
                tier_players.sort(key=lambda x: x.score, reverse=True)
                tier_top_players[tier.value] = tier_players[:3]
            
            # Create a simple text display
            axes[1, 1].axis('off')
            y_pos = 0.9
            for tier_name, players in tier_top_players.items():
                if players:
                    axes[1, 1].text(0.1, y_pos, f"{tier_name} Tier:", fontweight='bold', 
                                   transform=axes[1, 1].transAxes)
                    y_pos -= 0.05
                    for player in players:
                        axes[1, 1].text(0.15, y_pos, 
                                       f"{player.player_name} ({player.score:.1f})",
                                       transform=axes[1, 1].transAxes)
                        y_pos -= 0.04
                    y_pos -= 0.02
            axes[1, 1].set_title('Top Players by Tier')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        plot_path = output_dir / f'tier_analysis_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.success(f"Visualizations saved to {plot_path}")
        return plot_path
    
    def export_results(self, output_dir: Path) -> Path:
        """Export tier analysis results to CSV."""
        logger.info("Exporting tier analysis results...")
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        csv_path = output_dir / f'tier_analysis_results_{timestamp}.csv'
        
        # Prepare data for export
        export_data = []
        
        for player_name, player_info in self.player_tiers.items():
            export_data.append({
                'Player_Name': player_info.player_name,
                'Role': player_info.role,
                'Tier': player_info.tier.value,
                'Score': player_info.score,
                'Percentile': player_info.percentile,
                'Tier_Rank': player_info.tier_rank,
                'Role_Rank': player_info.role_rank
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(export_data)
        df = df.sort_values(['Tier', 'Tier_Rank'])
        df.to_csv(csv_path, index=False)
        
        logger.success(f"Results exported to {csv_path}")
        return csv_path
    
    def get_summary_stats(self) -> Dict[str, any]:
        """Get summary statistics for the tier analysis."""
        if not self.tier_analysis:
            return {}
        
        total_players = sum(self.tier_analysis.tier_distribution.values())
        
        return {
            'total_players_classified': total_players,
            'elite_players': self.tier_analysis.tier_distribution[TierLevel.ELITE],
            'high_tier_players': self.tier_analysis.tier_distribution[TierLevel.HIGH],
            'medium_tier_players': self.tier_analysis.tier_distribution[TierLevel.MEDIUM],
            'low_tier_players': self.tier_analysis.tier_distribution[TierLevel.LOW],
            'bench_players': self.tier_analysis.tier_distribution[TierLevel.BENCH],
            'elite_threshold': self.thresholds.elite_min if self.thresholds else 0,
            'high_threshold': self.thresholds.high_min if self.thresholds else 0
        }


def classify_players_by_tier(config: dict, output_dir: Path) -> dict:
    """Main function to run tier classification analysis."""
    classifier = TierClassifier(config)
    classifier.load_data()
    
    # Run classification
    score_column = config.get('tier_column', 'FantaMedia')
    player_tiers = classifier.classify_players_by_tier(score_column)
    tier_analysis = classifier.analyze_tier_distribution()
    
    # Generate outputs
    plot_path = classifier.create_visualizations(output_dir)
    csv_path = classifier.export_results(output_dir)
    
    return {
        'player_tiers': player_tiers,
        'tier_analysis': tier_analysis,
        'summary_stats': classifier.get_summary_stats(),
        'plot_path': plot_path,
        'csv_path': csv_path
    }


def assign_tiers(players_df: pd.DataFrame, score_column: str = 'FantaMedia') -> pd.DataFrame:
    """Simple function to assign tiers to a DataFrame of players."""
    df = players_df.copy()
    
    # Calculate quantile thresholds
    scores = df[score_column].dropna()
    elite_threshold = scores.quantile(0.90)
    high_threshold = scores.quantile(0.75)
    medium_threshold = scores.quantile(0.50)
    low_threshold = scores.quantile(0.25)
    
    # Assign tiers
    def assign_tier(score):
        if pd.isna(score):
            return 'Unknown'
        elif score >= elite_threshold:
            return 'Elite'
        elif score >= high_threshold:
            return 'High'
        elif score >= medium_threshold:
            return 'Medium'
        elif score >= low_threshold:
            return 'Low'
        else:
            return 'Bench'
    
    df['Tier'] = df[score_column].apply(assign_tier)
    return df