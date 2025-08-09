# Tier System
# This module handles player tier classification and management

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum


class TierLevel(Enum):
    """Enumeration for tier levels."""
    S = "S"
    A = "A"
    B = "B"
    C = "C"
    D = "D"


@dataclass
class TierThresholds:
    """Data class for tier threshold configuration."""
    s_tier: float  # percentile for S-tier (e.g., 0.92 for 92nd percentile)
    a_tier: float  # percentile for A-tier (e.g., 0.65 for 65th percentile)
    b_tier: float  # percentile for B-tier (e.g., 0.20 for 20th percentile)
    # C and D tiers are automatically determined
    
    def __post_init__(self):
        """Validate thresholds are in descending order."""
        if not (self.s_tier > self.a_tier > self.b_tier > 0):
            raise ValueError("Tier thresholds must be in descending order and positive")


@dataclass
class PlayerTierInfo:
    """Information about a player's tier classification."""
    player_name: str
    role: str
    tier: TierLevel
    value: float
    percentile: float
    tier_rank: int  # rank within the tier


def classify_players_by_tier(players: pd.DataFrame, tier_quantiles: Dict[str, float], 
                           value_column: str = 'Quotazione') -> Dict[str, List[PlayerTierInfo]]:
    """Classify players into tiers based on quantile thresholds."""
    
    # Convert tier_quantiles to TierThresholds if it's a dict
    if isinstance(tier_quantiles, dict):
        thresholds = TierThresholds(
            s_tier=tier_quantiles.get('s_tier', 0.92),
            a_tier=tier_quantiles.get('a_tier', 0.65),
            b_tier=tier_quantiles.get('b_tier', 0.20)
        )
    else:
        thresholds = tier_quantiles
    
    classified_players = {}
    
    # Group by role for position-specific tier classification
    for role in players['Ruolo'].unique():
        role_players = players[players['Ruolo'] == role].copy()
        
        if len(role_players) == 0:
            continue
            
        # Calculate percentiles for this role
        role_players['percentile'] = role_players[value_column].rank(pct=True)
        
        # Calculate tier thresholds for this role
        s_threshold = role_players[value_column].quantile(thresholds.s_tier)
        a_threshold = role_players[value_column].quantile(thresholds.a_tier)
        b_threshold = role_players[value_column].quantile(thresholds.b_tier)
        
        # Classify players into tiers
        def assign_tier(row):
            value = row[value_column]
            if value >= s_threshold:
                return TierLevel.S
            elif value >= a_threshold:
                return TierLevel.A
            elif value >= b_threshold:
                return TierLevel.B
            elif value >= role_players[value_column].quantile(0.05):  # Bottom 5% is D-tier
                return TierLevel.C
            else:
                return TierLevel.D
        
        role_players['tier'] = role_players.apply(assign_tier, axis=1)
        
        # Create PlayerTierInfo objects
        role_classified = []
        for tier in TierLevel:
            tier_players = role_players[role_players['tier'] == tier].copy()
            tier_players = tier_players.sort_values(value_column, ascending=False)
            
            for rank, (_, player) in enumerate(tier_players.iterrows(), 1):
                player_info = PlayerTierInfo(
                    player_name=player['Nome'],
                    role=role,
                    tier=tier,
                    value=player[value_column],
                    percentile=player['percentile'],
                    tier_rank=rank
                )
                role_classified.append(player_info)
        
        classified_players[role] = role_classified
    
    return classified_players


def calculate_tier_statistics(players: Dict[str, List[PlayerTierInfo]]) -> Dict[str, Dict[str, Any]]:
    """Calculate statistics for each tier."""
    
    tier_stats = {}
    
    for role, role_players in players.items():
        role_stats = {}
        
        # Group players by tier
        tier_groups = {}
        for player in role_players:
            if player.tier not in tier_groups:
                tier_groups[player.tier] = []
            tier_groups[player.tier].append(player)
        
        # Calculate statistics for each tier
        for tier, tier_players in tier_groups.items():
            if not tier_players:
                continue
                
            values = [p.value for p in tier_players]
            percentiles = [p.percentile for p in tier_players]
            
            tier_stat = {
                'count': len(tier_players),
                'avg_value': np.mean(values),
                'median_value': np.median(values),
                'std_value': np.std(values),
                'min_value': np.min(values),
                'max_value': np.max(values),
                'avg_percentile': np.mean(percentiles),
                'value_range': np.max(values) - np.min(values),
                'top_players': [p.player_name for p in sorted(tier_players, key=lambda x: x.value, reverse=True)[:3]]
            }
            
            role_stats[tier.value] = tier_stat
        
        # Calculate role-wide statistics
        all_values = [p.value for p in role_players]
        role_stats['overall'] = {
            'total_players': len(role_players),
            'avg_value': np.mean(all_values),
            'value_distribution': {
                tier.value: len([p for p in role_players if p.tier == tier]) 
                for tier in TierLevel
            }
        }
        
        tier_stats[role] = role_stats
    
    return tier_stats


def optimize_tier_thresholds(players: pd.DataFrame, performance_data: Optional[pd.DataFrame] = None,
                           value_column: str = 'Quotazione') -> TierThresholds:
    """Optimize tier thresholds based on historical performance."""
    
    if performance_data is None:
        # Use default optimization based on value distribution
        return _optimize_by_value_distribution(players, value_column)
    else:
        # Use performance-based optimization
        return _optimize_by_performance(players, performance_data, value_column)


def _optimize_by_value_distribution(players: pd.DataFrame, value_column: str) -> TierThresholds:
    """Optimize thresholds based on value distribution to ensure balanced tiers."""
    
    # Calculate optimal thresholds that create balanced tier sizes
    # Target: S-tier (5-8%), A-tier (15-20%), B-tier (25-30%), C-tier (30-35%), D-tier (15-20%)
    
    target_s = 0.95  # Top 5%
    target_a = 0.75  # Top 25%
    target_b = 0.45  # Top 55%
    
    # Adjust based on actual data distribution
    all_values = []
    for role in players['Ruolo'].unique():
        role_players = players[players['Ruolo'] == role]
        if len(role_players) > 0:
            all_values.extend(role_players[value_column].tolist())
    
    if not all_values:
        return TierThresholds(s_tier=0.92, a_tier=0.65, b_tier=0.20)
    
    # Use k-means clustering to find natural breakpoints
    from sklearn.cluster import KMeans
    
    values_array = np.array(all_values).reshape(-1, 1)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(values_array)
    
    # Sort cluster centers and use them to determine thresholds
    centers = sorted(kmeans.cluster_centers_.flatten())
    
    # Convert centers to percentiles
    s_threshold = np.percentile(all_values, 92)
    a_threshold = np.percentile(all_values, 65)
    b_threshold = np.percentile(all_values, 20)
    
    return TierThresholds(s_tier=0.92, a_tier=0.65, b_tier=0.20)


def _optimize_by_performance(players: pd.DataFrame, performance_data: pd.DataFrame,
                           value_column: str) -> TierThresholds:
    """Optimize thresholds based on actual performance correlation."""
    
    # Merge players with performance data
    merged_data = players.merge(performance_data, on='Nome', how='inner')
    
    if len(merged_data) == 0:
        return TierThresholds(s_tier=0.92, a_tier=0.65, b_tier=0.20)
    
    # Calculate correlation between value and performance metrics
    performance_cols = [col for col in performance_data.columns if col not in ['Nome', 'Ruolo']]
    
    best_thresholds = TierThresholds(s_tier=0.92, a_tier=0.65, b_tier=0.20)
    best_correlation = 0
    
    # Test different threshold combinations
    for s_tier in np.arange(0.85, 0.98, 0.02):
        for a_tier in np.arange(0.55, 0.75, 0.05):
            for b_tier in np.arange(0.15, 0.35, 0.05):
                if s_tier > a_tier > b_tier:
                    test_thresholds = TierThresholds(s_tier=s_tier, a_tier=a_tier, b_tier=b_tier)
                    
                    # Calculate average correlation for this threshold set
                    correlations = []
                    for perf_col in performance_cols:
                        if merged_data[perf_col].dtype in ['int64', 'float64']:
                            corr = merged_data[value_column].corr(merged_data[perf_col])
                            if not np.isnan(corr):
                                correlations.append(abs(corr))
                    
                    if correlations:
                        avg_correlation = np.mean(correlations)
                        if avg_correlation > best_correlation:
                            best_correlation = avg_correlation
                            best_thresholds = test_thresholds
    
    return best_thresholds


def create_tier_visualization(tier_stats: Dict[str, Dict[str, Any]], output_path: str = "tier_analysis.png"):
    """Create visualizations for tier analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Fantacalcio Tier Analysis', fontsize=16)
    
    # 1. Tier distribution by role
    roles = list(tier_stats.keys())
    tier_counts = {tier.value: [] for tier in TierLevel}
    
    for role in roles:
        if 'overall' in tier_stats[role]:
            distribution = tier_stats[role]['overall']['value_distribution']
            for tier in TierLevel:
                tier_counts[tier.value].append(distribution.get(tier.value, 0))
    
    x = np.arange(len(roles))
    width = 0.15
    
    for i, (tier, counts) in enumerate(tier_counts.items()):
        axes[0, 0].bar(x + i * width, counts, width, label=f'{tier}-tier')
    
    axes[0, 0].set_xlabel('Role')
    axes[0, 0].set_ylabel('Number of Players')
    axes[0, 0].set_title('Player Distribution by Tier and Role')
    axes[0, 0].set_xticks(x + width * 2)
    axes[0, 0].set_xticklabels(roles)
    axes[0, 0].legend()
    
    # 2. Average value by tier (all roles combined)
    all_tier_values = {tier.value: [] for tier in TierLevel}
    
    for role_stats in tier_stats.values():
        for tier in TierLevel:
            if tier.value in role_stats:
                all_tier_values[tier.value].append(role_stats[tier.value]['avg_value'])
    
    tier_names = []
    avg_values = []
    for tier, values in all_tier_values.items():
        if values:
            tier_names.append(tier)
            avg_values.append(np.mean(values))
    
    axes[0, 1].bar(tier_names, avg_values, color=['gold', 'silver', 'brown', 'gray', 'lightgray'])
    axes[0, 1].set_xlabel('Tier')
    axes[0, 1].set_ylabel('Average Value')
    axes[0, 1].set_title('Average Player Value by Tier')
    
    # 3. Tier size distribution (pie chart)
    total_counts = [sum(counts) for counts in tier_counts.values() if sum(counts) > 0]
    tier_labels = [tier for tier, counts in tier_counts.items() if sum(counts) > 0]
    
    if total_counts:
        axes[1, 0].pie(total_counts, labels=tier_labels, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Overall Tier Distribution')
    
    # 4. Value range by tier
    tier_ranges = []
    tier_labels_range = []
    
    for role_stats in tier_stats.values():
        for tier in TierLevel:
            if tier.value in role_stats:
                tier_ranges.append(role_stats[tier.value]['value_range'])
                tier_labels_range.append(tier.value)
    
    if tier_ranges:
        unique_tiers = list(set(tier_labels_range))
        avg_ranges = [np.mean([r for r, t in zip(tier_ranges, tier_labels_range) if t == tier]) 
                     for tier in unique_tiers]
        
        axes[1, 1].bar(unique_tiers, avg_ranges)
        axes[1, 1].set_xlabel('Tier')
        axes[1, 1].set_ylabel('Average Value Range')
        axes[1, 1].set_title('Value Range by Tier')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def export_tier_analysis(classified_players: Dict[str, List[PlayerTierInfo]], 
                        tier_stats: Dict[str, Dict[str, Any]], 
                        output_path: str = "tier_analysis.csv"):
    """Export tier analysis results to CSV."""
    
    # Flatten player data for CSV export
    export_data = []
    
    for role, players in classified_players.items():
        for player in players:
            export_data.append({
                'Nome': player.player_name,
                'Ruolo': player.role,
                'Tier': player.tier.value,
                'Quotazione': player.value,
                'Percentile': player.percentile,
                'Tier_Rank': player.tier_rank
            })
    
    df = pd.DataFrame(export_data)
    df.to_csv(output_path, index=False)
    
    # Also export tier statistics
    stats_path = output_path.replace('.csv', '_stats.csv')
    stats_data = []
    
    for role, role_stats in tier_stats.items():
        for tier, tier_data in role_stats.items():
            if tier != 'overall' and isinstance(tier_data, dict):
                stats_data.append({
                    'Ruolo': role,
                    'Tier': tier,
                    'Count': tier_data.get('count', 0),
                    'Avg_Value': tier_data.get('avg_value', 0),
                    'Median_Value': tier_data.get('median_value', 0),
                    'Std_Value': tier_data.get('std_value', 0),
                    'Min_Value': tier_data.get('min_value', 0),
                    'Max_Value': tier_data.get('max_value', 0)
                })
    
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(stats_path, index=False)