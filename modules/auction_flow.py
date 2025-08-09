#!/usr/bin/env python3
"""
Auction Flow Analysis Module

Analyzes fantasy football auction dynamics, bidding patterns, and optimal strategies.
Provides insights into nomination strategies, budget allocation, and market timing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from datetime import datetime
import networkx as nx
from matplotlib.patches import FancyArrowPatch


@dataclass
class AuctionStrategy:
    """Data class for auction strategy recommendations."""
    player_name: str
    role: str
    tier: str
    nomination_round: int
    target_price: float
    max_bid: float
    strategy_type: str  # 'aggressive', 'patient', 'value'
    confidence: float


@dataclass
class BudgetAllocation:
    """Data class for budget allocation by position and tier."""
    role: str
    tier: str
    allocation_pct: float
    target_spend: float
    player_count: int
    avg_price: float


class AuctionFlowAnalysis:
    """Analyzes auction dynamics and provides strategic recommendations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_path = Path(config.get('data_path', 'data/Quotazioni_Fantacalcio_Stagione_2025_26.xlsx'))
        self.players_df = None
        self.auction_strategies: List[AuctionStrategy] = []
        self.budget_allocations: List[BudgetAllocation] = []
        
        # Auction parameters
        self.total_budget = config.get('total_budget', 500)
        self.num_teams = config.get('num_teams', 8)
        self.roster_size = config.get('roster_size', 25)
        self.starting_lineup = config.get('starting_lineup', {
            'P': 1, 'D': 4, 'C': 4, 'A': 2
        })
        
        # Strategy parameters
        self.tier_thresholds = config.get('tier_thresholds', {
            'S': 0.90, 'A': 0.75, 'B': 0.50, 'C': 0.25
        })
        self.nomination_rounds = config.get('nomination_rounds', 10)
        
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
    
    def categorize_players_by_tier(self):
        """Categorize players into tiers based on price percentiles within each position."""
        logger.info("Categorizing players by tier...")
        
        self.players_df['tier'] = 'D'  # Default tier
        
        for role in self.players_df['role'].unique():
            role_players = self.players_df[self.players_df['role'] == role]
            
            # Calculate percentiles for this position
            price_percentiles = role_players['price'].rank(pct=True)
            
            # Assign tiers based on percentiles
            tier_conditions = [
                (price_percentiles >= self.tier_thresholds['S'], 'S'),
                (price_percentiles >= self.tier_thresholds['A'], 'A'),
                (price_percentiles >= self.tier_thresholds['B'], 'B'),
                (price_percentiles >= self.tier_thresholds['C'], 'C')
            ]
            
            for condition, tier in tier_conditions:
                self.players_df.loc[
                    (self.players_df['role'] == role) & condition, 'tier'
                ] = tier
        
        # Log tier distribution
        tier_counts = self.players_df.groupby(['role', 'tier']).size().unstack(fill_value=0)
        logger.info(f"Player tier distribution:\n{tier_counts}")
    
    def analyze_budget_allocation(self):
        """Analyze optimal budget allocation across positions and tiers."""
        logger.info("Analyzing budget allocation...")
        
        self.budget_allocations = []
        
        # Calculate market values by position and tier
        for role in self.players_df['role'].unique():
            role_data = self.players_df[self.players_df['role'] == role]
            
            for tier in ['S', 'A', 'B', 'C', 'D']:
                tier_data = role_data[role_data['tier'] == tier]
                
                if len(tier_data) == 0:
                    continue
                
                # Calculate allocation metrics
                avg_price = tier_data['price'].mean()
                player_count = len(tier_data)
                
                # Estimate how much budget should be allocated to this tier/position
                # Based on starting lineup requirements and tier importance
                starting_need = self.starting_lineup.get(role, 0)
                tier_weight = {'S': 0.4, 'A': 0.3, 'B': 0.2, 'C': 0.08, 'D': 0.02}.get(tier, 0.02)
                
                allocation_pct = (starting_need / sum(self.starting_lineup.values())) * tier_weight
                target_spend = self.total_budget * allocation_pct
                
                allocation = BudgetAllocation(
                    role=role,
                    tier=tier,
                    allocation_pct=allocation_pct,
                    target_spend=target_spend,
                    player_count=player_count,
                    avg_price=avg_price
                )
                
                self.budget_allocations.append(allocation)
        
        logger.success(f"Analyzed budget allocation for {len(self.budget_allocations)} role-tier combinations")
    
    def generate_auction_strategies(self):
        """Generate specific auction strategies for top players."""
        logger.info("Generating auction strategies...")
        
        self.auction_strategies = []
        
        # Focus on top tiers (S and A) for strategic recommendations
        strategic_players = self.players_df[
            (self.players_df['tier'].isin(['S', 'A'])) & 
            (self.players_df['price'] >= 10)  # Focus on significant investments
        ].copy()
        
        # Sort by price within each position
        strategic_players = strategic_players.sort_values(['role', 'price'], ascending=[True, False])
        
        for _, player in strategic_players.iterrows():
            # Determine nomination round (earlier for higher tiers)
            if player['tier'] == 'S':
                nomination_round = np.random.randint(1, 4)
                strategy_type = 'aggressive'
                confidence = 0.9
            elif player['tier'] == 'A':
                nomination_round = np.random.randint(3, 7)
                strategy_type = 'patient' if player['price'] > 25 else 'value'
                confidence = 0.75
            else:
                nomination_round = np.random.randint(5, self.nomination_rounds)
                strategy_type = 'value'
                confidence = 0.6
            
            # Calculate target and max bid
            base_price = player['price']
            
            if strategy_type == 'aggressive':
                target_price = base_price * 1.1  # Willing to pay 10% premium
                max_bid = base_price * 1.25
            elif strategy_type == 'patient':
                target_price = base_price * 0.95  # Try to get slight discount
                max_bid = base_price * 1.1
            else:  # value
                target_price = base_price * 0.85  # Look for good value
                max_bid = base_price * 1.0
            
            strategy = AuctionStrategy(
                player_name=player['name'],
                role=player['role'],
                tier=player['tier'],
                nomination_round=nomination_round,
                target_price=target_price,
                max_bid=max_bid,
                strategy_type=strategy_type,
                confidence=confidence
            )
            
            self.auction_strategies.append(strategy)
        
        # Sort strategies by nomination round
        self.auction_strategies.sort(key=lambda x: x.nomination_round)
        
        logger.success(f"Generated {len(self.auction_strategies)} auction strategies")
    
    def create_flow_diagram(self, output_dir: Path) -> Path:
        """Create auction flow diagram showing nomination strategy."""
        logger.info("Creating auction flow diagram...")
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        fig.suptitle('Auction Flow Analysis', fontsize=16, fontweight='bold')
        
        # Left plot: Nomination timeline
        ax1.set_title('Nomination Strategy Timeline')
        
        # Group strategies by round
        round_strategies = {}
        for strategy in self.auction_strategies[:20]:  # Top 20 for visibility
            round_num = strategy.nomination_round
            if round_num not in round_strategies:
                round_strategies[round_num] = []
            round_strategies[round_num].append(strategy)
        
        # Plot timeline
        y_pos = 0
        colors = {'aggressive': 'red', 'patient': 'orange', 'value': 'green'}
        
        for round_num in sorted(round_strategies.keys()):
            strategies = round_strategies[round_num]
            
            # Round header
            ax1.text(0.1, y_pos, f"Round {round_num}", fontweight='bold', fontsize=12)
            y_pos -= 0.5
            
            for strategy in strategies:
                color = colors.get(strategy.strategy_type, 'blue')
                player_text = f"{strategy.player_name[:15]}... ({strategy.role}-{strategy.tier})"
                price_text = f"Target: {strategy.target_price:.0f}, Max: {strategy.max_bid:.0f}"
                
                ax1.text(0.2, y_pos, player_text, color=color, fontsize=10)
                ax1.text(0.2, y_pos-0.2, price_text, color='gray', fontsize=8)
                y_pos -= 0.8
            
            y_pos -= 0.3  # Extra space between rounds
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(y_pos, 1)
        ax1.axis('off')
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], color=color, lw=2, label=strategy_type.title()) 
                          for strategy_type, color in colors.items()]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # Right plot: Budget allocation heatmap
        ax2.set_title('Budget Allocation by Position and Tier')
        
        # Prepare data for heatmap
        allocation_matrix = pd.DataFrame(index=['P', 'D', 'C', 'A'], columns=['S', 'A', 'B', 'C', 'D'])
        
        for allocation in self.budget_allocations:
            if allocation.role in allocation_matrix.index and allocation.tier in allocation_matrix.columns:
                allocation_matrix.loc[allocation.role, allocation.tier] = allocation.target_spend
        
        # Fill NaN with 0 and convert to numeric
        allocation_matrix = allocation_matrix.fillna(0).astype(float)
        
        # Create heatmap
        sns.heatmap(allocation_matrix, annot=True, fmt='.0f', cmap='YlOrRd', 
                   ax=ax2, cbar_kws={'label': 'Target Spend (€)'})
        ax2.set_xlabel('Player Tier')
        ax2.set_ylabel('Position')
        
        plt.tight_layout()
        
        # Save diagram
        diagram_filename = f"auction_flow_diagram_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        diagram_path = output_dir / diagram_filename
        plt.savefig(diagram_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.success(f"Auction flow diagram saved to {diagram_path}")
        return diagram_path
    
    def create_budget_charts(self, output_dir: Path) -> Path:
        """Create detailed budget allocation charts."""
        logger.info("Creating budget allocation charts...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Budget Allocation Analysis', fontsize=16, fontweight='bold')
        
        # Prepare data
        allocation_df = pd.DataFrame([
            {
                'role': a.role,
                'tier': a.tier,
                'allocation_pct': a.allocation_pct * 100,
                'target_spend': a.target_spend,
                'avg_price': a.avg_price,
                'player_count': a.player_count
            }
            for a in self.budget_allocations
        ])
        
        # 1. Budget allocation by position
        position_budget = allocation_df.groupby('role')['target_spend'].sum()
        axes[0, 0].pie(position_budget.values, labels=position_budget.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Budget Allocation by Position')
        
        # 2. Budget allocation by tier
        tier_budget = allocation_df.groupby('tier')['target_spend'].sum()
        tier_colors = {'S': 'gold', 'A': 'silver', 'B': 'bronze', 'C': 'lightblue', 'D': 'lightgray'}
        colors = [tier_colors.get(tier, 'gray') for tier in tier_budget.index]
        axes[0, 1].bar(tier_budget.index, tier_budget.values, color=colors)
        axes[0, 1].set_title('Budget Allocation by Tier')
        axes[0, 1].set_xlabel('Tier')
        axes[0, 1].set_ylabel('Target Spend (€)')
        
        # 3. Average price by position and tier
        pivot_avg_price = allocation_df.pivot(index='role', columns='tier', values='avg_price')
        sns.heatmap(pivot_avg_price, annot=True, fmt='.1f', cmap='viridis', ax=axes[1, 0])
        axes[1, 0].set_title('Average Price by Position and Tier')
        
        # 4. Player availability by tier
        tier_counts = allocation_df.groupby('tier')['player_count'].sum()
        colors = [tier_colors.get(tier, 'gray') for tier in tier_counts.index]
        axes[1, 1].bar(tier_counts.index, tier_counts.values, color=colors)
        axes[1, 1].set_title('Player Availability by Tier')
        axes[1, 1].set_xlabel('Tier')
        axes[1, 1].set_ylabel('Number of Players')
        
        plt.tight_layout()
        
        # Save charts
        charts_filename = f"budget_allocation_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        charts_path = output_dir / charts_filename
        plt.savefig(charts_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.success(f"Budget allocation charts saved to {charts_path}")
        return charts_path
    
    def export_results(self, output_dir: Path) -> Path:
        """Export auction analysis results to CSV."""
        logger.info("Exporting auction analysis results...")
        
        # Prepare strategy export data
        strategy_data = []
        for strategy in self.auction_strategies:
            strategy_data.append({
                'player_name': strategy.player_name,
                'role': strategy.role,
                'tier': strategy.tier,
                'nomination_round': strategy.nomination_round,
                'target_price': strategy.target_price,
                'max_bid': strategy.max_bid,
                'strategy_type': strategy.strategy_type,
                'confidence': strategy.confidence
            })
        
        # Create DataFrames
        strategies_df = pd.DataFrame(strategy_data)
        
        allocation_data = []
        for allocation in self.budget_allocations:
            allocation_data.append({
                'role': allocation.role,
                'tier': allocation.tier,
                'allocation_pct': allocation.allocation_pct * 100,
                'target_spend': allocation.target_spend,
                'player_count': allocation.player_count,
                'avg_price': allocation.avg_price
            })
        
        allocations_df = pd.DataFrame(allocation_data)
        
        # Export to CSV files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        strategies_path = output_dir / f"auction_strategies_{timestamp}.csv"
        strategies_df.to_csv(strategies_path, index=False)
        
        allocations_path = output_dir / f"budget_allocations_{timestamp}.csv"
        allocations_df.to_csv(allocations_path, index=False)
        
        logger.success(f"Results exported to {strategies_path} and {allocations_path}")
        return strategies_path
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the auction analysis."""
        if not self.auction_strategies or not self.budget_allocations:
            return {}
        
        # Strategy statistics
        strategy_types = [s.strategy_type for s in self.auction_strategies]
        strategy_counts = {stype: strategy_types.count(stype) for stype in set(strategy_types)}
        
        total_target_spend = sum(s.target_price for s in self.auction_strategies)
        avg_confidence = np.mean([s.confidence for s in self.auction_strategies])
        
        # Budget statistics
        total_allocated = sum(a.target_spend for a in self.budget_allocations)
        position_allocations = {}
        for allocation in self.budget_allocations:
            if allocation.role not in position_allocations:
                position_allocations[allocation.role] = 0
            position_allocations[allocation.role] += allocation.target_spend
        
        return {
            'total_strategies': len(self.auction_strategies),
            'strategy_distribution': strategy_counts,
            'total_target_spend': total_target_spend,
            'avg_confidence': avg_confidence,
            'total_budget_allocated': total_allocated,
            'budget_utilization': (total_allocated / self.total_budget) * 100,
            'position_allocations': position_allocations,
            'nomination_rounds_used': len(set(s.nomination_round for s in self.auction_strategies))
        }
