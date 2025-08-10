#!/usr/bin/env python3
"""
Auction Strategy Module

Develops and implements auction strategies for fantasy football drafts,
including budget allocation, bidding logic, and opponent modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random
from collections import defaultdict


class BiddingStrategy(Enum):
    """Different bidding strategies."""
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    VALUE_HUNTING = "value_hunting"
    STARS_AND_SCRUBS = "stars_and_scrubs"


class PlayerPriority(Enum):
    """Player priority levels."""
    MUST_HAVE = "must_have"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    AVOID = "avoid"


@dataclass
class BudgetAllocation:
    """Budget allocation for different roles."""
    goalkeeper: float
    defenders: float
    midfielders: float
    forwards: float
    bench: float
    
    def total(self) -> float:
        return self.goalkeeper + self.defenders + self.midfielders + self.forwards + self.bench
    
    def validate(self, total_budget: float) -> bool:
        return abs(self.total() - total_budget) < 0.01


@dataclass
class PlayerTarget:
    """Target player information."""
    name: str
    role: str
    max_bid: float
    priority: PlayerPriority
    expected_price: float
    tier: str = ""
    backup_players: List[str] = field(default_factory=list)


@dataclass
class AuctionState:
    """Current state of the auction."""
    remaining_budget: float
    remaining_slots: Dict[str, int]
    acquired_players: List[str]
    failed_bids: List[str]
    current_round: int
    total_players_sold: int


@dataclass
class OpponentModel:
    """Model of opponent behavior."""
    name: str
    aggression_level: float  # 0-1
    budget_remaining: float
    preferred_formation: str
    bidding_pattern: Dict[str, float]  # role -> avg_bid_multiplier
    recent_bids: List[float]


class AuctionSimulator:
    """Simulates auction scenarios and develops strategies."""
    
    def __init__(self, config: dict):
        self.config = config
        self.data = None
        self.budget_allocation = None
        self.player_targets = {}
        self.opponents = []
        self.auction_history = []
        
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
    
    def calculate_budget_allocation(self, strategy: BiddingStrategy = BiddingStrategy.BALANCED) -> BudgetAllocation:
        """Calculate optimal budget allocation based on strategy."""
        logger.info(f"Calculating budget allocation for {strategy.value} strategy")
        
        total_budget = self.config.get('budget_cap', self.config.get('total_budget', 400))
        formation = self.config.get('formation', '3-5-2')
        
        # Parse formation
        formation_parts = formation.split('-')
        defenders_needed = int(formation_parts[0])
        midfielders_needed = int(formation_parts[1])
        forwards_needed = int(formation_parts[2])
        
        # Strategy-based allocation percentages
        if strategy == BiddingStrategy.AGGRESSIVE:
            # Spend more on top players
            allocations = {
                'goalkeeper': 0.08,
                'defenders': 0.25,
                'midfielders': 0.35,
                'forwards': 0.25,
                'bench': 0.07
            }
        elif strategy == BiddingStrategy.CONSERVATIVE:
            # More balanced, save money for depth
            allocations = {
                'goalkeeper': 0.10,
                'defenders': 0.22,
                'midfielders': 0.28,
                'forwards': 0.22,
                'bench': 0.18
            }
        elif strategy == BiddingStrategy.STARS_AND_SCRUBS:
            # Spend big on few stars, minimal on others
            allocations = {
                'goalkeeper': 0.05,
                'defenders': 0.20,
                'midfielders': 0.45,
                'forwards': 0.25,
                'bench': 0.05
            }
        elif strategy == BiddingStrategy.VALUE_HUNTING:
            # Look for undervalued players
            allocations = {
                'goalkeeper': 0.12,
                'defenders': 0.24,
                'midfielders': 0.28,
                'forwards': 0.24,
                'bench': 0.12
            }
        else:  # BALANCED
            allocations = {
                'goalkeeper': 0.10,
                'defenders': 0.25,
                'midfielders': 0.30,
                'forwards': 0.25,
                'bench': 0.10
            }
        
        # Adjust based on formation
        if midfielders_needed > 4:  # Midfield-heavy formation
            allocations['midfielders'] += 0.05
            allocations['forwards'] -= 0.03
            allocations['defenders'] -= 0.02
        elif forwards_needed > 2:  # Attack-heavy formation
            allocations['forwards'] += 0.05
            allocations['midfielders'] -= 0.03
            allocations['defenders'] -= 0.02
        
        self.budget_allocation = BudgetAllocation(
            goalkeeper=total_budget * allocations['goalkeeper'],
            defenders=total_budget * allocations['defenders'],
            midfielders=total_budget * allocations['midfielders'],
            forwards=total_budget * allocations['forwards'],
            bench=total_budget * allocations['bench']
        )
        
        logger.info(f"Budget allocation: GK={self.budget_allocation.goalkeeper:.0f}, "
                   f"DEF={self.budget_allocation.defenders:.0f}, "
                   f"MID={self.budget_allocation.midfielders:.0f}, "
                   f"FWD={self.budget_allocation.forwards:.0f}, "
                   f"BENCH={self.budget_allocation.bench:.0f}")
        
        return self.budget_allocation
    
    def identify_player_targets(self, tier_data: Optional[Dict] = None) -> Dict[str, PlayerTarget]:
        """Identify target players based on strategy and tiers."""
        logger.info("Identifying player targets...")
        
        if self.budget_allocation is None:
            logger.error("Budget allocation is None - cannot identify targets")
            return {}
        
        targets = {}
        
        # Group players by role
        role_groups = self.data.groupby('Ruolo')
        
        for role, role_data in role_groups:
            # Sort by FantaMedia descending
            sorted_players = role_data.sort_values('FantaMedia', ascending=False)
            
            # Determine budget for this role
            if role == 'P':  # Portiere
                role_budget = self.budget_allocation.goalkeeper
                target_count = 1
            elif role == 'D':  # Difensore
                role_budget = self.budget_allocation.defenders
                target_count = 3  # Assuming we want 3 defenders
            elif role == 'C':  # Centrocampista
                role_budget = self.budget_allocation.midfielders
                target_count = 5  # Assuming we want 5 midfielders
            elif role == 'A':  # Attaccante
                role_budget = self.budget_allocation.forwards
                target_count = 2  # Assuming we want 2 forwards
            else:
                continue
            
            # Calculate average budget per player for this role
            avg_budget_per_player = role_budget / target_count
            
            # Get target multiplier from config (default 3.0 for backward compatibility)
            target_multiplier = self.config.get('target_multiplier', 3.0)
            max_targets = int(target_count * target_multiplier)
            
            # Identify targets with different priorities
            for i, (_, player) in enumerate(sorted_players.head(max_targets).iterrows()):
                player_name = player['Nome']
                quotazione = player['Quotazione']
                fantamedia = player['FantaMedia']
                
                # Determine priority based on ranking and value
                if i < target_count:
                    if fantamedia > sorted_players['FantaMedia'].quantile(0.9):
                        priority = PlayerPriority.MUST_HAVE
                    else:
                        priority = PlayerPriority.HIGH
                elif i < target_count * 2:
                    priority = PlayerPriority.MEDIUM
                else:
                    priority = PlayerPriority.LOW
                
                # Calculate max bid (based on quotazione and strategy)
                base_bid = quotazione
                if priority == PlayerPriority.MUST_HAVE:
                    max_bid = min(base_bid * 1.3, avg_budget_per_player * 1.5)
                elif priority == PlayerPriority.HIGH:
                    max_bid = min(base_bid * 1.2, avg_budget_per_player * 1.2)
                elif priority == PlayerPriority.MEDIUM:
                    max_bid = min(base_bid * 1.1, avg_budget_per_player)
                else:
                    max_bid = min(base_bid, avg_budget_per_player * 0.8)
                
                # Expected price (slightly below quotazione)
                expected_price = quotazione * 0.95
                
                targets[player_name] = PlayerTarget(
                    name=player_name,
                    role=role,
                    max_bid=max_bid,
                    priority=priority,
                    expected_price=expected_price,
                    tier=tier_data.get(player_name).tier.value if tier_data and tier_data.get(player_name) else ''
                )
        
        self.player_targets = targets
        logger.success(f"Identified {len(targets)} player targets")
        
        return targets
    
    def simulate_opponent_behavior(self, num_opponents: int = 7) -> List[OpponentModel]:
        """Create models for opponent behavior."""
        logger.info(f"Creating models for {num_opponents} opponents")
        
        opponents = []
        formations = ['3-5-2', '4-4-2', '4-3-3', '3-4-3', '5-3-2']
        
        for i in range(num_opponents):
            # Random opponent characteristics
            aggression = random.uniform(0.3, 0.9)
            formation = random.choice(formations)
            
            # Bidding patterns based on aggression
            if aggression > 0.7:
                bidding_pattern = {
                    'Portiere': 1.1,
                    'Difensore': 1.15,
                    'Centrocampista': 1.2,
                    'Attaccante': 1.25
                }
            elif aggression > 0.5:
                bidding_pattern = {
                    'Portiere': 1.0,
                    'Difensore': 1.05,
                    'Centrocampista': 1.1,
                    'Attaccante': 1.1
                }
            else:
                bidding_pattern = {
                    'Portiere': 0.9,
                    'Difensore': 0.95,
                    'Centrocampista': 1.0,
                    'Attaccante': 1.0
                }
            
            opponent = OpponentModel(
                name=f"Opponent_{i+1}",
                aggression_level=aggression,
                budget_remaining=self.config.get('budget_cap', self.config.get('total_budget', 400)),
                preferred_formation=formation,
                bidding_pattern=bidding_pattern,
                recent_bids=[]
            )
            
            opponents.append(opponent)
        
        self.opponents = opponents
        return opponents
    
    def simulate_auction_round(self, player_name: str, starting_price: float) -> Tuple[str, float]:
        """Simulate a single auction round."""
        current_bid = starting_price
        current_bidder = None
        
        # Get our target for this player
        our_target = self.player_targets.get(player_name)
        our_max_bid = our_target.max_bid if our_target else starting_price * 0.8
        
        # Simulate bidding rounds
        active_bidders = ['us'] + [opp.name for opp in self.opponents]
        round_count = 0
        
        while len(active_bidders) > 1 and round_count < 20:
            round_count += 1
            new_bidders = []
            
            for bidder in active_bidders:
                if bidder == 'us':
                    # Our bidding logic
                    if our_target and current_bid < our_max_bid:
                        # Bid based on priority
                        if our_target.priority == PlayerPriority.MUST_HAVE:
                            bid_increment = max(1, current_bid * 0.05)
                        elif our_target.priority == PlayerPriority.HIGH:
                            bid_increment = max(1, current_bid * 0.03)
                        else:
                            bid_increment = 1
                        
                        new_bid = current_bid + bid_increment
                        if new_bid <= our_max_bid:
                            current_bid = new_bid
                            current_bidder = 'us'
                            new_bidders.append('us')
                else:
                    # Opponent bidding logic
                    opponent = next(opp for opp in self.opponents if opp.name == bidder)
                    
                    # Get player role for bidding pattern
                    player_role = self.data[self.data['Nome'] == player_name]['Ruolo'].iloc[0]
                    role_multiplier = opponent.bidding_pattern.get(player_role, 1.0)
                    
                    # Calculate opponent's max bid
                    base_quotazione = self.data[self.data['Nome'] == player_name]['Quotazione'].iloc[0]
                    opponent_max = base_quotazione * role_multiplier * opponent.aggression_level
                    
                    # Opponent bids if current bid is below their max
                    if current_bid < opponent_max and random.random() < opponent.aggression_level:
                        bid_increment = max(1, current_bid * random.uniform(0.02, 0.08))
                        new_bid = current_bid + bid_increment
                        
                        if new_bid <= opponent_max:
                            current_bid = new_bid
                            current_bidder = bidder
                            new_bidders.append(bidder)
            
            active_bidders = new_bidders
            
            # If no one bids, auction ends
            if not new_bidders:
                break
        
        return current_bidder or 'no_winner', current_bid
    
    def run_auction_simulation(self, num_simulations: int = 100) -> Dict[str, Any]:
        """Run multiple auction simulations."""
        logger.info(f"Running {num_simulations} auction simulations...")
        
        results = {
            'successful_acquisitions': defaultdict(int),
            'average_prices': defaultdict(list),
            'budget_utilization': [],
            'team_quality_scores': [],
            'strategy_effectiveness': {}
        }
        
        for sim in range(num_simulations):
            # Reset for each simulation
            auction_state = AuctionState(
                remaining_budget=self.config.get('budget_cap', self.config.get('total_budget', 400)),
                remaining_slots={'Portiere': 1, 'Difensore': 3, 'Centrocampista': 5, 'Attaccante': 2},
                acquired_players=[],
                failed_bids=[],
                current_round=0,
                total_players_sold=0
            )
            
            # Simulate auction for each target player
            for player_name, target in self.player_targets.items():
                if auction_state.remaining_slots.get(target.role, 0) > 0:
                    winner, final_price = self.simulate_auction_round(player_name, target.expected_price)
                    
                    if winner == 'us' and final_price <= auction_state.remaining_budget:
                        # We won the player
                        auction_state.acquired_players.append(player_name)
                        auction_state.remaining_budget -= final_price
                        auction_state.remaining_slots[target.role] -= 1
                        
                        results['successful_acquisitions'][player_name] += 1
                        results['average_prices'][player_name].append(final_price)
                    else:
                        auction_state.failed_bids.append(player_name)
            
            # Calculate team quality score
            team_score = 0
            for player_name in auction_state.acquired_players:
                player_data = self.data[self.data['Nome'] == player_name]
                if not player_data.empty:
                    team_score += player_data['FantaMedia'].iloc[0]
            
            results['team_quality_scores'].append(team_score)
            budget_cap = self.config.get('budget_cap', self.config.get('total_budget', 400))
            results['budget_utilization'].append(
                (budget_cap - auction_state.remaining_budget) / budget_cap
            )
        
        # Calculate average prices
        for player_name, prices in results['average_prices'].items():
            results['average_prices'][player_name] = np.mean(prices) if prices else 0
        
        logger.success(f"Completed {num_simulations} auction simulations")
        return results
    
    def optimize_strategy(self, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize strategy based on simulation results."""
        logger.info("Optimizing auction strategy...")
        
        optimizations = {
            'budget_adjustments': {},
            'target_adjustments': {},
            'strategy_recommendations': []
        }
        
        # Analyze budget utilization
        avg_budget_utilization = np.mean(simulation_results['budget_utilization'])
        if avg_budget_utilization < 0.85:
            optimizations['strategy_recommendations'].append(
                "Consider increasing max bids - budget underutilized"
            )
        elif avg_budget_utilization > 0.98:
            optimizations['strategy_recommendations'].append(
                "Consider reducing max bids - budget overextended"
            )
        
        # Analyze successful acquisitions
        for player_name, success_rate in simulation_results['successful_acquisitions'].items():
            success_percentage = success_rate / len(simulation_results['team_quality_scores']) * 100
            
            if success_percentage < 30 and player_name in self.player_targets:
                target = self.player_targets[player_name]
                if target.priority in [PlayerPriority.MUST_HAVE, PlayerPriority.HIGH]:
                    # Increase max bid for important players with low success rate
                    new_max_bid = target.max_bid * 1.15
                    optimizations['target_adjustments'][player_name] = {
                        'old_max_bid': target.max_bid,
                        'new_max_bid': new_max_bid,
                        'reason': f'Low success rate ({success_percentage:.1f}%)'
                    }
        
        return optimizations
    
    def create_visualizations(self, simulation_results: Dict[str, Any], output_dir: Path) -> Path:
        """Create auction strategy visualizations."""
        logger.info("Creating auction strategy visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Auction Strategy Analysis', fontsize=16, fontweight='bold')
        
        # 1. Budget Allocation
        if self.budget_allocation:
            roles = ['Goalkeeper', 'Defenders', 'Midfielders', 'Forwards', 'Bench']
            amounts = [
                self.budget_allocation.goalkeeper,
                self.budget_allocation.defenders,
                self.budget_allocation.midfielders,
                self.budget_allocation.forwards,
                self.budget_allocation.bench
            ]
            
            axes[0, 0].pie(amounts, labels=roles, autopct='%1.1f%%', startangle=90)
            axes[0, 0].set_title('Budget Allocation by Role')
        
        # 2. Success Rate by Priority
        if simulation_results['successful_acquisitions']:
            priority_success = defaultdict(list)
            
            for player_name, success_count in simulation_results['successful_acquisitions'].items():
                if player_name in self.player_targets:
                    priority = self.player_targets[player_name].priority.value
                    success_rate = success_count / len(simulation_results['team_quality_scores']) * 100
                    priority_success[priority].append(success_rate)
            
            priorities = list(priority_success.keys())
            avg_success_rates = [np.mean(rates) for rates in priority_success.values()]
            
            axes[0, 1].bar(priorities, avg_success_rates)
            axes[0, 1].set_title('Average Success Rate by Player Priority')
            axes[0, 1].set_ylabel('Success Rate (%)')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Budget Utilization Distribution
        if simulation_results['budget_utilization']:
            axes[1, 0].hist(simulation_results['budget_utilization'], bins=20, alpha=0.7)
            axes[1, 0].set_title('Budget Utilization Distribution')
            axes[1, 0].set_xlabel('Budget Utilization Ratio')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].axvline(np.mean(simulation_results['budget_utilization']), 
                              color='red', linestyle='--', label='Average')
            axes[1, 0].legend()
        
        # 4. Team Quality vs Budget Utilization
        if simulation_results['team_quality_scores'] and simulation_results['budget_utilization']:
            axes[1, 1].scatter(simulation_results['budget_utilization'], 
                              simulation_results['team_quality_scores'], alpha=0.6)
            axes[1, 1].set_title('Team Quality vs Budget Utilization')
            axes[1, 1].set_xlabel('Budget Utilization')
            axes[1, 1].set_ylabel('Team Quality Score')
            
            # Add trend line (filter out NaN values and check for variance)
            budget_util = np.array(simulation_results['budget_utilization'])
            quality_scores = np.array(simulation_results['team_quality_scores'])
            
            # Remove NaN values
            valid_mask = ~(np.isnan(budget_util) | np.isnan(quality_scores))
            if np.sum(valid_mask) > 1:  # Need at least 2 points for linear fit
                budget_util_clean = budget_util[valid_mask]
                quality_scores_clean = quality_scores[valid_mask]
                
                # Check if there's variance in the data
                if (len(np.unique(budget_util_clean)) > 1 and 
                    len(np.unique(quality_scores_clean)) > 1 and
                    np.std(budget_util_clean) > 1e-10):
                    try:
                        z = np.polyfit(budget_util_clean, quality_scores_clean, 1)
                        p = np.poly1d(z)
                        axes[1, 1].plot(budget_util_clean, 
                                       p(budget_util_clean), "r--", alpha=0.8)
                    except np.linalg.LinAlgError:
                        logger.warning("Could not fit trend line due to numerical issues")
        
        plt.tight_layout()
        
        # Save plot
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        plot_path = output_dir / f'auction_strategy_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.success(f"Visualizations saved to {plot_path}")
        return plot_path
    
    def export_results(self, simulation_results: Dict[str, Any], output_dir: Path) -> Path:
        """Export auction strategy results to CSV."""
        logger.info("Exporting auction strategy results...")
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        csv_path = output_dir / f'auction_strategy_results_{timestamp}.csv'
        
        # Prepare data for export
        export_data = []
        
        for player_name, target in self.player_targets.items():
            success_count = simulation_results['successful_acquisitions'].get(player_name, 0)
            success_rate = success_count / len(simulation_results['team_quality_scores']) * 100 if simulation_results['team_quality_scores'] else 0
            avg_price = simulation_results['average_prices'].get(player_name, 0)
            
            export_data.append({
                'Player_Name': target.name,
                'Role': target.role,
                'Priority': target.priority.value,
                'Max_Bid': target.max_bid,
                'Expected_Price': target.expected_price,
                'Success_Rate_Percent': success_rate,
                'Average_Winning_Price': avg_price,
                'Tier': target.tier
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(export_data)
        if not df.empty and 'Priority' in df.columns:
            df = df.sort_values(['Priority', 'Success_Rate_Percent'], ascending=[True, False])
        df.to_csv(csv_path, index=False)
        
        logger.success(f"Results exported to {csv_path}")
        return csv_path
    
    def get_summary_stats(self, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary statistics for the auction strategy."""
        # Always return expected keys for report builder compatibility
        stats = {
            'simulations_run': len(simulation_results.get('team_quality_scores', [])),
            'player_targets': f"{len(self.player_targets)} identified" if self.player_targets else "N/A",
            'budget_allocation': f"{self.budget_allocation.total():.0f} credits allocated" if self.budget_allocation else "N/A",
            'avg_confidence': 0.0,
            'total_target_players': len(self.player_targets),
            'high_priority_targets': len([t for t in self.player_targets.values() 
                                        if t.priority in [PlayerPriority.MUST_HAVE, PlayerPriority.HIGH]]) if self.player_targets else 0,
            'total_budget_allocated': self.budget_allocation.total() if self.budget_allocation else 0
        }
        
        # Add simulation-dependent stats if we have valid results
        if simulation_results.get('team_quality_scores'):
            stats.update({
                'average_team_quality': np.mean(simulation_results['team_quality_scores']),
                'average_budget_utilization': np.mean(simulation_results['budget_utilization']),
                'avg_confidence': min(0.95, np.mean(simulation_results['team_quality_scores']) / 100) if simulation_results['team_quality_scores'] else 0.0
            })
        else:
            stats.update({
                'average_team_quality': 0.0,
                'average_budget_utilization': 0.0
            })
        
        return stats


def build_auction_strategy(config: dict, tier_data: Optional[Dict] = None, output_dir: Path = None) -> Dict[str, Any]:
    """Main function to build auction strategy."""
    simulator = AuctionSimulator(config)
    simulator.load_data()
    
    # Build strategy components
    strategy_type = BiddingStrategy(config.get('bidding_strategy', 'balanced'))
    budget_allocation = simulator.calculate_budget_allocation(strategy_type)
    # Ensure budget_allocation is assigned to simulator instance
    simulator.budget_allocation = budget_allocation
    player_targets = simulator.identify_player_targets(tier_data)
    opponents = simulator.simulate_opponent_behavior()
    
    # Run simulations
    num_simulations = config.get('num_simulations', 100)
    simulation_results = simulator.run_auction_simulation(num_simulations)
    
    # Optimize strategy
    optimizations = simulator.optimize_strategy(simulation_results)
    
    # Generate outputs if output_dir provided
    plot_path = None
    csv_path = None
    if output_dir:
        plot_path = simulator.create_visualizations(simulation_results, output_dir)
        csv_path = simulator.export_results(simulation_results, output_dir)
    
    return {
        'budget_allocation': budget_allocation,
        'player_targets': player_targets,
        'simulation_results': simulation_results,
        'optimizations': optimizations,
        'summary_stats': simulator.get_summary_stats(simulation_results),
        'plot_path': plot_path,
        'csv_path': csv_path
    }


def build_strategy(config: dict, output_dir: Path) -> dict:
    """Simple wrapper function for building auction strategy."""
    return build_auction_strategy(config, output_dir=output_dir)