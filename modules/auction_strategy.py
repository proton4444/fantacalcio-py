# Auction Strategy
# This module implements auction strategies and bidding logic

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import random
from collections import defaultdict


class BiddingStrategy(Enum):
    """Different bidding strategy types."""
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    VALUE_BASED = "value_based"
    CONTRARIAN = "contrarian"


class AuctionPhase(Enum):
    """Different phases of the auction."""
    EARLY = "early"  # First 30% of players
    MIDDLE = "middle"  # Middle 40% of players
    LATE = "late"  # Final 30% of players


@dataclass
class MarketConditions:
    """Current market conditions during auction."""
    average_bid_multiplier: float  # How much above/below value players are going for
    competition_level: float  # 0-1 scale of how competitive the auction is
    budget_pressure: float  # 0-1 scale of how much budget pressure exists
    phase: AuctionPhase
    remaining_players: int
    total_budget_remaining: float


@dataclass
class BidDecision:
    """A bidding decision for a player."""
    player_name: str
    max_bid: float
    confidence: float  # 0-1 scale of confidence in this bid
    strategy_used: BiddingStrategy
    reasoning: str


@dataclass
class AuctionParticipant:
    """An auction participant with their strategy and constraints."""
    name: str
    budget: float
    strategy: BiddingStrategy
    risk_tolerance: float  # 0-1 scale
    position_needs: Dict[str, int]  # How many players needed per position
    current_roster: List[str]  # Current players owned
    
    def remaining_budget(self) -> float:
        """Calculate remaining budget."""
        return self.budget
    
    def needs_position(self, position: str) -> bool:
        """Check if participant needs players in this position."""
        return self.position_needs.get(position, 0) > 0


class AuctionSimulator:
    """Simulates fantasy football auction dynamics."""
    
    def __init__(self, participants: List[AuctionParticipant], base_inflation: float = 1.1):
        self.participants = participants
        self.base_inflation = base_inflation
        self.auction_history = []
        self.market_conditions = None
        
    def update_market_conditions(self, players_sold: int, total_players: int, 
                                total_budget_spent: float, total_budget: float):
        """Update market conditions based on auction progress."""
        progress = players_sold / total_players
        
        if progress < 0.3:
            phase = AuctionPhase.EARLY
        elif progress < 0.7:
            phase = AuctionPhase.MIDDLE
        else:
            phase = AuctionPhase.LATE
        
        self.market_conditions = MarketConditions(
            average_bid_multiplier=total_budget_spent / (total_budget * progress) if progress > 0 else 1.0,
            competition_level=min(1.0, len(self.participants) / 8),  # Normalize to 8-person league
            budget_pressure=total_budget_spent / total_budget,
            phase=phase,
            remaining_players=total_players - players_sold,
            total_budget_remaining=total_budget - total_budget_spent
        )


def calculate_optimal_bid(player: Dict[str, Any], market_conditions: MarketConditions, 
                         participant: AuctionParticipant, player_valuations: Dict[str, float]) -> BidDecision:
    """Calculate the optimal bid for a player given market conditions."""
    
    player_name = player['Nome']
    player_position = player['Ruolo']
    base_value = player_valuations.get(player_name, player.get('Quotazione', 0))
    
    # Base bid calculation
    base_bid = base_value
    
    # Adjust for market conditions
    market_multiplier = market_conditions.average_bid_multiplier
    competition_adjustment = 1 + (market_conditions.competition_level * 0.2)
    
    # Adjust for participant's strategy
    strategy_multiplier = _get_strategy_multiplier(participant.strategy, market_conditions.phase)
    
    # Adjust for position need
    need_multiplier = 1.2 if participant.needs_position(player_position) else 0.8
    
    # Adjust for budget constraints
    budget_ratio = participant.remaining_budget() / (participant.budget * 0.8)  # Keep 20% buffer
    budget_multiplier = min(1.5, max(0.5, budget_ratio))
    
    # Calculate final bid
    optimal_bid = base_bid * market_multiplier * competition_adjustment * strategy_multiplier * need_multiplier * budget_multiplier
    
    # Apply risk tolerance
    risk_adjustment = 1 + (participant.risk_tolerance - 0.5) * 0.3
    optimal_bid *= risk_adjustment
    
    # Ensure bid doesn't exceed budget constraints
    max_affordable = participant.remaining_budget() * 0.4  # Don't spend more than 40% on one player
    optimal_bid = min(optimal_bid, max_affordable)
    
    # Calculate confidence based on various factors
    confidence = _calculate_bid_confidence(player, market_conditions, participant, base_value, optimal_bid)
    
    # Generate reasoning
    reasoning = _generate_bid_reasoning(player, market_conditions, participant, optimal_bid, base_value)
    
    return BidDecision(
        player_name=player_name,
        max_bid=optimal_bid,
        confidence=confidence,
        strategy_used=participant.strategy,
        reasoning=reasoning
    )


def _get_strategy_multiplier(strategy: BiddingStrategy, phase: AuctionPhase) -> float:
    """Get bid multiplier based on strategy and auction phase."""
    multipliers = {
        BiddingStrategy.CONSERVATIVE: {
            AuctionPhase.EARLY: 0.9,
            AuctionPhase.MIDDLE: 0.95,
            AuctionPhase.LATE: 1.1
        },
        BiddingStrategy.AGGRESSIVE: {
            AuctionPhase.EARLY: 1.2,
            AuctionPhase.MIDDLE: 1.1,
            AuctionPhase.LATE: 1.0
        },
        BiddingStrategy.BALANCED: {
            AuctionPhase.EARLY: 1.0,
            AuctionPhase.MIDDLE: 1.0,
            AuctionPhase.LATE: 1.0
        },
        BiddingStrategy.VALUE_BASED: {
            AuctionPhase.EARLY: 0.95,
            AuctionPhase.MIDDLE: 1.0,
            AuctionPhase.LATE: 1.05
        },
        BiddingStrategy.CONTRARIAN: {
            AuctionPhase.EARLY: 0.8,
            AuctionPhase.MIDDLE: 0.9,
            AuctionPhase.LATE: 1.3
        }
    }
    
    return multipliers.get(strategy, {}).get(phase, 1.0)


def _calculate_bid_confidence(player: Dict[str, Any], market_conditions: MarketConditions,
                            participant: AuctionParticipant, base_value: float, bid: float) -> float:
    """Calculate confidence level for a bid."""
    confidence = 0.5  # Base confidence
    
    # Higher confidence if we need this position
    if participant.needs_position(player['Ruolo']):
        confidence += 0.2
    
    # Higher confidence if bid is close to value
    value_ratio = bid / base_value if base_value > 0 else 1
    if 0.9 <= value_ratio <= 1.1:
        confidence += 0.2
    elif value_ratio > 1.5 or value_ratio < 0.7:
        confidence -= 0.2
    
    # Adjust for market conditions
    if market_conditions.phase == AuctionPhase.LATE and participant.remaining_budget() > participant.budget * 0.5:
        confidence += 0.1
    
    # Adjust for strategy alignment
    if participant.strategy == BiddingStrategy.VALUE_BASED and 0.95 <= value_ratio <= 1.05:
        confidence += 0.15
    
    return max(0.0, min(1.0, confidence))


def _generate_bid_reasoning(player: Dict[str, Any], market_conditions: MarketConditions,
                          participant: AuctionParticipant, bid: float, base_value: float) -> str:
    """Generate human-readable reasoning for the bid."""
    reasons = []
    
    value_ratio = bid / base_value if base_value > 0 else 1
    
    if value_ratio > 1.2:
        reasons.append(f"Bidding {value_ratio:.1f}x value due to high need/competition")
    elif value_ratio < 0.8:
        reasons.append(f"Conservative bid at {value_ratio:.1f}x value")
    else:
        reasons.append(f"Fair value bid at {value_ratio:.1f}x")
    
    if participant.needs_position(player['Ruolo']):
        reasons.append(f"High priority - need {player['Ruolo']}")
    
    if market_conditions.phase == AuctionPhase.LATE:
        reasons.append("Late auction - adjusted strategy")
    
    if participant.remaining_budget() < participant.budget * 0.3:
        reasons.append("Budget constraints limiting bid")
    
    return "; ".join(reasons)


def simulate_auction_round(players: List[Dict[str, Any]], participants: List[AuctionParticipant], 
                          player_valuations: Dict[str, float]) -> Dict[str, Any]:
    """Simulate a round of auction bidding."""
    
    simulator = AuctionSimulator(participants)
    results = {
        'sales': [],
        'unsold': [],
        'participant_spending': {p.name: 0 for p in participants},
        'market_analysis': []
    }
    
    total_budget = sum(p.budget for p in participants)
    total_spent = 0
    
    for i, player in enumerate(players):
        # Update market conditions
        simulator.update_market_conditions(i, len(players), total_spent, total_budget)
        
        # Get bids from all participants
        bids = []
        for participant in participants:
            if participant.remaining_budget() > 1:  # Must have at least 1 unit to bid
                bid_decision = calculate_optimal_bid(player, simulator.market_conditions, participant, player_valuations)
                if bid_decision.max_bid >= 1:  # Minimum bid of 1
                    bids.append((participant, bid_decision))
        
        if not bids:
            results['unsold'].append(player['Nome'])
            continue
        
        # Simulate bidding war
        winning_participant, winning_bid = _simulate_bidding_war(bids, player)
        
        # Record sale
        sale_price = winning_bid.max_bid
        results['sales'].append({
            'player': player['Nome'],
            'buyer': winning_participant.name,
            'price': sale_price,
            'value': player_valuations.get(player['Nome'], player.get('Quotazione', 0)),
            'strategy': winning_bid.strategy_used.value
        })
        
        # Update participant budget and roster
        winning_participant.budget -= sale_price
        winning_participant.current_roster.append(player['Nome'])
        if player['Ruolo'] in winning_participant.position_needs:
            winning_participant.position_needs[player['Ruolo']] -= 1
        
        results['participant_spending'][winning_participant.name] += sale_price
        total_spent += sale_price
        
        # Record market analysis
        results['market_analysis'].append({
            'player': player['Nome'],
            'market_multiplier': simulator.market_conditions.average_bid_multiplier,
            'competition_level': simulator.market_conditions.competition_level,
            'phase': simulator.market_conditions.phase.value
        })
    
    return results


def _simulate_bidding_war(bids: List[Tuple[AuctionParticipant, BidDecision]], 
                         player: Dict[str, Any]) -> Tuple[AuctionParticipant, BidDecision]:
    """Simulate a bidding war between participants."""
    
    # Sort bids by max bid amount
    sorted_bids = sorted(bids, key=lambda x: x[1].max_bid, reverse=True)
    
    # Add some randomness to simulate real auction dynamics
    top_bid = sorted_bids[0]
    
    # If there's competition, add some variance
    if len(sorted_bids) > 1:
        second_bid = sorted_bids[1][1].max_bid
        # Winner pays slightly more than second highest bid
        final_price = min(top_bid[1].max_bid, second_bid * 1.05 + random.uniform(0, 2))
        
        # Update the winning bid decision
        winning_bid = BidDecision(
            player_name=top_bid[1].player_name,
            max_bid=final_price,
            confidence=top_bid[1].confidence,
            strategy_used=top_bid[1].strategy_used,
            reasoning=top_bid[1].reasoning
        )
        
        return top_bid[0], winning_bid
    else:
        return top_bid[0], top_bid[1]


def generate_bidding_strategy(player_valuations: Dict[str, float], 
                            risk_tolerance: float = 0.5,
                            budget: float = 200,
                            position_needs: Dict[str, int] = None) -> Dict[str, Any]:
    """Generate a comprehensive bidding strategy."""
    
    if position_needs is None:
        position_needs = {'P': 3, 'D': 8, 'C': 8, 'A': 6}  # Default formation
    
    strategy = {
        'budget_allocation': {},
        'target_players': {},
        'fallback_options': {},
        'phase_strategies': {},
        'risk_management': {}
    }
    
    # Budget allocation by position
    total_positions = sum(position_needs.values())
    for position, count in position_needs.items():
        position_players = {name: value for name, value in player_valuations.items() 
                          if any(position in name for name in player_valuations.keys())}
        
        if position_players:
            avg_value = np.mean(list(position_players.values()))
            strategy['budget_allocation'][position] = {
                'total_budget': (count / total_positions) * budget,
                'avg_per_player': avg_value,
                'recommended_max': avg_value * 1.5
            }
    
    # Identify target players (top value players within budget)
    sorted_players = sorted(player_valuations.items(), key=lambda x: x[1], reverse=True)
    
    for position in position_needs.keys():
        position_targets = []
        position_fallbacks = []
        
        for name, value in sorted_players:
            if position in name:  # Simple position matching
                if len(position_targets) < position_needs[position]:
                    if value <= strategy['budget_allocation'].get(position, {}).get('recommended_max', float('inf')):
                        position_targets.append({'name': name, 'value': value, 'priority': 'high'})
                    else:
                        position_fallbacks.append({'name': name, 'value': value, 'priority': 'medium'})
                elif len(position_fallbacks) < position_needs[position] * 2:
                    position_fallbacks.append({'name': name, 'value': value, 'priority': 'low'})
        
        strategy['target_players'][position] = position_targets
        strategy['fallback_options'][position] = position_fallbacks
    
    # Phase-specific strategies
    strategy['phase_strategies'] = {
        'early': {
            'approach': 'selective',
            'max_overpay': 1.1,
            'focus': 'elite players only'
        },
        'middle': {
            'approach': 'balanced',
            'max_overpay': 1.2,
            'focus': 'fill key positions'
        },
        'late': {
            'approach': 'aggressive',
            'max_overpay': 1.5,
            'focus': 'complete roster'
        }
    }
    
    # Risk management
    strategy['risk_management'] = {
        'max_single_player_budget': budget * 0.3,
        'reserve_budget': budget * 0.1,
        'diversification_target': 'no more than 40% budget on one position',
        'contingency_plans': 'maintain 2x fallback options per position'
    }
    
    return strategy


def analyze_auction_outcomes(auction_results: Dict[str, Any], 
                           expected_values: Dict[str, float]) -> Dict[str, Any]:
    """Analyze the outcomes of auction simulations."""
    
    analysis = {
        'overall_metrics': {},
        'participant_analysis': {},
        'market_efficiency': {},
        'strategy_effectiveness': {},
        'recommendations': []
    }
    
    sales = auction_results['sales']
    
    if not sales:
        return analysis
    
    # Overall market metrics
    total_spent = sum(sale['price'] for sale in sales)
    total_value = sum(expected_values.get(sale['player'], sale['price']) for sale in sales)
    
    analysis['overall_metrics'] = {
        'total_players_sold': len(sales),
        'total_money_spent': total_spent,
        'average_sale_price': total_spent / len(sales),
        'market_inflation': total_spent / total_value if total_value > 0 else 1.0,
        'unsold_players': len(auction_results['unsold'])
    }
    
    # Participant analysis
    for participant, spending in auction_results['participant_spending'].items():
        participant_sales = [sale for sale in sales if sale['buyer'] == participant]
        
        if participant_sales:
            participant_value = sum(expected_values.get(sale['player'], sale['price']) for sale in participant_sales)
            
            analysis['participant_analysis'][participant] = {
                'players_acquired': len(participant_sales),
                'total_spent': spending,
                'average_price': spending / len(participant_sales),
                'value_acquired': participant_value,
                'efficiency_ratio': participant_value / spending if spending > 0 else 0,
                'overpay_amount': spending - participant_value
            }
    
    # Market efficiency analysis
    overpays = []
    underpays = []
    
    for sale in sales:
        expected_value = expected_values.get(sale['player'], sale['price'])
        difference = sale['price'] - expected_value
        
        if difference > 0:
            overpays.append(difference)
        else:
            underpays.append(abs(difference))
    
    analysis['market_efficiency'] = {
        'average_overpay': np.mean(overpays) if overpays else 0,
        'average_underpay': np.mean(underpays) if underpays else 0,
        'price_variance': np.std([sale['price'] for sale in sales]),
        'efficiency_score': 1 - (np.mean(overpays) / np.mean([sale['price'] for sale in sales])) if overpays else 1.0
    }
    
    # Strategy effectiveness
    strategy_performance = defaultdict(list)
    
    for sale in sales:
        strategy = sale['strategy']
        expected_value = expected_values.get(sale['player'], sale['price'])
        efficiency = expected_value / sale['price'] if sale['price'] > 0 else 0
        strategy_performance[strategy].append(efficiency)
    
    for strategy, efficiencies in strategy_performance.items():
        analysis['strategy_effectiveness'][strategy] = {
            'average_efficiency': np.mean(efficiencies),
            'consistency': 1 - np.std(efficiencies),  # Lower std = more consistent
            'usage_count': len(efficiencies)
        }
    
    # Generate recommendations
    recommendations = []
    
    if analysis['overall_metrics']['market_inflation'] > 1.2:
        recommendations.append("Market is overheated - consider more conservative bidding")
    
    if analysis['market_efficiency']['efficiency_score'] < 0.8:
        recommendations.append("Market inefficiency detected - opportunity for value bidding")
    
    best_strategy = max(analysis['strategy_effectiveness'].items(), 
                       key=lambda x: x[1]['average_efficiency'], default=(None, None))
    if best_strategy[0]:
        recommendations.append(f"Most effective strategy: {best_strategy[0]}")
    
    analysis['recommendations'] = recommendations
    
    return analysis


def create_auction_visualizations(auction_results: Dict[str, Any], 
                                expected_values: Dict[str, float],
                                output_dir: str = "reports"):
    """Create visualizations for auction analysis."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    sales = auction_results['sales']
    
    if not sales:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Auction Analysis Dashboard', fontsize=16)
    
    # 1. Price vs Value scatter plot
    prices = [sale['price'] for sale in sales]
    values = [expected_values.get(sale['player'], sale['price']) for sale in sales]
    
    axes[0, 0].scatter(values, prices, alpha=0.6)
    axes[0, 0].plot([min(values), max(values)], [min(values), max(values)], 'r--', label='Perfect Value Line')
    axes[0, 0].set_xlabel('Expected Value')
    axes[0, 0].set_ylabel('Sale Price')
    axes[0, 0].set_title('Price vs Expected Value')
    axes[0, 0].legend()
    
    # 2. Spending by participant
    participants = list(auction_results['participant_spending'].keys())
    spending = list(auction_results['participant_spending'].values())
    
    axes[0, 1].bar(participants, spending)
    axes[0, 1].set_xlabel('Participant')
    axes[0, 1].set_ylabel('Total Spending')
    axes[0, 1].set_title('Spending by Participant')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Strategy effectiveness
    strategies = {}
    for sale in sales:
        strategy = sale['strategy']
        if strategy not in strategies:
            strategies[strategy] = []
        expected_value = expected_values.get(sale['player'], sale['price'])
        efficiency = expected_value / sale['price'] if sale['price'] > 0 else 0
        strategies[strategy].append(efficiency)
    
    strategy_names = list(strategies.keys())
    avg_efficiencies = [np.mean(strategies[s]) for s in strategy_names]
    
    axes[1, 0].bar(strategy_names, avg_efficiencies)
    axes[1, 0].set_xlabel('Strategy')
    axes[1, 0].set_ylabel('Average Efficiency')
    axes[1, 0].set_title('Strategy Effectiveness')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Market progression
    if 'market_analysis' in auction_results:
        market_data = auction_results['market_analysis']
        multipliers = [m['market_multiplier'] for m in market_data]
        
        axes[1, 1].plot(range(len(multipliers)), multipliers)
        axes[1, 1].set_xlabel('Auction Progress')
        axes[1, 1].set_ylabel('Market Multiplier')
        axes[1, 1].set_title('Market Inflation Over Time')
        axes[1, 1].axhline(y=1.0, color='r', linestyle='--', label='Fair Value')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/auction_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()


def export_auction_strategy(strategy: Dict[str, Any], output_path: str = "auction_strategy.csv"):
    """Export auction strategy to CSV format."""
    
    # Export budget allocation
    budget_data = []
    for position, allocation in strategy['budget_allocation'].items():
        budget_data.append({
            'Position': position,
            'Total_Budget': allocation['total_budget'],
            'Avg_Per_Player': allocation['avg_per_player'],
            'Recommended_Max': allocation['recommended_max']
        })
    
    budget_df = pd.DataFrame(budget_data)
    budget_df.to_csv(output_path.replace('.csv', '_budget.csv'), index=False)
    
    # Export target players
    target_data = []
    for position, targets in strategy['target_players'].items():
        for target in targets:
            target_data.append({
                'Position': position,
                'Player': target['name'],
                'Value': target['value'],
                'Priority': target['priority'],
                'Type': 'Target'
            })
    
    for position, fallbacks in strategy['fallback_options'].items():
        for fallback in fallbacks:
            target_data.append({
                'Position': position,
                'Player': fallback['name'],
                'Value': fallback['value'],
                'Priority': fallback['priority'],
                'Type': 'Fallback'
            })
    
    target_df = pd.DataFrame(target_data)
    target_df.to_csv(output_path.replace('.csv', '_targets.csv'), index=False)