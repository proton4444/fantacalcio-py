#!/usr/bin/env python3
"""
Mantra Auction List Generator
Quick CLI tool for generating optimized auction lists
"""

import argparse
import sys
import os
from datetime import datetime
import pandas as pd
from loguru import logger

# Import our strategy engine
from mantra_strategy import MantraStrategyEngine

def create_quick_auction_list(budget=500, teams=8, tier_focus=['S', 'A'], max_players=25, role_filter=None):
    """Generate a quick auction list with default settings"""
    
    logger.info("🚀 Generating Mantra auction list...")
    
    # Check if data exists
    if not os.path.exists('giocatori.csv'):
        logger.error("❌ No player data found. Run data collection first:")
        logger.error("   python mantra_strategy.py --collect-data")
        return None
    
    # Initialize engine
    engine = MantraStrategyEngine()
    
    # Load data
    if not engine.load_data():
        return None
    
    # Update config with provided parameters
    engine.config['auction']['total_budget'] = budget
    engine.config['auction']['teams_count'] = teams
    
    # Run analysis
    engine.create_tier_system()
    engine.calculate_polyvalence_bonus()
    engine.calculate_enhanced_value()
    
    # Generate target list
    target_list = engine.create_auction_target_list(
        tier_preference=tier_focus, 
        max_players=max_players
    )
    
    if target_list is None:
        logger.error("❌ Failed to generate target list")
        return None
    
    # Filter by role if specified
    if role_filter:
        role_mapping = {'P': 'Portieri', 'D': 'Difensori', 'C': 'Centrocampisti', 'T': 'Trequartisti', 'A': 'Attaccanti'}
        role_name = role_mapping.get(role_filter)
        if role_name:
            target_list = target_list[target_list['Ruolo'] == role_name]
            logger.info(f"🎯 Filtered for {role_name} only")
    
    # Create output directory
    output_dir = f"reports/{datetime.now().strftime('%Y-%m-%d')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save auction list
    auction_file = os.path.join(output_dir, "quick_auction_list.csv")
    target_list.to_csv(auction_file, index=False)
    
    # Generate budget allocation
    budget_plan = engine.generate_budget_allocation()
    budget_file = os.path.join(output_dir, "budget_plan.csv")
    budget_df = pd.DataFrame.from_dict(budget_plan, orient='index')
    budget_df.to_csv(budget_file)
    
    # Generate module scarcity analysis
    scarcity_analysis = engine.analyze_module_scarcity()
    if scarcity_analysis:
        scarcity_file = os.path.join(output_dir, "scarcity_analysis.csv")
        scarcity_df = pd.DataFrame.from_dict(scarcity_analysis, orient='index')
        scarcity_df.to_csv(scarcity_file)
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 MANTRA AUCTION LIST GENERATED")
    print("="*60)
    print(f"📊 Total Players: {len(target_list)}")
    print(f"💰 Budget: €{budget}")
    print(f"👥 Teams: {teams}")
    print(f"🏆 Tier Focus: {', '.join(tier_focus)}")
    if role_filter:
        print(f"⚽ Role Filter: {role_filter}")
    print(f"📁 Files saved to: {output_dir}")
    
    # Show top 10 targets
    print(f"\n🔥 TOP 10 TARGETS:")
    print("-" * 60)
    for i, (_, player) in enumerate(target_list.head(10).iterrows(), 1):
        bidding = player['Bidding_Strategy']
        poly_bonus = f"+{player['Polyvalence_Bonus']*100:.0f}%" if player['Polyvalence_Bonus'] > 0 else ""
        print(f"{i:2d}. {player['Nome']:<20} ({player['Ruolo']:<3}) "
              f"Tier {player['Tier']} - Bid: €{bidding['suggested_bid']:.0f} {poly_bonus}")
    
    # Show budget allocation
    print(f"\n💰 BUDGET ALLOCATION:")
    print("-" * 30)
    for role, data in budget_plan.items():
        print(f"{role}: €{data['budget']:.0f} ({data['percentage']:.1f}%)")
    
    # Show scarcity warnings
    if scarcity_analysis:
        print(f"\n⚠️  SCARCITY WARNINGS:")
        print("-" * 30)
        for role, data in scarcity_analysis.items():
            if data['scarcity_ratio'] > 0.5:  # High scarcity
                print(f"{role}: {data['scarcity_ratio']:.2f} (HIGH SCARCITY)")
    
    print(f"\n📋 Full list available in: {auction_file}")
    print("="*60)
    
    return output_dir

def show_player_analysis(player_name):
    """Show detailed analysis for a specific player"""
    if not os.path.exists('giocatori.csv'):
        logger.error("❌ No player data found.")
        return
    
    engine = MantraStrategyEngine()
    engine.load_data()
    engine.create_tier_system()
    engine.calculate_polyvalence_bonus()
    engine.calculate_enhanced_value()
    
    # Find player
    player_data = engine.df[engine.df['Nome'].str.contains(player_name, case=False, na=False)]
    
    if len(player_data) == 0:
        print(f"❌ Player '{player_name}' not found")
        return
    
    if len(player_data) > 1:
        print(f"🔍 Multiple players found:")
        for i, (_, p) in enumerate(player_data.iterrows(), 1):
            print(f"{i}. {p['Nome']} ({p['Ruolo']}) - {p['Squadra']}")
        return
    
    player = player_data.iloc[0]
    bidding = engine.suggest_bidding_strategy(player['Nome'], player['Enhanced_Value'])
    
    print("\n" + "="*50)
    print(f"📊 PLAYER ANALYSIS: {player['Nome']}")
    print("="*50)
    print(f"🏟️  Team: {player['Squadra']}")
    print(f"⚽ Role: {player['Ruolo']}")
    print(f"🏆 Tier: {player['Tier']}")
    print(f"💰 Quotazione: €{player['Quotazione']}")
    print(f"📈 Convenience Score: {player['Convenienza']:.2f}")
    print(f"🔄 Polyvalence Bonus: +{player['Polyvalence_Bonus']*100:.1f}%")
    print(f"⭐ Enhanced Value: {player['Enhanced_Value']:.2f}")
    print(f"💡 Suggested Bid: €{bidding['suggested_bid']:.0f}")
    print(f"🚨 Max Bid: €{bidding['max_bid']:.0f}")
    print(f"📋 Strategy: {bidding['strategy'].upper()}")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(
        description='Generate optimized Mantra auction list',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--budget', '-b', 
        type=int, 
        default=500,
        help='Total auction budget'
    )
    
    parser.add_argument(
        '--teams', '-t',
        type=int,
        default=8,
        help='Number of teams in the auction'
    )
    
    parser.add_argument(
        '--tiers',
        type=str,
        default='S,A',
        help='Comma-separated tier focus (S,A,B,C)'
    )
    
    parser.add_argument(
        '--max-players', '-m',
        type=int,
        default=25,
        help='Maximum players in target list'
    )
    
    parser.add_argument(
        '--role',
        type=str,
        choices=['P', 'D', 'C', 'T', 'A'],
        help='Focus on specific role only'
    )
    
    parser.add_argument(
        '--collect-data',
        action='store_true',
        help='Collect fresh data before analysis'
    )
    
    parser.add_argument(
        '--player',
        type=str,
        help='Analyze specific player (partial name matching)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick list generation with defaults'
    )
    
    args = parser.parse_args()
    
    # Show player analysis if requested
    if args.player:
        show_player_analysis(args.player)
        return 0
    
    # Parse tier focus
    tier_focus = [t.strip().upper() for t in args.tiers.split(',')]
    valid_tiers = ['S', 'A', 'B', 'C']
    tier_focus = [t for t in tier_focus if t in valid_tiers]
    
    if not tier_focus:
        logger.error("❌ Invalid tier focus. Use S, A, B, or C")
        return 1
    
    # Collect data if requested
    if args.collect_data:
        logger.info("🔄 Collecting fresh data...")
        os.system("python mantra_strategy.py --collect-data")
    
    # Quick mode with defaults
    if args.quick:
        result = create_quick_auction_list()
    else:
        # Generate auction list with custom parameters
        result = create_quick_auction_list(
            budget=args.budget,
            teams=args.teams,
            tier_focus=tier_focus,
            max_players=args.max_players,
            role_filter=args.role
        )
    
    if result:
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
