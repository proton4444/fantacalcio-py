#!/usr/bin/env python3
"""
Generate final comprehensive report with all successful analyses.
"""

import yaml
from pathlib import Path
from modules.report_builder import ReportBuilder
from modules.scarcity_sim import ScarcitySimulation
from modules.shading_sim import ShadingSimulation
from modules.tier_system import classify_players_by_tier
from modules.auction_flow import AuctionFlowAnalysis
from modules.report_builder import ReportBuilder
from loguru import logger
import sys

def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    logger.info("Generating comprehensive Fantacalcio report...")
    
    try:
        # Run scarcity analysis
        logger.info("Running scarcity analysis...")
        scarcity_sim = ScarcitySimulation(config)
        
        # Load data and run analysis steps
        scarcity_sim.load_data()
        scarcity_sim.calculate_position_scarcity()
        scarcity_sim.calculate_expected_points()
        scarcity_sim.calculate_risk_factors()
        scarcity_sim.calculate_ev_scores()
        
        # Generate outputs
        plot_path = scarcity_sim.create_visualizations(output_dir)
        csv_path = scarcity_sim.export_results(output_dir)
        
        # Get results
        scarcity_results = {
            'player_evs': scarcity_sim.player_evs,
            'summary_stats': scarcity_sim.get_summary_stats(),
            'plot_path': plot_path,
            'csv_path': csv_path
        }
        
        # Run shading analysis
        logger.info("Running shading analysis...")
        shading_sim = ShadingSimulation(config)
        
        # Load data and run analysis steps
        shading_sim.load_data()
        shading_sim.estimate_ownership()
        shading_sim.calculate_shading_values()
        
        # Generate outputs
        plot_path = shading_sim.create_visualizations(output_dir)
        csv_path = shading_sim.export_results(output_dir)
        
        # Get results
        shading_results = {
            'shading_results': shading_sim.shading_results,
            'summary_stats': shading_sim.get_summary_stats(),
            'plot_path': plot_path,
            'csv_path': csv_path
        }
        
        # Run tier analysis
        logger.info("Running tier analysis...")
        tier_results = classify_players_by_tier(config, output_dir)
        
        # Run auction flow analysis
        logger.info("Running auction flow analysis...")
        auction_flow = AuctionFlowAnalysis(config)
        
        # Load data and run analysis steps
        auction_flow.load_data()
        auction_flow.categorize_players_by_tier()
        auction_flow.analyze_budget_allocation()
        auction_flow.generate_auction_strategies()
        
        # Generate outputs
        flow_diagram_path = auction_flow.create_flow_diagram(output_dir)
        budget_charts_path = auction_flow.create_budget_charts(output_dir)
        csv_path = auction_flow.export_results(output_dir)
        
        # Get results
        auction_flow_results = {
            'summary_stats': auction_flow.get_summary_stats(),
            'visualizations': [flow_diagram_path, budget_charts_path],
            'csv_path': csv_path
        }
        
        # Generate comprehensive PDF report
        logger.info("Generating comprehensive PDF report...")
        report_builder = ReportBuilder(config)
        
        from datetime import datetime
        
        report_path = report_builder.generate_report(
            scarcity_results=scarcity_results,
            shading_results=shading_results,
            tier_results=tier_results,
            auction_results=auction_flow_results,
            output_path=output_dir / f"fantacalcio_comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        
        logger.success(f"Comprehensive report generated: {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("FANTACALCIO COMPREHENSIVE ANALYSIS SUMMARY")
        print("="*60)
        print(f"\nSCARCITY ANALYSIS:")
        print(f"  Players Analyzed: {scarcity_results['summary_stats']['total_players']}")
        print(f"  Positions: {scarcity_results['summary_stats']['positions_analyzed']}")
        print(f"  Avg EV Score: {scarcity_results['summary_stats']['avg_ev_score']:.2f}")
        
        print(f"\nSHADING ANALYSIS:")
        print(f"  Players Analyzed: {shading_results['summary_stats']['total_players']}")
        print(f"  Avg Shading Value: {shading_results['summary_stats']['avg_shading_value']:.3f}")
        print(f"  Leverage Opportunities: {shading_results['summary_stats']['leverage_opportunities']}")
        
        print(f"\nTIER ANALYSIS:")
        print(f"  Players Classified: {tier_results['summary_stats']['total_players']}")
        print(f"  Elite Tier: {tier_results['summary_stats']['tier_distribution']['Elite']} players")
        print(f"  High Tier: {tier_results['summary_stats']['tier_distribution']['High']} players")
        print(f"  Medium Tier: {tier_results['summary_stats']['tier_distribution']['Medium']} players")
        print(f"  Low Tier: {tier_results['summary_stats']['tier_distribution']['Low']} players")
        
        print(f"\nAUCTION FLOW ANALYSIS:")
        print(f"  Total Strategies: {auction_flow_results['summary_stats']['total_strategies']}")
        print(f"  Budget Utilization: {auction_flow_results['summary_stats']['budget_utilization']:.1f}%")
        print(f"  Avg Confidence: {auction_flow_results['summary_stats']['avg_confidence']:.2f}")
        
        print(f"\nFinal Report: {report_path}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()