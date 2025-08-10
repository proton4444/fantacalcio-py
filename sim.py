#!/usr/bin/env python3
"""
Fantacalcio Simulation Suite - CLI Entry Point

Combined analysis tool for fantasy football that generates comprehensive PDF reports.
Includes scarcity analysis, shading simulation, and auction flow analysis.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger
import yaml

# Import simulation modules
from modules.scarcity_sim import ScarcitySimulation
from modules.shading_sim import ShadingSimulation
from modules.auction_flow import AuctionFlowAnalysis
from modules.report_builder import ReportBuilder
from modules.tier_system import classify_players_by_tier
from modules.auction_strategy import build_auction_strategy
from modules.bids_round1 import generate_bids_round1


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity level."""
    logger.remove()  # Remove default handler
    
    if verbose:
        logger.add(sys.stderr, level="DEBUG", 
                  format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    else:
        logger.add(sys.stderr, level="INFO",
                  format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    try:
        logger.info(f"Attempting to load config from: {config_path}")
        logger.info(f"Config file exists: {config_path.exists()}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        logger.info(f"Config type: {type(config)}, Config content preview: {str(config)[:100] if config else 'None'}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        sys.exit(1)


def run_scarcity_analysis(config: dict, output_dir: Path) -> dict:
    """Run scarcity analysis simulation."""
    logger.info("Starting scarcity analysis...")
    
    scarcity_sim = ScarcitySimulation(config)
    
    # Load data
    scarcity_sim.load_data()
    
    # Run analysis
    scarcity_sim.calculate_position_scarcity()
    scarcity_sim.calculate_expected_points()
    scarcity_sim.calculate_risk_factors()
    scarcity_sim.calculate_ev_scores()
    
    # Generate outputs
    plot_path = scarcity_sim.create_visualizations(output_dir)
    csv_path = scarcity_sim.export_results(output_dir)
    
    # Get results
    results = {
        'player_evs': scarcity_sim.player_evs,
        'summary_stats': scarcity_sim.get_summary_stats(),
        'plot_path': plot_path,
        'csv_path': csv_path
    }
    
    logger.success("Scarcity analysis completed")
    return results


def run_shading_analysis(config: dict, output_dir: Path) -> dict:
    """Run shading simulation analysis."""
    logger.info("Starting shading analysis...")
    
    shading_sim = ShadingSimulation(config)
    
    # Load data
    shading_sim.load_data()
    
    # Run analysis
    shading_sim.estimate_ownership()
    shading_sim.calculate_shading_values()
    
    # Generate outputs
    plot_path = shading_sim.create_visualizations(output_dir)
    csv_path = shading_sim.export_results(output_dir)
    
    # Get results
    results = {
        'shading_results': shading_sim.shading_results,
        'summary_stats': shading_sim.get_summary_stats(),
        'plot_path': plot_path,
        'csv_path': csv_path
    }
    
    logger.success("Shading analysis completed")
    return results


def run_tier_analysis(config: dict, output_dir: Path) -> dict:
    """Run tier classification analysis."""
    logger.info("Starting tier analysis...")
    results = classify_players_by_tier(config, output_dir)
    logger.success("Tier analysis completed")
    return results


def run_auction_strategy_analysis(config: dict, output_dir: Path, tier_results: dict | None = None) -> dict:
    """Run auction strategy analysis, optionally using tier data."""
    logger.info("Starting auction strategy analysis...")
    tier_data = tier_results.get('player_tiers') if tier_results else None
    results = build_auction_strategy(config, tier_data=tier_data, output_dir=output_dir)
    logger.success("Auction strategy analysis completed")
    return results


def run_auction_flow_analysis(config: dict, output_dir: Path) -> dict:
    """Run auction flow analysis."""
    logger.info("Starting auction flow analysis...")
    
    auction_sim = AuctionFlowAnalysis(config)
    
    # Load data
    auction_sim.load_data()
    
    # Run analysis
    auction_sim.categorize_players_by_tier()
    auction_sim.analyze_budget_allocation()
    auction_sim.generate_auction_strategies()
    
    # Generate outputs
    flow_diagram_path = auction_sim.create_flow_diagram(output_dir)
    budget_charts_path = auction_sim.create_budget_charts(output_dir)
    csv_path = auction_sim.export_results(output_dir)
    
    # Get results
    results = {
        'summary_stats': auction_sim.get_summary_stats(),
        'visualizations': [flow_diagram_path, budget_charts_path],
        'csv_path': csv_path
    }
    
    logger.success("Auction flow analysis completed")
    return results


def run_bids_round1_analysis(config: dict, output_dir: Path, tier_results: dict = None, 
                           scarcity_results: dict = None, shading_results: dict = None) -> dict:
    """Run bids round 1 analysis with audit trail."""
    logger.info("Starting bids round 1 analysis...")
    
    # Check if module is enabled
    if not config.get('bids_round1', {}).get('enabled', False):
        logger.info("Bids round 1 analysis is disabled in config")
        return {}
    
    # Generate bids with audit trail
    results = generate_bids_round1(
        config=config,
        tier_data=tier_results,
        scarcity_data=scarcity_results,
        shading_data=shading_results,
        output_dir=output_dir
    )
    
    logger.success("Bids round 1 analysis completed")
    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fantacalcio Simulation Suite - Comprehensive Fantasy Football Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config config.yaml --all
  %(prog)s --config config.yaml --scarcity --shading
  %(prog)s --config config.yaml --auction-flow --output reports/auction_analysis
        """
    )
    
    # Required arguments
    parser.add_argument('--config', '-c', type=Path, required=True,
                       help='Path to configuration YAML file')
    
    # Analysis type selection
    analysis_group = parser.add_argument_group('Analysis Types')
    analysis_group.add_argument('--all', action='store_true',
                               help='Run all analysis types')
    analysis_group.add_argument('--scarcity', action='store_true',
                               help='Run scarcity analysis (Expected Value)')
    analysis_group.add_argument('--shading', action='store_true',
                               help='Run shading analysis (Monte Carlo)')
    analysis_group.add_argument('--auction-flow', action='store_true',
                               help='Run auction flow analysis')
    
    # Output options
    parser.add_argument('--output', '-o', type=Path, default='reports',
                       help='Output directory for reports (default: reports)')
    
    # Logging options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate arguments
    if not args.config.exists():
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    if not any([args.all, args.scarcity, args.shading, args.auction_flow]):
        logger.error("At least one analysis type must be specified")
        parser.print_help()
        sys.exit(1)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir.absolute()}")
    
    # Determine which analyses to run
    run_scarcity = args.all or args.scarcity
    run_shading = (args.all or args.shading) and not config.get('shading_disabled', False)
    run_auction = args.all or args.auction_flow
    run_tiers = args.all
    run_auction_strategy = args.all
    run_bids_round1 = args.all
    
    # Check for incompatible configurations
    if config.get('shading_disabled', False) and run_bids_round1:
        logger.error("Cannot run bids_round1 analysis when shading is disabled")
        sys.exit(1)
    # Run analyses
    results = {}
    try:
        if run_scarcity:
            results['scarcity'] = run_scarcity_analysis(config, output_dir)
        
        if run_shading:
            results['shading'] = run_shading_analysis(config, output_dir)
        
        if run_tiers:
            results['tiers'] = run_tier_analysis(config, output_dir)
        
        if run_auction_strategy:
            results['auction_strategy'] = run_auction_strategy_analysis(
                config, output_dir, tier_results=results.get('tiers'))
        
        if run_auction:
            results['auction_flow'] = run_auction_flow_analysis(config, output_dir)
        
        if run_bids_round1:
            results['bids_round1'] = run_bids_round1_analysis(
                config, output_dir, 
                tier_results=results.get('tiers'),
                scarcity_results=results.get('scarcity'),
                shading_results=results.get('shading')
            )
        
        # Generate combined PDF report
        if results:
            logger.info("Generating combined PDF report...")
            report_builder = ReportBuilder(config)
            report_path = report_builder.generate_report(
                scarcity_results=results.get('scarcity'),
                shading_results=results.get('shading'),
                auction_results=results.get('auction_flow'),
                tier_results=results.get('tiers'),
                auction_strategy_results=results.get('auction_strategy'),
                output_path=output_dir / f"fantacalcio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )
            
            logger.success(f"Analysis complete! Combined report saved to: {report_path}")
            
            # Print summary
            print("\n" + "="*60)
            print("FANTACALCIO ANALYSIS SUMMARY")
            print("="*60)
            
            for analysis_type, data in results.items():
                print(f"\n{analysis_type.upper()} ANALYSIS:")
                if 'summary_stats' in data:
                    for key, value in data['summary_stats'].items():
                        print(f"  {key.replace('_', ' ').title()}: {value}")
            
            print(f"\nCombined PDF Report: {report_path}")
            print("="*60)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == '__main__':
    main()
