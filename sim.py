#!/usr/bin/env python3
"""
Fantacalcio Simulation Suite - Enhanced CLI Entry Point

Combined analysis tool for fantasy football that generates comprehensive PDF reports.
Includes scarcity analysis, shading simulation, auction flow analysis, and module comparison.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger
import yaml
import copy

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


def apply_cli_overrides(config: dict, args) -> dict:
    """Apply command-line argument overrides to config."""
    config = copy.deepcopy(config)
    
    # Override modules if specified
    if args.modules:
        modules_list = [m.strip() for m in args.modules.split(',')]
        if modules_list:
            config['primary_module'] = modules_list[0]
            logger.info(f"CLI override: primary_module = {modules_list[0]}")
            
            if len(modules_list) > 1:
                config['backup_modules'] = modules_list[1:]
                logger.info(f"CLI override: backup_modules = {modules_list[1:]}")
    
    # Override budget settings if specified
    if args.budget_round1:
        if 'bids_round1' not in config:
            config['bids_round1'] = {}
        config['bids_round1']['budget_round1'] = args.budget_round1
        logger.info(f"CLI override: budget_round1 = {args.budget_round1}")
    
    # Override max goalkeepers if specified
    if args.max_gk:
        config['max_gk'] = args.max_gk
        logger.info(f"CLI override: max_gk = {args.max_gk}")
    
    # Enable bids_round1 if flag is specified
    if args.bids_round1:
        if 'bids_round1' not in config:
            config['bids_round1'] = {}
        config['bids_round1']['enabled'] = True
        logger.info("CLI override: bids_round1 enabled")
    
    return config


def run_module_analysis(config: dict, output_dir: Path) -> dict:
    """Run module comparison analysis across different formations."""
    logger.info("Starting module analysis...")
    
    primary_module = config.get('primary_module', '4-2-3-1')
    backup_modules = config.get('backup_modules', ['4-3-3', '4-4-2'])
    all_modules = [primary_module] + backup_modules
    
    module_results = {}
    
    for module in all_modules:
        logger.info(f"Analyzing module: {module}")
        
        # Create a temporary config for this module
        module_config = copy.deepcopy(config)
        module_config['primary_module'] = module
        module_config['backup_modules'] = []
        
        try:
            # Run scarcity analysis for this module
            scarcity_sim = ScarcitySimulation(module_config)
            scarcity_sim.load_data()
            scarcity_sim.calculate_position_scarcity()
            scarcity_sim.calculate_expected_points()
            scarcity_sim.calculate_risk_factors()
            scarcity_sim.calculate_ev_scores()
            
            module_results[module] = {
                'player_evs': scarcity_sim.player_evs,
                'summary_stats': scarcity_sim.get_summary_stats(),
                'module_requirements': module_config.get('modules', {}).get(module, {})
            }
            
            logger.info(f"Module {module} analysis completed")
            
        except Exception as e:
            logger.error(f"Failed to analyze module {module}: {e}")
            module_results[module] = {'error': str(e)}
    
    # Export module comparison results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = output_dir / f"module_analysis_comparison_{timestamp}.csv"
    
    # Create comparison CSV
    import pandas as pd
    comparison_data = []
    
    for module, results in module_results.items():
        if 'error' not in results:
            summary = results['summary_stats']
            comparison_data.append({
                'module': module,
                'total_players_analyzed': summary.get('total_players', 0),
                'avg_expected_value': summary.get('average_ev', 0),
                'total_expected_value': summary.get('total_ev', 0),
                'high_value_players': summary.get('high_value_count', 0),
                'requirements': str(results['module_requirements'])
            })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        df.to_csv(csv_path, index=False)
        logger.info(f"Module comparison exported to: {csv_path}")
    
    logger.success("Module analysis completed")
    return {
        'module_results': module_results,
        'comparison_csv': csv_path,
        'summary_stats': {
            'modules_analyzed': len(module_results),
            'primary_module': primary_module,
            'backup_modules': backup_modules
        }
    }


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
        description="Fantacalcio Simulation Suite - Enhanced Comprehensive Fantasy Football Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config config.yaml --all
  %(prog)s --config config.yaml --scarcity --shading --tier-system
  %(prog)s --config config.yaml --modules "4-2-3-1,4-4-2" --budget-round1 320 --bids-round1
  %(prog)s --config config.yaml --module-analysis --verbose
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
    analysis_group.add_argument('--tier-system', action='store_true',
                               help='Run tier classification analysis')
    analysis_group.add_argument('--module-analysis', action='store_true',
                               help='Run module comparison analysis')
    analysis_group.add_argument('--bids-round1', action='store_true',
                               help='Run first round bidding recommendations')
    
    # Configuration overrides
    config_group = parser.add_argument_group('Configuration Overrides')
    config_group.add_argument('--modules', type=str,
                             help='Comma-separated list of modules to analyze (e.g., "4-2-3-1,4-4-2")')
    config_group.add_argument('--budget-round1', type=int,
                             help='Budget allocation for round 1 (overrides config)')
    config_group.add_argument('--max-gk', type=int,
                             help='Maximum number of goalkeepers (overrides config)')
    
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
    
    if not any([args.all, args.scarcity, args.shading, args.auction_flow, 
                args.tier_system, args.module_analysis, args.bids_round1]):
        logger.error("At least one analysis type must be specified")
        parser.print_help()
        sys.exit(1)
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply CLI overrides
    config = apply_cli_overrides(config, args)
    
    # Create output directory with date
    today = datetime.now().strftime('%Y-%m-%d')
    output_dir = args.output / today
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir.absolute()}")
    
    # Determine which analyses to run
    run_scarcity = args.all or args.scarcity
    run_shading = (args.all or args.shading) and not config.get('shading_disabled', False)
    run_auction = args.all or args.auction_flow
    run_tiers = args.all or args.tier_system
    run_auction_strategy = args.all
    run_bids_analysis = args.all or args.bids_round1
    run_module_comp = args.module_analysis
    
    # Check for incompatible configurations
    if config.get('shading_disabled', False) and run_bids_analysis:
        logger.error("Cannot run bids_round1 analysis when shading is disabled")
        sys.exit(1)
    
    # Log analysis plan
    logger.info("Analysis plan:")
    logger.info(f"  Scarcity: {run_scarcity}")
    logger.info(f"  Shading: {run_shading}")
    logger.info(f"  Auction Flow: {run_auction}")
    logger.info(f"  Tier System: {run_tiers}")
    logger.info(f"  Auction Strategy: {run_auction_strategy}")
    logger.info(f"  Bids Round 1: {run_bids_analysis}")
    logger.info(f"  Module Analysis: {run_module_comp}")
    
    # Run analyses
    results = {}
    try:
        if run_module_comp:
            results['module_analysis'] = run_module_analysis(config, output_dir)
        
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
        
        if run_bids_analysis:
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
                module_analysis_results=results.get('module_analysis'),
                output_path=output_dir / f"fantacalcio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )
            
            logger.success(f"Analysis complete! Combined report saved to: {report_path}")
            
            # Print summary
            print("\n" + "="*60)
            print("FANTACALCIO ANALYSIS SUMMARY")
            print("="*60)
            
            # Show CLI overrides that were applied
            if args.modules or args.budget_round1 or args.max_gk:
                print("\nCLI OVERRIDES APPLIED:")
                if args.modules:
                    print(f"  Modules: {args.modules}")
                if args.budget_round1:
                    print(f"  Budget Round 1: {args.budget_round1}")
                if args.max_gk:
                    print(f"  Max Goalkeepers: {args.max_gk}")
            
            for analysis_type, data in results.items():
                print(f"\n{analysis_type.upper().replace('_', ' ')} ANALYSIS:")
                if 'summary_stats' in data:
                    for key, value in data['summary_stats'].items():
                        print(f"  {key.replace('_', ' ').title()}: {value}")
            
            print(f"\nOutput Directory: {output_dir}")
            print(f"Combined PDF Report: {report_path}")
            print("="*60)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == '__main__':
    main()
