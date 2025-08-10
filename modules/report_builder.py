#!/usr/bin/env python3
"""
Report Builder Module

Merges outputs from all simulation modules into a unified PDF report.
Provides comprehensive analysis combining scarcity, shading, and auction flow insights.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import io
from PIL import Image


class ReportBuilder:
    """Builds comprehensive PDF reports from simulation results."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.report_title = config.get('report_title', 'Fantacalcio Simulation Suite Report')
        self.author = config.get('author', 'Fantacalcio Analysis System')
        
        # Report styling
        self.page_size = (8.27, 11.69)  # A4 size in inches
        self.title_fontsize = 16
        self.header_fontsize = 14
        self.body_fontsize = 12
        self.small_fontsize = 10
        
        # Color scheme
        self.primary_color = '#2E86AB'
        self.secondary_color = '#A23B72'
        self.accent_color = '#F18F01'
        self.text_color = '#333333'
        
    def generate_report(self, 
                       scarcity_results: Optional[Dict[str, Any]] = None,
                       shading_results: Optional[Dict[str, Any]] = None,
                       auction_results: Optional[Dict[str, Any]] = None,
                       tier_results: Optional[Dict[str, Any]] = None,
                       auction_strategy_results: Optional[Dict[str, Any]] = None,
                       output_path: Path = None) -> Path:
        """Generate comprehensive PDF report from all simulation results."""
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = Path(f"fantacalcio_report_{timestamp}.pdf")
        
        logger.info(f"Generating comprehensive report: {output_path}")
        
        with PdfPages(output_path) as pdf:
            # Page 1: Title page
            self._create_title_page(pdf)
            
            # Page 2: Executive summary
            self._create_executive_summary(pdf, scarcity_results, shading_results, auction_results, tier_results, auction_strategy_results)
            
            # Pages 3+: Scarcity analysis
            if scarcity_results:
                self._add_scarcity_section(pdf, scarcity_results)
            
            # Pages: Shading analysis
            if shading_results:
                self._add_shading_section(pdf, shading_results)
            
            # Pages: Tier analysis
            if tier_results:
                self._add_tier_section(pdf, tier_results)
            
            # Pages: Auction strategy analysis
            if auction_strategy_results:
                self._add_auction_strategy_section(pdf, auction_strategy_results)
            
            # Pages: Auction flow analysis
            if auction_results:
                self._add_auction_section(pdf, auction_results)
            
            # Final page: Methodology and notes
            self._create_methodology_page(pdf)
        
        logger.success(f"Report generated successfully: {output_path}")
        return output_path
    
    def _create_title_page(self, pdf: PdfPages):
        """Create the title page of the report."""
        fig, ax = plt.subplots(figsize=self.page_size)
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.8, self.report_title, 
                ha='center', va='center', fontsize=20, fontweight='bold',
                color=self.primary_color)
        
        # Subtitle
        ax.text(0.5, 0.7, 'Comprehensive Fantasy Football Analysis',
                ha='center', va='center', fontsize=14, 
                color=self.text_color)
        
        # Date and author
        current_date = datetime.now().strftime('%B %d, %Y')
        ax.text(0.5, 0.6, f"Generated on {current_date}",
                ha='center', va='center', fontsize=12,
                color=self.text_color)
        
        ax.text(0.5, 0.55, f"By {self.author}",
                ha='center', va='center', fontsize=12,
                color=self.text_color)
        
        # Configuration summary
        config_text = self._format_config_summary()
        ax.text(0.5, 0.4, config_text,
                ha='center', va='top', fontsize=10,
                color=self.text_color,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.3))
        
        # Footer
        ax.text(0.5, 0.1, 'Fantacalcio Simulation Suite v1.0',
                ha='center', va='center', fontsize=8,
                color='gray')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _format_config_summary(self) -> str:
        """Format configuration summary for title page."""
        summary_lines = ["Analysis Configuration:"]
        
        if 'num_teams' in self.config:
            summary_lines.append(f"• Teams: {self.config['num_teams']}")
        
        if 'total_budget' in self.config:
            summary_lines.append(f"• Budget: €{self.config['total_budget']}")
        
        if 'data_path' in self.config:
            data_file = Path(self.config['data_path']).name
            summary_lines.append(f"• Data: {data_file}")
        
        return "\n".join(summary_lines)
    
    def _create_executive_summary(self, pdf: PdfPages, 
                                scarcity_results: Optional[Dict] = None,
                                shading_results: Optional[Dict] = None,
                                auction_results: Optional[Dict] = None,
                                tier_results: Optional[Dict] = None,
                                auction_strategy_results: Optional[Dict] = None):
        """Create executive summary page."""
        fig, ax = plt.subplots(figsize=self.page_size)
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'Executive Summary', 
                ha='center', va='top', fontsize=self.title_fontsize, 
                fontweight='bold', color=self.primary_color)
        
        y_pos = 0.85
        
        # Scarcity summary
        if scarcity_results and 'summary_stats' in scarcity_results:
            stats = scarcity_results['summary_stats']
            scarcity_text = f"""Scarcity Analysis:
• Analyzed {stats.get('total_players', 'N/A')} players across {stats.get('positions_analyzed', 'N/A')} positions
• Average EV score: {stats.get('avg_ev_score', 0):.2f}
• Top tier players: {stats.get('top_tier_count', 'N/A')}
• Risk assessment completed for all positions"""
            
            ax.text(0.05, y_pos, scarcity_text, va='top', fontsize=self.body_fontsize,
                   color=self.text_color)
            y_pos -= 0.2
        
        # Shading summary
        if shading_results and 'summary_stats' in shading_results:
            stats = shading_results['summary_stats']
            shading_text = f"""Shading Analysis:
• Monte Carlo simulations: {stats.get('total_simulations', 'N/A')}
• Ownership variance analyzed for {stats.get('players_analyzed', 'N/A')} players
• Tournament EV calculated with {stats.get('confidence_level', 95)}% confidence
• Optimal lineup construction strategies identified"""
            
            ax.text(0.05, y_pos, shading_text, va='top', fontsize=self.body_fontsize,
                   color=self.text_color)
            y_pos -= 0.2
        
        # Tier summary
        if tier_results and 'summary_stats' in tier_results:
            stats = tier_results['summary_stats']
            tier_text = f"""Tier Analysis:
• Players Classified: {stats.get('total_players_classified', 'N/A')}
• Elite / High / Medium: {stats.get('elite_players', 0)} / {stats.get('high_tier_players', 0)} / {stats.get('medium_tier_players', 0)}
• Low / Bench: {stats.get('low_tier_players', 0)} / {stats.get('bench_players', 0)}"""
            ax.text(0.05, y_pos, tier_text, va='top', fontsize=self.body_fontsize,
                   color=self.text_color)
            y_pos -= 0.2
        
        # Auction strategy summary
        if auction_strategy_results and 'summary_stats' in auction_strategy_results:
            stats = auction_strategy_results['summary_stats']
            strat_text = f"""Auction Strategy:
• Simulations Run: {stats.get('simulations_run', 'N/A')}
• Avg Team Quality: {stats.get('average_team_quality', 0):.2f}
• Avg Budget Utilization: {stats.get('average_budget_utilization', 0):.1%}
• Targets (High+): {stats.get('high_priority_targets', 0)} / Total Targets: {stats.get('total_target_players', 0)}"""
            ax.text(0.05, y_pos, strat_text, va='top', fontsize=self.body_fontsize,
                   color=self.text_color)
            y_pos -= 0.2
        
        # Auction summary
        if auction_results and 'summary_stats' in auction_results:
            stats = auction_results['summary_stats']
            auction_text = f"""Auction Flow Analysis:
• Generated {stats.get('total_strategies', 'N/A')} strategic recommendations
• Budget allocation across {len(stats.get('position_allocations', {}))} positions
• {stats.get('nomination_rounds_used', 'N/A')} nomination rounds planned
• Average strategy confidence: {stats.get('avg_confidence', 0):.1%}"""
            
            ax.text(0.05, y_pos, auction_text, va='top', fontsize=self.body_fontsize,
                   color=self.text_color)
            y_pos -= 0.2
        
        # Key recommendations
        recommendations_text = """Key Recommendations:
• Focus on high-EV players identified in scarcity analysis
• Consider ownership levels when making final lineup decisions
• Follow nomination timeline for optimal auction strategy
• Monitor budget allocation to avoid overspending in early rounds"""
        
        ax.text(0.05, y_pos, recommendations_text, va='top', fontsize=self.body_fontsize,
               color=self.secondary_color, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _add_scarcity_section(self, pdf: PdfPages, results: Dict[str, Any]):
        """Add scarcity analysis section to the report."""
        # Section title page
        fig, ax = plt.subplots(figsize=self.page_size)
        ax.axis('off')
        
        ax.text(0.5, 0.8, 'Scarcity Analysis', 
                ha='center', va='center', fontsize=self.title_fontsize, 
                fontweight='bold', color=self.primary_color)
        
        ax.text(0.5, 0.6, 'Expected Value and Position Scarcity Assessment',
                ha='center', va='center', fontsize=self.header_fontsize,
                color=self.text_color)
        
        # Add summary if available
        if 'summary_stats' in results:
            stats = results['summary_stats']
            summary_text = f"""Analysis Overview:
• Total Players Analyzed: {stats.get('total_players', 'N/A')}
• Positions Covered: {stats.get('positions_analyzed', 'N/A')}
• Average EV Score: {stats.get('avg_ev_score', 0):.2f}
• Top Tier Players: {stats.get('top_tier_count', 'N/A')}"""
            
            ax.text(0.5, 0.4, summary_text, ha='center', va='center', 
                   fontsize=self.body_fontsize, color=self.text_color,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Add visualizations if available
        if 'visualizations' in results:
            for viz_path in results['visualizations']:
                if Path(viz_path).exists():
                    self._add_image_page(pdf, viz_path, "Scarcity Analysis Visualization")
    
    def _add_shading_section(self, pdf: PdfPages, results: Dict[str, Any]):
        """Add shading analysis section to the report."""
        # Section title page
        fig, ax = plt.subplots(figsize=self.page_size)
        ax.axis('off')
        
        ax.text(0.5, 0.8, 'Shading Analysis', 
                ha='center', va='center', fontsize=self.title_fontsize, 
                fontweight='bold', color=self.secondary_color)
        
        ax.text(0.5, 0.6, 'Monte Carlo Ownership and Tournament Variance',
                ha='center', va='center', fontsize=self.header_fontsize,
                color=self.text_color)
        
        # Add summary if available
        if 'summary_stats' in results:
            stats = results['summary_stats']
            summary_text = f"""Simulation Overview:
• Monte Carlo Runs: {stats.get('total_simulations', 'N/A')}
• Players Analyzed: {stats.get('players_analyzed', 'N/A')}
• Confidence Level: {stats.get('confidence_level', 95)}%
• Ownership Variance: {stats.get('ownership_variance', 'N/A')}"""
            
            ax.text(0.5, 0.4, summary_text, ha='center', va='center', 
                   fontsize=self.body_fontsize, color=self.text_color,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.3))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Add visualizations if available
        if 'visualizations' in results:
            for viz_path in results['visualizations']:
                if Path(viz_path).exists():
                    self._add_image_page(pdf, viz_path, "Shading Analysis Visualization")
    
    def _add_tier_section(self, pdf: PdfPages, results: Dict[str, Any]):
        """Add tier analysis section to the report."""
        # Section title page
        fig, ax = plt.subplots(figsize=self.page_size)
        ax.axis('off')
        
        ax.text(0.5, 0.8, 'Tier Analysis', 
                ha='center', va='center', fontsize=self.title_fontsize, 
                fontweight='bold', color=self.accent_color)
        
        ax.text(0.5, 0.6, 'Player Classification and Tier Distribution',
                ha='center', va='center', fontsize=self.header_fontsize,
                color=self.text_color)
        
        # Add summary if available
        if 'summary_stats' in results:
            stats = results['summary_stats']
            summary_text = f"""Classification Overview:
• Total Players Classified: {stats.get('total_players_classified', 'N/A')}
• Elite Tier: {stats.get('elite_players', 'N/A')}
• High Tier: {stats.get('high_tier_players', 'N/A')}
• Medium Tier: {stats.get('medium_tier_players', 'N/A')}
• Low/Bench Tier: {stats.get('low_tier_players', 0) + stats.get('bench_players', 0)}"""
            
            ax.text(0.5, 0.4, summary_text, ha='center', va='center', 
                   fontsize=self.body_fontsize, color=self.text_color,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.3))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Add visualizations if available
        if 'visualizations' in results:
            for viz_path in results['visualizations']:
                if Path(viz_path).exists():
                    self._add_image_page(pdf, viz_path, "Tier Analysis Visualization")
    
    def _add_auction_strategy_section(self, pdf: PdfPages, results: Dict[str, Any]):
        """Add auction strategy analysis section to the report."""
        # Section title page
        fig, ax = plt.subplots(figsize=self.page_size)
        ax.axis('off')
        
        ax.text(0.5, 0.8, 'Auction Strategy Analysis', 
                ha='center', va='center', fontsize=self.title_fontsize, 
                fontweight='bold', color=self.secondary_color)
        
        ax.text(0.5, 0.6, 'Strategic Bidding and Player Targeting',
                ha='center', va='center', fontsize=self.header_fontsize,
                color=self.text_color)
        
        # Add summary if available
        if 'summary_stats' in results:
            stats = results['summary_stats']
            summary_text = f"""Strategy Overview:
• Simulations Run: {stats.get('simulations_run', 'N/A')}
• Player Targets: {stats.get('player_targets', 'N/A')}
• Budget Allocation: {stats.get('budget_allocation', 'N/A')}
• Strategy Confidence: {stats.get('avg_confidence', 0):.1%}"""
            
            ax.text(0.5, 0.4, summary_text, ha='center', va='center', 
                   fontsize=self.body_fontsize, color=self.text_color,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcyan', alpha=0.3))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Add visualizations if available
        if 'visualizations' in results:
            for viz_path in results['visualizations']:
                if Path(viz_path).exists():
                    self._add_image_page(pdf, viz_path, "Auction Strategy Visualization")
    
    def _add_auction_section(self, pdf: PdfPages, results: Dict[str, Any]):
        """Add auction flow analysis section to the report."""
        # Section title page
        fig, ax = plt.subplots(figsize=self.page_size)
        ax.axis('off')
        
        ax.text(0.5, 0.8, 'Auction Flow Analysis', 
                ha='center', va='center', fontsize=self.title_fontsize, 
                fontweight='bold', color=self.accent_color)
        
        ax.text(0.5, 0.6, 'Strategic Bidding and Budget Allocation',
                ha='center', va='center', fontsize=self.header_fontsize,
                color=self.text_color)
        
        # Add summary if available
        if 'summary_stats' in results:
            stats = results['summary_stats']
            summary_text = f"""Strategy Overview:
• Total Strategies: {stats.get('total_strategies', 'N/A')}
• Budget Utilization: {stats.get('budget_utilization', 0):.1f}%
• Nomination Rounds: {stats.get('nomination_rounds_used', 'N/A')}
• Avg Confidence: {stats.get('avg_confidence', 0):.1%}"""
            
            ax.text(0.5, 0.4, summary_text, ha='center', va='center', 
                   fontsize=self.body_fontsize, color=self.text_color,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.3))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Add visualizations if available
        if 'visualizations' in results:
            for viz_path in results['visualizations']:
                if Path(viz_path).exists():
                    self._add_image_page(pdf, viz_path, "Auction Flow Visualization")
    
    def _add_image_page(self, pdf: PdfPages, image_path: Path, title: str = ""):
        """Add a page with an image to the PDF."""
        try:
            fig, ax = plt.subplots(figsize=self.page_size)
            ax.axis('off')
            
            if title:
                ax.text(0.5, 0.95, title, ha='center', va='top', 
                       fontsize=self.header_fontsize, fontweight='bold')
            
            # Load and display image
            img = plt.imread(image_path)
            ax.imshow(img)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            logger.warning(f"Could not add image {image_path} to report: {e}")
    
    def _create_methodology_page(self, pdf: PdfPages):
        """Create methodology and notes page."""
        fig, ax = plt.subplots(figsize=self.page_size)
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'Methodology & Notes', 
                ha='center', va='top', fontsize=self.title_fontsize, 
                fontweight='bold', color=self.primary_color)
        
        methodology_text = """Analysis Methodology:

Scarcity Analysis:
• Expected Value (EV) calculation based on historical performance
• Position scarcity assessment using supply/demand ratios
• Risk factor analysis incorporating injury history and consistency
• Tier-based player categorization for strategic planning

Shading Analysis:
• Monte Carlo simulation with 10,000+ iterations
• Ownership projection based on player popularity and price
• Tournament variance calculation for GPP optimization
• Lineup construction with correlation considerations

Auction Flow Analysis:
• Budget allocation optimization across positions and tiers
• Nomination strategy based on market timing
• Bidding recommendations with confidence intervals
• Strategic categorization (aggressive, patient, value)

Data Sources:
• Player valuations from official Fantacalcio sources
• Historical performance data and projections
• Market trends and ownership patterns

Disclaimer:
• All projections are estimates based on available data
• Fantasy football involves inherent uncertainty
• Use this analysis as one factor in decision-making
• Past performance does not guarantee future results"""
        
        ax.text(0.05, 0.85, methodology_text, va='top', fontsize=self.small_fontsize,
               color=self.text_color)
        
        # Footer
        ax.text(0.5, 0.05, f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ha='center', va='bottom', fontsize=8, color='gray')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
