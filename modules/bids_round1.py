#!/usr/bin/env python3
"""
Bids Round 1 Module

Generates first round bidding recommendations with comprehensive audit trail
for Fantacalcio auction strategy analysis.
"""

import pandas as pd
import numpy as np
import json
import hashlib
import subprocess
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from loguru import logger
from .auction_strategy import AuctionSimulator


@dataclass
class BidRecommendation:
    """Individual bid recommendation."""
    player_name: str
    role: str
    tier: str
    base_value: float
    shading_factor: float
    scarcity_weight: float
    recommended_bid: int
    max_bid: int
    priority: str
    confidence: float


@dataclass
class AuditManifest:
    """Comprehensive audit manifest for bids round 1."""
    modules_used: List[str]
    totals: Dict[str, Any]
    constraints: Dict[str, Any]
    shading: Dict[str, Any]
    tiers: Dict[str, Any]
    scarcity: Dict[str, Any]
    bidding: Dict[str, Any]
    provenance: Dict[str, Any]
    timestamp: str


class BidsRound1Generator:
    """Generates first round bidding recommendations with audit trail."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data = None
        self.tier_data = None
        self.scarcity_data = None
        self.shading_data = None
        self.recommendations = []
        self.audit_manifest = None
        
    def load_data(self, data_path: str = None) -> None:
        """Load player data from Excel file."""
        if data_path is None:
            data_path = self.config.get('excel_path', 'data/Quotazioni_Fantacalcio_Stagione_2025_26.xlsx')
        
        logger.info(f"Loading data from {data_path}")
        self.data = pd.read_excel(data_path, header=1)  # Use header=1 based on Excel structure
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
        logger.success(f"After cleaning: {len(self.data)} players")
        
    def load_analysis_results(self, tier_data: Dict = None, scarcity_data: Dict = None, 
                            shading_data: Dict = None) -> None:
        """Load results from other analysis modules."""
        self.tier_data = tier_data
        self.scarcity_data = scarcity_data
        self.shading_data = shading_data
        logger.info("Analysis results loaded")
        
    def generate_recommendations(self) -> List[BidRecommendation]:
        """Generate bidding recommendations for round 1."""
        logger.info("Generating round 1 bidding recommendations...")
        
        recommendations = []
        reserve_credits = self.config.get('bids_round1', {}).get('reserve_credits', 80)
        total_budget = self.config.get('total_budget', 400)
        available_budget = total_budget - reserve_credits
        
        # Check if we should filter to target players only
        filter_to_targets = self.config.get('bids_round1', {}).get('filter_to_targets', False)
        target_players = set()
        
        if filter_to_targets:
            # Use auction strategy to identify target players
            auction_simulator = AuctionSimulator(self.config)
            auction_simulator.load_data()
            auction_simulator.calculate_budget_allocation()
            
            # Apply target multiplier from config
            target_multiplier = self.config.get('bids_round1', {}).get('target_multiplier', 3.0)
            original_config = auction_simulator.config.copy()
            auction_simulator.config['target_multiplier'] = target_multiplier
            
            player_targets = auction_simulator.identify_player_targets()
            target_players = set(player_targets.keys())
            logger.info(f"Filtering to {len(target_players)} target players based on formation requirements (multiplier: {target_multiplier})")
        
        # Get formation requirements
        formation = self.config.get('formation', {})
        
        for _, player in self.data.iterrows():
            if pd.isna(player.get('FantaMedia', 0)):
                continue
                
            player_name = player['Nome']
            
            # Skip players not in target list if filtering is enabled
            if filter_to_targets and player_name not in target_players:
                continue
            role = player['Ruolo']
            base_value = float(player.get('FantaMedia', 0))
            
            # Get tier information
            tier = self._get_player_tier(player_name)
            
            # Calculate shading factor
            shading_factor = self._get_shading_factor(player_name)
            
            # Calculate scarcity weight
            scarcity_weight = self._get_scarcity_weight(player_name, role)
            
            # Calculate recommended bid
            adjusted_value = base_value * (1 + shading_factor) * (1 + scarcity_weight)
            recommended_bid = int(adjusted_value)
            
            # Apply +1 rule for anti-parità
            if self.config.get('plus_one_rule', True) and np.random.random() < 0.6:
                recommended_bid = (recommended_bid // 10) * 10 + 1
            
            # Set max bid (with buffer)
            max_bid = min(int(recommended_bid * 1.2), available_budget // 2)
            
            # Adjust max_bid to avoid round numbers as per user request
            if max_bid % 5 == 0:
                max_bid += 1

            # Determine priority
            priority = self._determine_priority(tier, role, base_value)
            
            # Calculate confidence
            confidence = self._calculate_confidence(tier, shading_factor, scarcity_weight)
            
            recommendation = BidRecommendation(
                player_name=player_name,
                role=role,
                tier=tier,
                base_value=base_value,
                shading_factor=shading_factor,
                scarcity_weight=scarcity_weight,
                recommended_bid=recommended_bid,
                max_bid=max_bid,
                priority=priority,
                confidence=confidence
            )
            
            recommendations.append(recommendation)
        
        # Apply max_total_budget filtering if enabled
        max_total_budget = self.config.get('bids_round1', {}).get('max_total_budget')
        if max_total_budget and recommendations:
            # Sort by priority and value first
            recommendations.sort(key=lambda x: (x.priority, -x.recommended_bid))
            
            # Filter to stay within budget
            filtered_recommendations = []
            total_budget = 0
            
            for rec in recommendations:
                if total_budget + rec.recommended_bid <= max_total_budget:
                    filtered_recommendations.append(rec)
                    total_budget += rec.recommended_bid
                else:
                    break
            
            recommendations = filtered_recommendations
            logger.info(f"Applied max_total_budget filter: {len(recommendations)} recommendations, total budget: {total_budget}")
        else:
            # Sort by priority and value
            recommendations.sort(key=lambda x: (x.priority, -x.recommended_bid))
        
        self.recommendations = recommendations
        logger.success(f"Generated {len(recommendations)} bidding recommendations")
        
        return recommendations
    
    def _get_player_tier(self, player_name: str) -> str:
        """Get player tier from tier analysis results."""
        if self.tier_data and 'player_tiers' in self.tier_data:
            player_info = self.tier_data['player_tiers'].get(player_name)
            if player_info:
                return player_info.tier.value
        return "Medium"
    
    def _get_shading_factor(self, player_name: str) -> float:
        """Get shading factor from shading analysis results."""
        if self.shading_data and 'player_analysis' in self.shading_data:
            player_data = self.shading_data['player_analysis'].get(player_name, {})
            return player_data.get('shading_value', 0.0)
        return 0.0
    
    def _get_scarcity_weight(self, player_name: str, role: str) -> float:
        """Get scarcity weight from scarcity analysis results."""
        if self.scarcity_data and 'player_analysis' in self.scarcity_data:
            player_data = self.scarcity_data['player_analysis'].get(player_name, {})
            return player_data.get('scarcity_factor', 0.0)
        
        # Default scarcity by role
        role_scarcity = {
            'Portiere': 0.1,
            'Difensore': 0.05,
            'Centrocampista': 0.03,
            'Attaccante': 0.08
        }
        return role_scarcity.get(role, 0.0)
    
    def _determine_priority(self, tier: str, role: str, base_value: float) -> str:
        """Determine bidding priority."""
        if tier == "Elite" and base_value > 12:
            return "High"
        elif tier in ["Elite", "High"] and base_value > 8:
            return "Medium"
        else:
            return "Low"
    
    def _calculate_confidence(self, tier: str, shading_factor: float, scarcity_weight: float) -> float:
        """Calculate confidence score for recommendation."""
        base_confidence = {
            "Elite": 0.9,
            "High": 0.8,
            "Medium": 0.7,
            "Low": 0.6,
            "Bench": 0.5
        }.get(tier, 0.6)
        
        # Adjust for shading and scarcity
        adjustment = abs(shading_factor) * 0.1 + scarcity_weight * 0.1
        return min(1.0, base_confidence + adjustment)
    
    def generate_audit_manifest(self, output_dir: Path) -> AuditManifest:
        """Generate comprehensive audit manifest."""
        logger.info("Generating audit manifest...")
        
        # Calculate totals
        total_bids = len(self.recommendations)
        sum_recommended = sum(r.recommended_bid for r in self.recommendations)
        sum_max_bid = sum(r.max_bid for r in self.recommendations)
        
        # Count by tier
        tier_counts = {}
        for rec in self.recommendations:
            tier_counts[rec.tier] = tier_counts.get(rec.tier, 0) + 1
        
        # Count by role
        role_counts = {}
        gk_count = 0
        for rec in self.recommendations:
            role_counts[rec.role] = role_counts.get(rec.role, 0) + 1
            if rec.role == 'Portiere':
                gk_count += 1
        
        # Shading analysis
        shading_by_tier = {}
        for rec in self.recommendations:
            if rec.tier not in shading_by_tier:
                shading_by_tier[rec.tier] = []
            shading_by_tier[rec.tier].append(rec.shading_factor)
        
        # Calculate best f per tier
        f_by_tier = {}
        for tier, factors in shading_by_tier.items():
            if factors:
                f_by_tier[tier] = max(factors)
        
        # Bidding analysis
        plus_one_count = sum(1 for r in self.recommendations if r.recommended_bid % 10 == 1)
        percent_plus_one = (plus_one_count / total_bids * 100) if total_bids > 0 else 0
        
        # Scarcity analysis
        scarcity_weights = [r.scarcity_weight for r in self.recommendations]
        module_feasibility_weight = np.mean(scarcity_weights) if scarcity_weights else 0.0
        
        # Provenance
        excel_hash = self._calculate_excel_hash()
        git_commit = self._get_git_commit()
        
        self.audit_manifest = AuditManifest(
            modules_used=[
                "scarcity_sim",
                "shading_sim", 
                "tier_system",
                "auction_strategy",
                "bids_round1"
            ],
            totals={
                "total_bids": total_bids,
                "sum_recommended": sum_recommended,
                "sum_max_bid": sum_max_bid
            },
            constraints={
                "gk_count": gk_count,
                "budget_cap": self.config.get('budget_cap', 400),
                "reserve_credits": self.config.get('bids_round1', {}).get('reserve_credits', 80)
            },
            shading={
                "f_by_tier": f_by_tier,
                "total_players_shaded": len([r for r in self.recommendations if r.shading_factor != 0])
            },
            tiers={
                "counts": tier_counts,
                "total_classified": sum(tier_counts.values())
            },
            scarcity={
                "module_feasibility_weight": module_feasibility_weight,
                "avg_scarcity_weight": np.mean(scarcity_weights) if scarcity_weights else 0.0
            },
            bidding={
                "plus_one_rule": self.config.get('plus_one_rule', True),
                "percent_plus_one": percent_plus_one,
                "plus_one_count": plus_one_count
            },
            provenance={
                "excel_sha256": excel_hash,
                "git_commit": git_commit,
                "generation_time": datetime.now().isoformat()
            },
            timestamp=datetime.now().isoformat()
        )
        
        logger.success("Audit manifest generated")
        return self.audit_manifest
    
    def _calculate_excel_hash(self) -> str:
        """Calculate SHA256 hash of Excel file."""
        try:
            data_path = self.config.get('data_file', 'data/Quotazioni_Fantacalcio_Stagione_2025_26.xlsx')
            with open(data_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.warning(f"Could not calculate Excel hash: {e}")
            return "unavailable"
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'], 
                capture_output=True, 
                text=True, 
                cwd=Path.cwd()
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.warning(f"Could not get git commit: {e}")
        return "unavailable"
    
    def export_results(self, output_dir: Path) -> Tuple[Path, Path]:
        """Export recommendations and audit manifest."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export recommendations CSV
        recommendations_df = pd.DataFrame([
            asdict(rec) for rec in self.recommendations
        ])
        csv_path = output_dir / f"bids_round1_recommendations_{timestamp}.csv"
        recommendations_df.to_csv(csv_path, index=False)
        
        # Export audit manifest JSON
        audit_path = output_dir / f"bids_round1_audit.json"
        with open(audit_path, 'w') as f:
            json.dump(asdict(self.audit_manifest), f, indent=2)
        
        logger.success(f"Results exported to {csv_path} and {audit_path}")
        return csv_path, audit_path
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for reporting."""
        if not self.recommendations:
            return {}
        
        return {
            'total_recommendations': len(self.recommendations),
            'avg_recommended_bid': np.mean([r.recommended_bid for r in self.recommendations]),
            'total_budget_required': sum(r.recommended_bid for r in self.recommendations),
            'high_priority_count': len([r for r in self.recommendations if r.priority == 'High']),
            'avg_confidence': np.mean([r.confidence for r in self.recommendations]),
            'plus_one_percentage': len([r for r in self.recommendations if r.recommended_bid % 10 == 1]) / len(self.recommendations) * 100
        }


def generate_bids_round1(config: dict, tier_data: dict = None, scarcity_data: dict = None, 
                        shading_data: dict = None, output_dir: Path = None) -> dict:
    """Main function to generate round 1 bidding recommendations."""
    generator = BidsRound1Generator(config)
    generator.load_data()
    generator.load_analysis_results(tier_data, scarcity_data, shading_data)
    
    # Generate recommendations
    recommendations = generator.generate_recommendations()
    
    # Generate audit manifest
    audit_manifest = generator.generate_audit_manifest(output_dir)
    
    # Export results
    if output_dir:
        csv_path, audit_path = generator.export_results(output_dir)
    else:
        csv_path, audit_path = None, None
    
    return {
        'recommendations': recommendations,
        'audit_manifest': audit_manifest,
        'summary_stats': generator.get_summary_stats(),
        'csv_path': csv_path,
        'audit_path': audit_path
    }