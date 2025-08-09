# Module Analysis
# This module provides analysis functionality for fantacalcio modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import time
from datetime import datetime


class ModulePerformanceAnalyzer:
    """Analyzes performance metrics for fantacalcio modules."""
    
    def __init__(self):
        self.metrics = {}
        self.execution_times = {}
        self.error_counts = {}
        
    def track_execution_time(self, module_name: str, execution_time: float):
        """Track execution time for a module."""
        if module_name not in self.execution_times:
            self.execution_times[module_name] = []
        self.execution_times[module_name].append(execution_time)
        
    def track_error(self, module_name: str, error_type: str):
        """Track errors for a module."""
        if module_name not in self.error_counts:
            self.error_counts[module_name] = {}
        if error_type not in self.error_counts[module_name]:
            self.error_counts[module_name][error_type] = 0
        self.error_counts[module_name][error_type] += 1
        
    def add_metric(self, module_name: str, metric_name: str, value: float):
        """Add a custom metric for a module."""
        if module_name not in self.metrics:
            self.metrics[module_name] = {}
        if metric_name not in self.metrics[module_name]:
            self.metrics[module_name][metric_name] = []
        self.metrics[module_name][metric_name].append(value)


def analyze_module_performance(analyzer: ModulePerformanceAnalyzer) -> Dict[str, Any]:
    """Analyze the performance of different modules."""
    analysis_results = {
        'execution_stats': {},
        'error_analysis': {},
        'custom_metrics': {},
        'recommendations': []
    }
    
    # Analyze execution times
    for module_name, times in analyzer.execution_times.items():
        if times:
            analysis_results['execution_stats'][module_name] = {
                'avg_time': np.mean(times),
                'median_time': np.median(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'total_executions': len(times)
            }
            
            # Performance recommendations
            avg_time = np.mean(times)
            if avg_time > 5.0:  # seconds
                analysis_results['recommendations'].append(
                    f"Module '{module_name}' has high average execution time ({avg_time:.2f}s). Consider optimization."
                )
    
    # Analyze errors
    for module_name, errors in analyzer.error_counts.items():
        total_errors = sum(errors.values())
        analysis_results['error_analysis'][module_name] = {
            'total_errors': total_errors,
            'error_breakdown': errors,
            'error_rate': total_errors / analyzer.execution_times.get(module_name, [1]).__len__()
        }
        
        if total_errors > 0:
            analysis_results['recommendations'].append(
                f"Module '{module_name}' has {total_errors} errors. Review error handling."
            )
    
    # Analyze custom metrics
    for module_name, metrics in analyzer.metrics.items():
        analysis_results['custom_metrics'][module_name] = {}
        for metric_name, values in metrics.items():
            if values:
                analysis_results['custom_metrics'][module_name][metric_name] = {
                    'avg': np.mean(values),
                    'std': np.std(values),
                    'trend': 'improving' if len(values) > 1 and values[-1] > values[0] else 'stable'
                }
    
    return analysis_results


def generate_module_report(analysis_results: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """Generate a comprehensive report of module analysis."""
    report_lines = []
    report_lines.append("# Fantacalcio Module Performance Report")
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("\n" + "="*50 + "\n")
    
    # Execution Statistics
    report_lines.append("## Execution Statistics")
    if analysis_results['execution_stats']:
        for module_name, stats in analysis_results['execution_stats'].items():
            report_lines.append(f"\n### {module_name}")
            report_lines.append(f"- Average execution time: {stats['avg_time']:.3f}s")
            report_lines.append(f"- Median execution time: {stats['median_time']:.3f}s")
            report_lines.append(f"- Standard deviation: {stats['std_time']:.3f}s")
            report_lines.append(f"- Min/Max time: {stats['min_time']:.3f}s / {stats['max_time']:.3f}s")
            report_lines.append(f"- Total executions: {stats['total_executions']}")
    else:
        report_lines.append("No execution statistics available.")
    
    # Error Analysis
    report_lines.append("\n## Error Analysis")
    if analysis_results['error_analysis']:
        for module_name, error_info in analysis_results['error_analysis'].items():
            report_lines.append(f"\n### {module_name}")
            report_lines.append(f"- Total errors: {error_info['total_errors']}")
            report_lines.append(f"- Error rate: {error_info['error_rate']:.2%}")
            if error_info['error_breakdown']:
                report_lines.append("- Error breakdown:")
                for error_type, count in error_info['error_breakdown'].items():
                    report_lines.append(f"  - {error_type}: {count}")
    else:
        report_lines.append("No errors recorded.")
    
    # Custom Metrics
    report_lines.append("\n## Custom Metrics")
    if analysis_results['custom_metrics']:
        for module_name, metrics in analysis_results['custom_metrics'].items():
            report_lines.append(f"\n### {module_name}")
            for metric_name, metric_data in metrics.items():
                report_lines.append(f"- {metric_name}:")
                report_lines.append(f"  - Average: {metric_data['avg']:.3f}")
                report_lines.append(f"  - Standard deviation: {metric_data['std']:.3f}")
                report_lines.append(f"  - Trend: {metric_data['trend']}")
    else:
        report_lines.append("No custom metrics available.")
    
    # Recommendations
    report_lines.append("\n## Recommendations")
    if analysis_results['recommendations']:
        for i, recommendation in enumerate(analysis_results['recommendations'], 1):
            report_lines.append(f"{i}. {recommendation}")
    else:
        report_lines.append("No specific recommendations at this time.")
    
    report_content = "\n".join(report_lines)
    
    # Save to file if path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
    
    return report_content


def create_performance_visualizations(analysis_results: Dict[str, Any], output_dir: str = "reports"):
    """Create visualizations for module performance analysis."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Execution time comparison
    if analysis_results['execution_stats']:
        modules = list(analysis_results['execution_stats'].keys())
        avg_times = [analysis_results['execution_stats'][mod]['avg_time'] for mod in modules]
        
        plt.figure(figsize=(10, 6))
        plt.bar(modules, avg_times)
        plt.title('Average Execution Time by Module')
        plt.xlabel('Module')
        plt.ylabel('Average Time (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/module_execution_times.png")
        plt.close()
    
    # Error rate comparison
    if analysis_results['error_analysis']:
        modules = list(analysis_results['error_analysis'].keys())
        error_rates = [analysis_results['error_analysis'][mod]['error_rate'] for mod in modules]
        
        plt.figure(figsize=(10, 6))
        plt.bar(modules, error_rates)
        plt.title('Error Rate by Module')
        plt.xlabel('Module')
        plt.ylabel('Error Rate')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/module_error_rates.png")
        plt.close()


# Context manager for performance tracking
class PerformanceTracker:
    """Context manager for tracking module performance."""
    
    def __init__(self, analyzer: ModulePerformanceAnalyzer, module_name: str):
        self.analyzer = analyzer
        self.module_name = module_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = time.time() - self.start_time
        self.analyzer.track_execution_time(self.module_name, execution_time)
        
        if exc_type is not None:
            self.analyzer.track_error(self.module_name, exc_type.__name__)
        
        return False  # Don't suppress exceptions