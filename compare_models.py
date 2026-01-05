#!/usr/bin/env python
"""
Model comparison visualization script.
Run this after training to visualize model performance comparison.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import sys

def load_comparison_results():
    """Load the most recent model comparison results."""
    json_files = glob('model_comparison_*.json')
    csv_files = glob('model_comparison_*.csv')
    
    if not json_files and not csv_files:
        print("Error: No model comparison files found.")
        print("Please run train.py first to generate comparison results.")
        sys.exit(1)
    
    # Use the most recent file
    if json_files:
        latest_file = sorted(json_files)[-1]
        with open(latest_file, 'r') as f:
            results = json.load(f)
    else:
        latest_file = sorted(csv_files)[-1]
        results = pd.read_csv(latest_file).to_dict('records')
    
    return results, latest_file

def visualize_comparison(results):
    """Create visualization of model comparison."""
    df = pd.DataFrame(results)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Comparison: Regression vs Tree-based Models', fontsize=16, fontweight='bold')
    
    # 1. AUC Comparison
    ax1 = axes[0, 0]
    colors = ['#3498db' if mt == 'Regression' else '#2ecc71' for mt in df['model_type']]
    bars = ax1.barh(df['model_name'], df['auc'], color=colors)
    ax1.set_xlabel('AUC Score', fontsize=12)
    ax1.set_title('AUC Score Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlim([0, 1])
    ax1.grid(axis='x', alpha=0.3)
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df['auc'])):
        ax1.text(val + 0.01, i, f'{val:.4f}', va='center', fontsize=10)
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Regression'),
        Patch(facecolor='#2ecc71', label='Tree-based')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    # 2. F1 Score Comparison
    ax2 = axes[0, 1]
    bars2 = ax2.barh(df['model_name'], df['f1'], color=colors)
    ax2.set_xlabel('F1 Score', fontsize=12)
    ax2.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 1])
    ax2.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars2, df['f1'])):
        ax2.text(val + 0.01, i, f'{val:.4f}', va='center', fontsize=10)
    
    # 3. Precision-Recall Comparison
    ax3 = axes[1, 0]
    x = range(len(df))
    width = 0.35
    ax3.bar([i - width/2 for i in x], df['precision'], width, label='Precision', color='#e74c3c', alpha=0.8)
    ax3.bar([i + width/2 for i in x], df['recall'], width, label='Recall', color='#f39c12', alpha=0.8)
    ax3.set_xlabel('Models', fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Precision vs Recall', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(df['model_name'], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # 4. Comprehensive Metrics Radar (simplified as bar chart)
    ax4 = axes[1, 1]
    metrics = ['auc', 'f1', 'precision', 'recall']
    x_pos = range(len(df))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width
        ax4.bar([p + offset for p in x_pos], df[metric], width, label=metric.upper(), alpha=0.8)
    
    ax4.set_xlabel('Models', fontsize=12)
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(df['model_name'], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'model_comparison_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    
    # Also create a summary table
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    return output_file

def main():
    """Main function."""
    print("Loading model comparison results...")
    results, filename = load_comparison_results()
    
    print(f"Loaded results from: {filename}")
    print(f"Number of models compared: {len(results)}")
    
    print("\nCreating visualizations...")
    output_file = visualize_comparison(results)
    
    print(f"\nComparison visualization complete!")
    print(f"Open {output_file} to view the results.")

if __name__ == '__main__':
    main()

