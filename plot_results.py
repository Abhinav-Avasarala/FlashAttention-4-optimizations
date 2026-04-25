#!/usr/bin/env python3
"""
Generate accuracy vs speedup visualization from benchmark results.
Run after exponential_kernel benchmark.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def read_benchmark_csv(filename):
    """Read benchmark results CSV"""
    data = {
        'method': [],
        'time_ms': [],
        'rmse_error': [],
        'speedup': []
    }
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['method'].append(row['method'])
            data['time_ms'].append(float(row['time_ms']))
            data['rmse_error'].append(float(row['rmse_error']))
            data['speedup'].append(float(row['speedup']))
    
    return data

def create_accuracy_plot(data):
    """Create speedup vs accuracy loss plot"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ========== Plot 1: Speedup vs RMSE Error ==========
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    for i, method in enumerate(data['method']):
        ax1.scatter(
            data['rmse_error'][i],
            data['speedup'][i],
            s=300,
            c=colors[i],
            marker=markers[i],
            alpha=0.7,
            edgecolors='black',
            linewidth=2,
            label=method
        )
    
    ax1.set_xlabel('RMSE Error (log scale)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Speedup (x)', fontsize=12, fontweight='bold')
    ax1.set_title('Speedup vs Accuracy Tradeoff', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10, loc='best')
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline (1x)')
    
    # Add annotations
    for i, method in enumerate(data['method']):
        ax1.annotate(
            f"{data['speedup'][i]:.2f}x",
            (data['rmse_error'][i], data['speedup'][i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9
        )
    
    # ========== Plot 2: Execution Time Comparison ==========
    methods = [m.split('(')[0].strip() for m in data['method']]  # Clean up method names
    times = data['time_ms']
    
    bars = ax2.bar(range(len(methods)), times, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, time) in enumerate(zip(bars, times)):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{time:.4f}ms',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    ax2.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Execution Time Comparison (1M elements)', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=15, ha='right')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Color the baseline differently
    bars[0].set_hatch('///')
    
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved accuracy_comparison.png")
    plt.close()

def create_detailed_analysis(data):
    """Create a detailed analysis figure"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Error reduction (%)
    ax1 = fig.add_subplot(gs[0, 0])
    methods_short = ['HW', 'Poly3', 'Poly4', 'Poly5']
    errors = [e * 1e6 for e in data['rmse_error']]  # Scale to micro units
    ax1.bar(methods_short, errors, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)
    ax1.set_ylabel('RMSE Error (×1e-6)', fontsize=11, fontweight='bold')
    ax1.set_title('Approximation Error (lower is better)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Speedup distribution
    ax2 = fig.add_subplot(gs[0, 1])
    speedups = data['speedup']
    ax2.bar(methods_short, speedups, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax2.set_ylabel('Speedup (x)', fontsize=11, fontweight='bold')
    ax2.set_title('Speedup vs Hardware __expf', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Efficiency curve (speedup per unit error)
    ax3 = fig.add_subplot(gs[1, 0])
    efficiency = [s / e if e > 0 else 0 for s, e in zip(speedups, [1e-10] + errors[1:])]
    colors_eff = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    ax3.plot(methods_short, efficiency, marker='o', markersize=10, linewidth=2, color='#e74c3c')
    ax3.fill_between(range(len(methods_short)), efficiency, alpha=0.3, color='#e74c3c')
    ax3.set_ylabel('Speedup / RMSE', fontsize=11, fontweight='bold')
    ax3.set_title('Efficiency (speedup per unit error)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Time breakdown
    ax4 = fig.add_subplot(gs[1, 1])
    times = data['time_ms']
    speedups_for_time = [1.0] + [1/s if s != 0 else 1.0 for s in speedups[1:]]
    theoretical_times = [t * (1/s) if s != 0 else t for t, s in zip([times[0]], speedups[1:])]
    
    x = np.arange(len(methods_short))
    width = 0.35
    
    ax4.bar(x - width/2, times, width, label='Actual Time', alpha=0.8, color='#3498db')
    ax4.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
    ax4.set_title('Execution Time per Method', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(methods_short)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Summary table as text
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('tight')
    ax5.axis('off')
    
    table_data = []
    table_data.append(['Method', 'Time (ms)', 'RMSE Error', 'Speedup', 'Ops/Error'])
    
    for i, method in enumerate(data['method']):
        ops_per_error = speedups[i] / (errors[i] if errors[i] > 0 else 1e-10)
        table_data.append([
            method,
            f"{times[i]:.4f}",
            f"{data['rmse_error'][i]:.2e}",
            f"{speedups[i]:.2f}x",
            f"{ops_per_error:.1e}"
        ])
    
    table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.25, 0.15, 0.2, 0.15, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('#ffffff')
    
    plt.savefig('detailed_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved detailed_analysis.png")
    plt.close()

def print_summary(data):
    """Print summary statistics"""
    print("\n" + "="*70)
    print("EXPONENTIAL APPROXIMATION BENCHMARK SUMMARY")
    print("="*70)
    
    for i, method in enumerate(data['method']):
        print(f"\n{method}:")
        print(f"  Execution Time:   {data['time_ms'][i]:.4f} ms")
        print(f"  RMSE Error:       {data['rmse_error'][i]:.2e}")
        print(f"  Speedup:          {data['speedup'][i]:.2f}x")
    
    # Find best accuracy/speed tradeoff
    best_idx = 0
    best_ratio = float('inf')
    
    for i in range(1, len(data['method'])):
        # Normalize speedup (1 is baseline) and error (lower is better)
        # We want high speedup with low error
        speedup_benefit = data['speedup'][i]
        error_cost = data['rmse_error'][i] / data['rmse_error'][0]  # Relative to hardware
        
        ratio = error_cost / speedup_benefit if speedup_benefit > 1 else float('inf')
        
        if ratio < best_ratio:
            best_ratio = ratio
            best_idx = i
    
    print("\n" + "-"*70)
    print(f"RECOMMENDATION: {data['method'][best_idx]}")
    print(f"  Best accuracy/speed tradeoff in this benchmark")
    print("="*70 + "\n")

def create_compute_bound_plot(filename='compute_bound_results.csv'):
    """Plot compute-bound benchmark: hardware vs polynomial, register-resident."""
    data = {'method': [], 'time_ms': [], 'speedup': []}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['method'].append(row['method'])
            data['time_ms'].append(float(row['time_ms']))
            data['speedup'].append(float(row['speedup']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Compute-Bound Benchmark (512 exp() calls per thread, register-resident)',
                 fontsize=13, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e']

    # Left: execution time
    bars = ax1.bar(data['method'], data['time_ms'], color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    for bar, t in zip(bars, data['time_ms']):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{t:.4f} ms', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Execution Time', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Right: speedup bar with 1x baseline line
    speedup_val = data['speedup'][1]
    color = '#2ca02c' if speedup_val > 1.0 else '#d62728'
    bar = ax2.bar(['Polynomial Degree 4'], [speedup_val], color=color,
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Hardware baseline (1x)')
    ax2.text(bar[0].get_x() + bar[0].get_width() / 2, speedup_val,
             f'{speedup_val:.2f}x', ha='center', va='bottom', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Speedup vs Hardware __expf', fontsize=12, fontweight='bold')
    ax2.set_title('Speedup (compute-bound regime)', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(1.5, speedup_val + 0.2))
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    verdict = "Polynomial FASTER — matches Flash Attention 4 claim" if speedup_val > 1.05 \
         else "Hardware wins — polynomial overhead exceeds gain" if speedup_val < 0.95 \
         else "Roughly equal — hardware throughput ceiling reached"
    fig.text(0.5, 0.01, verdict, ha='center', fontsize=11,
             color='green' if speedup_val > 1.05 else 'red', fontweight='bold')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('compute_bound_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved compute_bound_comparison.png")
    plt.close()


if __name__ == "__main__":
    try:
        print("Reading benchmark results...")
        data = read_benchmark_csv('benchmark_results.csv')

        print("Creating visualizations...")
        create_accuracy_plot(data)
        create_detailed_analysis(data)

        print("\nGenerating summary...")
        print_summary(data)

        if __import__('os').path.exists('compute_bound_results.csv'):
            print("\nCreating compute-bound plot...")
            create_compute_bound_plot()
            print("  - compute_bound_comparison.png")

        print("\nAll plots generated successfully!")
        print("Files created:")
        print("  - accuracy_comparison.png")
        print("  - detailed_analysis.png")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure benchmark_results.csv exists (run exponential_kernel first)")
