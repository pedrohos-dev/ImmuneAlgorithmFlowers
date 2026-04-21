"""
Population Size Impact Analysis
Quick script to analyze the impact of population size on CLONALG accuracy
"""

import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from clonalg import CLONALG

# Set random seed
np.random.seed(42)

# Load data
print("Loading Iris dataset...")
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True, random_state=42, stratify=y
)

n_classes = 3
n_features = 4
n_gen_analysis = 50

# Test different population sizes
population_sizes = range(10, 51, 5)
results = []

print(f"\nTesting population sizes: {list(population_sizes)}")
print(f"Number of generations: {n_gen_analysis}\n")

for pop_size in population_sizes:
    print(f"Training with population size: {pop_size:2d}...", end=" ")
    
    # Train model
    clonalg = CLONALG(
        population_size=pop_size,
        n_generations=n_gen_analysis,
        n_classes=n_classes,
        n_features=n_features,
        beta=1.0
    )
    
    clonalg.fit(X_train, y_train)
    
    # Evaluate
    test_acc = clonalg.score(X_test, y_test)
    train_acc = clonalg.score(X_train, y_train)
    
    results.append({
        'Population Size': pop_size,
        'Test Accuracy': test_acc,
        'Train Accuracy': train_acc
    })
    
    print(f"Test Acc: {test_acc:.4f}, Train Acc: {train_acc:.4f}")

# Create results dataframe
results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("POPULATION SIZE ANALYSIS RESULTS")
print("=" * 60)
print(results_df.to_string(index=False))

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Accuracy vs Population Size
axes[0].plot(results_df['Population Size'], results_df['Test Accuracy'], 'o-', 
             linewidth=2.5, markersize=8, color='#FF6B6B', label='Test Accuracy')
axes[0].plot(results_df['Population Size'], results_df['Train Accuracy'], 's--', 
             linewidth=2, markersize=7, color='#4ECDC4', label='Train Accuracy')

axes[0].set_xlabel('Population Size', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
axes[0].set_title('Impact of Population Size on Final Accuracy\n(50 Generations)', 
                  fontsize=13, fontweight='bold')
axes[0].set_ylim([0, 1.05])
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10)

# Plot 2: Bar chart
axes[1].bar(results_df['Population Size'], results_df['Test Accuracy'], 
            width=4, color='#45B7D1', alpha=0.8, edgecolor='black', linewidth=1.5)

axes[1].set_xlabel('Population Size', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
axes[1].set_title('Test Accuracy by Population Size', fontsize=13, fontweight='bold')
axes[1].set_ylim([0, 1.05])
axes[1].grid(True, alpha=0.3, axis='y')

# Add value labels
for pop_size, acc in zip(results_df['Population Size'], results_df['Test Accuracy']):
    axes[1].text(pop_size, acc + 0.02, f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('population_size_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Summary
print("\n" + "=" * 60)
print("ANALYSIS SUMMARY")
print("=" * 60)
best_idx = results_df['Test Accuracy'].idxmax()
worst_idx = results_df['Test Accuracy'].idxmin()

best_pop = results_df.loc[best_idx, 'Population Size']
best_acc = results_df.loc[best_idx, 'Test Accuracy']
worst_pop = results_df.loc[worst_idx, 'Population Size']
worst_acc = results_df.loc[worst_idx, 'Test Accuracy']

print(f"Best population size:  {best_pop} (Accuracy: {best_acc:.4f})")
print(f"Worst population size: {worst_pop} (Accuracy: {worst_acc:.4f})")
print(f"Accuracy difference: {best_acc - worst_acc:.4f}")
print(f"Average accuracy: {results_df['Test Accuracy'].mean():.4f}")
