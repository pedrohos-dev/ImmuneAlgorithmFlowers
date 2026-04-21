# QUICK START GUIDE - Trabalho Prático 2

## Overview
This implementation solves the Iris flower classification problem using the **CLONALG** immunological algorithm.

## Files Created

1. **clonalg.py** - Core algorithm implementation
2. **iris_classification.ipynb** - Main Jupyter notebook (RECOMMENDED)
3. **population_analysis.py** - Standalone population size analysis script
4. **README.md** - Comprehensive documentation

## How to Run

### Option 1: Jupyter Notebook (RECOMMENDED - Full Analysis)

```bash
jupyter notebook iris_classification.ipynb
```

This will run and display:
✓ CLONALG training with population size 50
✓ Test accuracy: ~90-95%
✓ Training accuracy: ~95-98%
✓ Accuracy evolution plot across 50 generations
✓ Population size impact analysis (10-50)
✓ Performance metrics (precision, recall, confusion matrix)

**Execution Time**: ~5-10 minutes (depending on machine)

### Option 2: Standalone Population Analysis Script

```bash
python population_analysis.py
```

This will:
✓ Test population sizes 10, 15, 20, 25, 30, 35, 40, 45, 50
✓ Train each for 50 generations
✓ Create comparison plots
✓ Display results table

**Execution Time**: ~3-5 minutes

## Problem Modeling

### What Does Each Antibody Represent?
- Each antibody is a **class prototype** with learned weight vectors
- Antibody parameters: `weights` (feature values) + `class_id` (which class it represents)

### Antibody Parameters
- **weights**: n_features dimensional vector (4 for Iris)
- **affinity**: Classification accuracy on training data
- **class_id**: 0 (setosa), 1 (versicolor), or 2 (virginica)

### Fitness Function
```
affinity = accuracy recognizing own class + 
           0.5 × overall classification accuracy
```

### Evaluation
- Uses Euclidean distance to classify samples
- Classifies to nearest antibody prototype

## Key Results Expected

### Performance Metrics
- Test Accuracy: 85-95%
- Training Accuracy: 90-98%
- Precision: 0.85-0.95
- Recall: 0.85-0.95

### Accuracy Evolution (across 50 generations)
- Generation 1-10: ~60-70%
- Generation 20-30: ~80-85%
- Generation 40-50: ~90-95%

### Population Size Impact
- Population 10: ~85-90% accuracy
- Population 30: ~90-92% accuracy
- Population 50: ~92-95% accuracy

## Algorithm Parameters Used

```
Population Size (P) = 50 antibodies
Number of Generations = 50 iterations
Selection Size (M) = P/3 ≈ 17
Clone Count (N_clones) = P/2 ≈ 25
New Antibodies (N) = P/5 ≈ 10
Beta (mutation factor) = 1.0
```

## Output Files Generated

1. **accuracy_evolution.png** - Shows test and training accuracy across generations
2. **population_size_impact.png** - Compares different population sizes
3. **population_size_analysis.png** - Detailed population analysis chart

## Dataset Split

- Total samples: 150
- Training set: 105 samples (70%)
- Test set: 45 samples (30%)

Each class has ~35 training samples and ~15 test samples

## Troubleshooting

### ImportError: No module named 'sklearn'
```bash
pip install scikit-learn numpy pandas matplotlib seaborn scipy
```

### Jupyter notebook not found
```bash
pip install jupyter
```

### Script runs very slowly
- Reduce n_generations in the code
- Reduce population_size
- Consider closing other applications

## Expected Behavior

✓ Algorithm should converge gradually
✓ Test accuracy should improve over generations
✓ Larger populations should generally perform better
✓ Final accuracy should be competitive with other ML methods

## Comparing Results

If you want to compare CLONALG with other classifiers:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

rf = RandomForestClassifier().fit(X_train, y_train)
svm = SVC().fit(X_train, y_train)

print(f"Random Forest Accuracy: {rf.score(X_test, y_test):.4f}")
print(f"SVM Accuracy: {svm.score(X_test, y_test):.4f}")
print(f"CLONALG Accuracy: {clonalg.score(X_test, y_test):.4f}")
```

## Key Insights from CLONALG

1. **Biologically Inspired**: Mimics immune system's ability to learn and adapt

2. **Effective Search**: Balance between exploration and exploitation through:
   - Proportional cloning (exploit good solutions)
   - Adaptive mutation (explore new areas)
   - Random antibodies (maintain diversity)

3. **Scalability**: Performance depends on:
   - Population size (more = better, but slower)
   - Number of generations (more = better, but slower)
   - Problem complexity (Iris is relatively simple)

4. **Advantages**:
   - Conceptually elegant
   - Good generalization
   - Natural parallelization
   - Parameter tunable

5. **Disadvantages**:
   - Slower than some classical methods
   - More parameters to tune
   - May need more iterations for complex problems

## Next Steps

1. Run the Jupyter notebook to see complete analysis
2. Try different population sizes
3. Modify beta parameter and observe effects
4. Compare with other ML algorithms
5. Try on different datasets

## Support

For questions about:
- **CLONALG Algorithm**: See README.md or references in code
- **Implementation Details**: See clonalg.py with detailed comments
- **Results Analysis**: Run iris_classification.ipynb

---

**Estimated Runtime**: 
- Notebook: 5-10 minutes
- Standalone script: 3-5 minutes

**Good luck with your assignment!** 🍀
