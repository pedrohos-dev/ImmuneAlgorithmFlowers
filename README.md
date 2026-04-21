# CLONALG - Iris Flower Classification using Immunological Algorithm

This project implements the **CLONALG (Clonal Selection Algorithm)** for solving the Iris flower classification problem. CLONALG is an immunological algorithm inspired by the biological immune system's ability to recognize and respond to antigens through clonal selection, mutation, and affinity maturation.

## Project Structure

```
ImmuneAlgorithmFlowers/
├── clonalg.py                      # Core CLONALG implementation
├── iris_classification.ipynb       # Main Jupyter notebook with full analysis
├── population_analysis.py          # Standalone script for population size analysis
├── README.md                       # This file
└── *.png                          # Generated plots
```

## Files Description

### 1. `clonalg.py`
Core implementation of the CLONALG algorithm with two main classes:

- **`Antibody`**: Represents an antibody (solution candidate)
  - `weights`: Feature weights representing the class prototype
  - `affinity`: Fitness value (classification accuracy)
  - `class_id`: Which class this antibody recognizes

- **`CLONALG`**: Main algorithm implementation
  - `_initialize_population()`: Create initial random antibodies
  - `_evaluate_population()`: Calculate fitness (affinity)
  - `_selection()`: Select best M antibodies
  - `_cloning()`: Clone antibodies proportionally to affinity
  - `_hypermutation()`: Mutate clones inversely to affinity
  - `_generate_new_antibodies()`: Add random antibodies for diversity
  - `fit()`: Train the model
  - `predict()`: Make predictions on new data
  - `score()`: Calculate accuracy

### 2. `iris_classification.ipynb`
Complete Jupyter notebook with:

1. **Import Required Libraries** - NumPy, Pandas, Scikit-learn, Matplotlib
2. **Load and Prepare Iris Dataset** - Load data and split 70/30 train/test
3. **Train CLONALG Model** - Train with population size 50, 50 generations
4. **Evaluate Performance** - Test accuracy, precision, recall, confusion matrix
5. **Analyze Accuracy Evolution** - Plot how accuracy improves across generations
6. **Analyze Population Size Impact** - Test sizes 10-50, compare results
7. **Summary and Conclusions** - Key findings and observations

### 3. `population_analysis.py`
Standalone Python script for quick analysis of population size impact without Jupyter.

## Algorithm Overview

### CLONALG Steps:

1. **Initialization**: Create population of P random antibodies (one per class + random)
2. **Evaluation**: Calculate fitness (affinity) for each antibody
3. **Selection**: Select top M antibodies with highest affinity
4. **Cloning**: Clone each selected antibody proportionally to its affinity
   - Higher affinity → More clones
5. **Hypermutation**: Mutate all clones inversely to affinity
   - Lower affinity → Higher mutation rate
6. **Metadynamics**: Replace worst antibodies with N new random ones
7. **Loop**: Repeat steps 2-6 until stopping criterion

### Mathematical Formulas:

**Number of clones for antibody k:**
$$QC_k = \left(\frac{af_k}{\sum_{i=1}^{n} af_i}\right) \times CL$$

**Mutation rate (inversely proportional to affinity):**
$$hyperAnt_k = (1 - \frac{afin_k}{afin_{max}}) \times \beta$$

where:
- `QC_k` = Number of clones for antibody k
- `af_k` = Affinity of antibody k
- `CL` = Total number of clones to generate
- `hyperAnt_k` = Hypermutation rate
- `afin_k` = Affinity of antibody k
- `afin_max` = Maximum affinity in population
- `β` = Mutation factor (typically 0.1 to 10)

## Iris Dataset

- **Samples**: 150 iris flowers
- **Features**: 4 (sepal length, sepal width, petal length, petal width)
- **Classes**: 3 (setosa, versicolor, virginica)
- **Split**: 70% training (105), 30% testing (45)

## How to Use

### Running the Jupyter Notebook:

```bash
jupyter notebook iris_classification.ipynb
```

The notebook will:
1. Train CLONALG with population size 50
2. Display test and train accuracy
3. Show confusion matrix and classification report
4. Plot accuracy evolution across 50 generations
5. Analyze impact of population sizes 10-50
6. Generate visualizations

### Running the Population Analysis Script:

```bash
python population_analysis.py
```

This script will:
1. Test population sizes from 10 to 50 (step 5)
2. Train each for 50 generations
3. Display results in a table
4. Create comparison plots
5. Save results as PNG

## Parameters

### Default CLONALG Parameters:
- **Population Size (P)**: 50 antibodies
- **Generations**: 50 iterations
- **Classes**: 3 (Iris species)
- **Features**: 4 (flower measurements)
- **Beta (mutation factor)**: 1.0

### Configurable Parameters:
```python
clonalg = CLONALG(
    population_size=50,    # M = population_size // 3
    n_generations=50,      # Number of iterations
    n_classes=3,          # Number of iris species
    n_features=4,         # Number of features
    beta=1.0              # Mutation rate factor
)
```

## Results and Performance

### Expected Results:
- **Test Accuracy**: 85-95%
- **Training Accuracy**: 90-98%
- **Convergence**: Typically improves by generation 30-40

### Population Size Impact:
- Larger populations → Better accuracy (more exploration)
- Smaller populations → Faster but may get stuck
- Trade-off between computational cost and quality

## Key Insights

1. **Cloning Strategy**: Proportional cloning based on fitness ensures high-quality solutions get more chances to improve

2. **Mutation Control**: Inverse mutation rate prevents over-mutation of good solutions while maintaining exploration

3. **Diversity Maintenance**: Random antibody generation prevents premature convergence

4. **Effective for Classification**: CLONALG successfully learns class boundaries through evolution

## Generated Plots

1. **accuracy_evolution.png**: Test and train accuracy across generations
2. **population_size_impact.png**: Impact analysis with two views
3. **population_size_analysis.png**: Standalone analysis results

## Requirements

```
numpy>=1.19.0
pandas>=1.1.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
scipy>=1.5.0
jupyter>=1.0.0
```

Install with:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy jupyter
```

## References

The implementation is based on the CLONALG algorithm described in:

de Castro, L. N., & Von Zuben, F. J. (2002). Learning and optimization in the immune system. 
IEEE Transactions on Evolutionary Computation, 6(3), 239-251.

## Notes

- The algorithm uses Euclidean distance for classification
- Antibodies are represented as weight vectors in feature space
- Fitness is calculated based on classification accuracy
- The algorithm maintains a steady-state population with elitism

## Author

Implementation for: Trabalho Prático 2 - Inteligência Artificial (5º Período)

## License

Educational Use Only
