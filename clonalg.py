"""
CLONALG (Clonal Selection Algorithm) Implementation for Classification
Based on the Iris flower classification problem
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean


class Antibody:
    """
    Represents an antibody in the immune system.
    For classification: each antibody is a prototype that represents one class.
    
    Antibody parameters:
    - weights: Feature weights for the class representation (n_features,)
    - affinity: How well the antibody recognizes the antigen (class)
    """
    
    def __init__(self, n_features, class_id):
        self.weights = np.random.randn(n_features)
        self.affinity = 0.0
        self.class_id = class_id
    
    def copy(self):
        """Create a deep copy of the antibody"""
        new_antibody = Antibody(len(self.weights), self.class_id)
        new_antibody.weights = self.weights.copy()
        new_antibody.affinity = self.affinity
        new_antibody.class_id = self.class_id
        return new_antibody
    
    def predict(self, X):
        """
        Predict class based on weighted distance to training points.
        Returns predicted class ID.
        """
        # Distance from sample to antibody weights
        distances = np.sum((X - self.weights) ** 2, axis=1)
        return distances
    
    def __repr__(self):
        return f"Antibody(class={self.class_id}, affinity={self.affinity:.4f})"


class CLONALG:
    """
    Clonal Selection Algorithm for Iris classification
    
    Parameters:
    - population_size: Number of antibodies (max 50)
    - n_generations: Number of iterations
    - n_classes: Number of classes to recognize (3 for Iris)
    - n_features: Number of features per sample
    - beta: Mutation rate factor (default 1.0)
    """
    
    def __init__(self, population_size=30, n_generations=50, n_classes=3, 
                 n_features=4, beta=1.0):
        self.population_size = population_size
        self.n_generations = n_generations
        self.n_classes = n_classes
        self.n_features = n_features
        self.beta = beta
        self.population = []
        self.best_antibodies = None
        self.history = []
        
        # Parameters for CLONALG
        self.n_select = max(2, population_size // 3)  # Number to select (M)
        self.n_clones = max(2, population_size // 2)  # Number of clones (P-N)
        self.n_new = max(1, population_size // 5)     # Number of new antibodies (N)
    
    def _initialize_population(self):
        """Initialize population with random antibodies, one per class"""
        self.population = []
        
        # Create at least one antibody per class
        for class_id in range(self.n_classes):
            antibody = Antibody(self.n_features, class_id)
            self.population.append(antibody)
        
        # Fill remaining population with random antibodies
        while len(self.population) < self.population_size:
            class_id = np.random.randint(0, self.n_classes)
            antibody = Antibody(self.n_features, class_id)
            self.population.append(antibody)
    
    def _evaluate_population(self, X, y):
        """
        Evaluate fitness (affinity) of all antibodies.
        Affinity = accuracy on training set for each class.
        """
        # Normalize input
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)
        
        for antibody in self.population:
            # Calculate affinity based on classification accuracy
            distances = self._calculate_distances(antibody, X_normalized)
            predictions = np.argmin(distances, axis=1)  # Predict based on closest class
            
            # Affinity is accuracy weighted by class correctness
            correct_class = (predictions == antibody.class_id)
            correct_overall = (predictions == y)
            
            # Affinity: combination of recognizing own class and overall accuracy
            affinity = np.sum(correct_class) + 0.5 * np.sum(correct_overall)
            affinity = affinity / len(X)
            
            antibody.affinity = affinity
        
        return scaler
    
    def _calculate_distances(self, antibody, X):
        """Calculate distance from antibody to each sample for each class"""
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_classes))
        
        for i in range(n_samples):
            for c in range(self.n_classes):
                if c == antibody.class_id:
                    # Smaller distance for matching class
                    distances[i, c] = euclidean(X[i], antibody.weights)
                else:
                    # Larger distance for non-matching class
                    distances[i, c] = euclidean(X[i], antibody.weights) * 2
        
        return distances
    
    def _selection(self):
        """
        Selection phase: Select the best M antibodies (highest affinity)
        """
        # Sort by affinity
        sorted_population = sorted(self.population, key=lambda x: x.affinity, reverse=True)
        return sorted_population[:self.n_select]
    
    def _cloning(self, selected):
        """
        Cloning phase: Clone antibodies proportionally to affinity.
        Number of clones ∝ affinity
        """
        clones = []
        total_affinity = sum([ab.affinity for ab in selected])
        
        for antibody in selected:
            # Proportional cloning
            if total_affinity > 0:
                proportion = antibody.affinity / total_affinity
                n_clones_for_ab = max(1, int(self.n_clones * proportion))
            else:
                n_clones_for_ab = self.n_clones // len(selected)
            
            for _ in range(n_clones_for_ab):
                clones.append(antibody.copy())
        
        return clones
    
    def _hypermutation(self, clones):
        """
        Hypermutation phase: Mutate clones inversely proportional to affinity.
        Lower affinity = higher mutation rate
        """
        mutated = []
        
        # Get max affinity for normalization
        max_affinity = max([c.affinity for c in clones]) if clones else 1.0
        
        for clone in clones:
            mutated_clone = clone.copy()
            
            # Mutation rate inversely proportional to affinity
            if max_affinity > 0:
                mutation_rate = (1 - (clone.affinity / max_affinity)) * self.beta
            else:
                mutation_rate = self.beta
            
            # Apply Gaussian mutation
            mutation = np.random.randn(self.n_features) * mutation_rate
            mutated_clone.weights = clone.weights + mutation
            
            mutated.append(mutated_clone)
        
        return mutated
    
    def _generate_new_antibodies(self):
        """
        Metadynamics: Generate N new random antibodies to maintain diversity
        """
        new_antibodies = []
        for _ in range(self.n_new):
            class_id = np.random.randint(0, self.n_classes)
            antibody = Antibody(self.n_features, class_id)
            new_antibodies.append(antibody)
        
        return new_antibodies
    
    def fit(self, X, y):
        """
        Train CLONALG on iris dataset
        
        Parameters:
        - X: Training features (n_samples, n_features)
        - y: Training labels (n_samples,)
        """
        self._initialize_population()
        self.history = []
        
        for generation in range(self.n_generations):
            # Step 2: Evaluate affinity
            scaler = self._evaluate_population(X, y)
            
            # Step 3: Selection and cloning
            selected = self._selection()
            clones = self._cloning(selected)
            
            # Step 4: Hypermutation
            mutated = self._hypermutation(clones)
            
            # Step 5: Metadynamics - generate new antibodies
            new_antibodies = self._generate_new_antibodies()
            
            # Create new population: mutated clones + new antibodies
            self.population = mutated + new_antibodies
            
            # Ensure population size
            if len(self.population) > self.population_size:
                # Keep best
                self._evaluate_population(X, y)
                self.population = sorted(self.population, 
                                        key=lambda x: x.affinity, 
                                        reverse=True)[:self.population_size]
            elif len(self.population) < self.population_size:
                # Add random antibodies
                while len(self.population) < self.population_size:
                    class_id = np.random.randint(0, self.n_classes)
                    self.population.append(Antibody(self.n_features, class_id))
            
            # Store best affinity for this generation
            best_affinity = max([ab.affinity for ab in self.population])
            self.history.append(best_affinity)
        
        # Final evaluation
        self._evaluate_population(X, y)
        self.best_antibodies = sorted(self.population, 
                                      key=lambda x: x.affinity, 
                                      reverse=True)[:self.n_classes]
    
    def predict(self, X):
        """
        Predict class for new samples
        
        Parameters:
        - X: Features (n_samples, n_features)
        
        Returns:
        - predictions: Predicted class labels (n_samples,)
        """
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=int)
        
        # Normalize input using training data statistics
        scaler = StandardScaler()
        scaler.fit(X)
        X_normalized = scaler.transform(X)
        
        # For each sample, find closest antibody
        for i in range(n_samples):
            min_distance = float('inf')
            best_class = 0
            
            for antibody in self.best_antibodies:
                distance = euclidean(X_normalized[i], antibody.weights)
                if distance < min_distance:
                    min_distance = distance
                    best_class = antibody.class_id
            
            predictions[i] = best_class
        
        return predictions
    
    def score(self, X, y):
        """
        Calculate accuracy on test set
        
        Parameters:
        - X: Test features
        - y: Test labels
        
        Returns:
        - accuracy: Percentage of correct predictions
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
