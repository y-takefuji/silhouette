import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from silclass import SilhouetteClassifier

class EnsembleRefinement:
    """
    Advanced clustering refinement using an ensemble of unsupervised metrics
    and multi-stage optimization.
    """
    
    def __init__(self, max_iterations=30, threshold=0.0005, ensemble_weights=None, 
                 n_neighbors=10, use_connectivity=True, adaptive_weights=True,
                 batch_size=50, early_stopping=5):
        """
        Initialize the ensemble refinement process.
        
        Parameters:
        -----------
        max_iterations : int
            Maximum number of refinement iterations
        threshold : float
            Convergence threshold for improvement in ensemble score
        ensemble_weights : dict or None
            Optional weights for the ensemble metrics. If None, use default weights.
        n_neighbors : int
            Number of neighbors to consider for connectivity-based refinement
        use_connectivity : bool
            Whether to use connectivity constraints during refinement
        adaptive_weights : bool
            Whether to adaptively adjust ensemble weights during refinement
        batch_size : int
            Number of points to process in each batch
        early_stopping : int
            Stop if no improvement after this many iterations
        """
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.adaptive_weights = adaptive_weights
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        
        # Default ensemble weights (higher is better for all metrics as configured)
        if ensemble_weights is None:
            self.ensemble_weights = {
                'silhouette': 1.0,           # Higher is better
                'davies_bouldin': -0.7,      # Lower is better, so negative weight
                'calinski_harabasz': 0.5,    # Higher is better
                'connectivity': 0.8 if use_connectivity else 0.0,  # Higher is better
                'density_ratio': 0.6         # Higher is better - new density-based metric
            }
        else:
            self.ensemble_weights = ensemble_weights
            
        self.n_neighbors = n_neighbors
        self.use_connectivity = use_connectivity
        
        # History tracking
        self.history = {
            'iteration': [],
            'ensemble_score': [],
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': [],
            'connectivity': [],
            'density_ratio': [],
            'swapped_points': []
        }
        
        # Storage for adaptive metrics
        self.baseline_scores = {}
        self.score_improvements = {}
        
    def calculate_density_ratio(self, X, labels):
        """
        Calculate the ratio of within-cluster density to between-cluster density.
        Higher values indicate better clustering.
        
        Parameters:
        -----------
        X : array-like
            Input data features
        labels : array-like
            Cluster labels
            
        Returns:
        --------
        density_ratio : float
            Ratio of within-cluster to between-cluster density
        """
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0.0
            
        # Calculate pairwise distances
        try:
            distances = squareform(pdist(X))
            
            # Calculate within-cluster and between-cluster distances
            within_distances = []
            between_distances = []
            
            for i in range(len(X)):
                for j in range(i+1, len(X)):
                    if labels[i] == labels[j]:
                        within_distances.append(distances[i, j])
                    else:
                        between_distances.append(distances[i, j])
            
            if not within_distances or not between_distances:
                return 0.0
                
            # Convert to numpy arrays for efficient computation
            within_distances = np.array(within_distances)
            between_distances = np.array(between_distances)
            
            # Calculate densities (inverse of average distance)
            within_density = 1.0 / (np.mean(within_distances) + 1e-10)
            between_density = 1.0 / (np.mean(between_distances) + 1e-10)
            
            # Density ratio (higher is better)
            if between_density > 0:
                return within_density / between_density
            else:
                return 0.0
        except:
            return 0.0
        
    def calculate_ensemble_score(self, X, labels, connectivity_matrix=None):
        """
        Calculate weighted ensemble score from multiple clustering metrics.
        
        Parameters:
        -----------
        X : array-like
            Input data features
        labels : array-like
            Cluster labels
        connectivity_matrix : array-like or None
            Optional precomputed connectivity matrix
            
        Returns:
        --------
        ensemble_score : float
            Weighted ensemble score
        individual_scores : dict
            Dictionary of individual metric scores
        """
        individual_scores = {}
        
        # Calculate silhouette score if used in ensemble
        if 'silhouette' in self.ensemble_weights and self.ensemble_weights['silhouette'] != 0:
            try:
                individual_scores['silhouette'] = silhouette_score(X, labels)
            except:
                # Fallback if silhouette fails (e.g., single-element clusters)
                individual_scores['silhouette'] = -1.0
        
        # Calculate Davies-Bouldin score if used
        if 'davies_bouldin' in self.ensemble_weights and self.ensemble_weights['davies_bouldin'] != 0:
            # Check for singleton clusters which cause errors
            if len(np.unique(labels)) > 1 and all(np.bincount(labels) > 1):
                individual_scores['davies_bouldin'] = davies_bouldin_score(X, labels)
            else:
                # Penalize singleton clusters
                individual_scores['davies_bouldin'] = 10.0  # High value (poor score)
        
        # Calculate Calinski-Harabasz score if used
        if 'calinski_harabasz' in self.ensemble_weights and self.ensemble_weights['calinski_harabasz'] != 0:
            try:
                individual_scores['calinski_harabasz'] = calinski_harabasz_score(X, labels) / 1000.0  # Scale down
            except:
                individual_scores['calinski_harabasz'] = 0.0
                
        # Calculate connectivity score if used
        if 'connectivity' in self.ensemble_weights and self.ensemble_weights['connectivity'] != 0:
            if connectivity_matrix is None and self.use_connectivity:
                # Create connectivity matrix using k-nearest neighbors
                nbrs = NearestNeighbors(n_neighbors=min(self.n_neighbors + 1, len(X))).fit(X)
                distances, indices = nbrs.kneighbors(X)
                connectivity_matrix = np.zeros((len(X), len(X)))
                
                # Fill connectivity matrix (1 if connected, 0 otherwise)
                for i in range(len(indices)):
                    connectivity_matrix[i, indices[i]] = 1
                    
                # Remove self-connections
                np.fill_diagonal(connectivity_matrix, 0)
                
            if connectivity_matrix is not None:
                # Calculate connectivity consistency (% of connected points in same cluster)
                consistency_sum = 0
                connection_count = 0
                
                for i in range(len(X)):
                    # Get connected neighbors
                    neighbors = np.where(connectivity_matrix[i] > 0)[0]
                    if len(neighbors) > 0:
                        # Calculate percentage of neighbors in same cluster
                        same_cluster = np.sum(labels[neighbors] == labels[i])
                        consistency_sum += same_cluster / len(neighbors)
                        connection_count += 1
                
                if connection_count > 0:
                    individual_scores['connectivity'] = consistency_sum / connection_count
                else:
                    individual_scores['connectivity'] = 0.0
        
        # Calculate density ratio if used
        if 'density_ratio' in self.ensemble_weights and self.ensemble_weights['density_ratio'] != 0:
            individual_scores['density_ratio'] = self.calculate_density_ratio(X, labels)
                    
        # Calculate weighted ensemble score
        ensemble_score = 0.0
        total_weight = 0.0
        
        for metric, value in individual_scores.items():
            if metric in self.ensemble_weights:
                weight = self.ensemble_weights[metric]
                ensemble_score += weight * value
                total_weight += abs(weight)
        
        # Normalize by total weight
        if total_weight > 0:
            ensemble_score /= total_weight
        
        return ensemble_score, individual_scores
    
    def update_adaptive_weights(self, iteration, individual_scores, improvement):
        """
        Update ensemble weights based on which metrics are improving.
        
        Parameters:
        -----------
        iteration : int
            Current iteration number
        individual_scores : dict
            Current metric scores
        improvement : float
            Overall improvement in ensemble score
        """
        if not self.adaptive_weights:
            return
            
        # In the first iteration, just store baseline scores
        if iteration == 1:
            self.baseline_scores = individual_scores.copy()
            return
            
        # Calculate relative improvements for each metric
        for metric, current_score in individual_scores.items():
            if metric in self.baseline_scores:
                # For davies_bouldin, lower is better
                if metric == 'davies_bouldin':
                    rel_improvement = (self.baseline_scores[metric] - current_score) / (abs(self.baseline_scores[metric]) + 1e-10)
                else:
                    rel_improvement = (current_score - self.baseline_scores[metric]) / (abs(self.baseline_scores[metric]) + 1e-10)
                
                # Update tracking of improvement trend
                if metric not in self.score_improvements:
                    self.score_improvements[metric] = []
                self.score_improvements[metric].append(rel_improvement)
                
        # After several iterations, adjust weights
        if iteration >= 5 and iteration % 3 == 0:
            # Calculate average improvement for each metric over recent iterations
            avg_improvements = {}
            for metric, improvements in self.score_improvements.items():
                avg_improvements[metric] = np.mean(improvements[-3:])
            
            # Normalize improvements
            total_improvement = sum(abs(imp) for imp in avg_improvements.values())
            if total_improvement > 0:
                norm_improvements = {m: imp/total_improvement for m, imp in avg_improvements.items()}
                
                # Update weights gradually (30% adjustment)
                adjustment_factor = 0.3
                for metric in self.ensemble_weights:
                    if metric in norm_improvements:
                        # Increase weight for metrics showing good improvement
                        self.ensemble_weights[metric] *= (1 + adjustment_factor * norm_improvements[metric])
                        
                # Re-normalize weights
                total_weight = sum(abs(w) for w in self.ensemble_weights.values())
                if total_weight > 0:
                    scale_factor = len(self.ensemble_weights) / total_weight
                    for metric in self.ensemble_weights:
                        self.ensemble_weights[metric] *= scale_factor
                
                print(f"\nAdjusted ensemble weights:")
                for metric, weight in self.ensemble_weights.items():
                    print(f"  {metric}: {weight:.4f}")
    
    def estimate_point_contributions(self, X, labels, connectivity_matrix=None):
        """
        Estimate the contribution of each point to the overall ensemble score.
        
        Parameters:
        -----------
        X : array-like
            Input data features
        labels : array-like
            Current cluster labels
        connectivity_matrix : array-like or None
            Optional precomputed connectivity matrix
            
        Returns:
        --------
        contributions : array
            Estimated contribution of each point to the ensemble score
            (negative values indicate potential improvement by label flip)
        """
        # Get baseline ensemble score
        base_score, _ = self.calculate_ensemble_score(X, labels, connectivity_matrix)
        
        # For binary clustering, estimate improvement from flipping each point
        contributions = np.zeros(len(X))
        
        # Sample a subset of points for efficiency if dataset is large
        sample_size = min(len(X), 1000)  # Cap at 1000 evaluations for large datasets
        if len(X) > sample_size:
            # Stratified sampling to ensure we evaluate points from both clusters
            indices0 = np.where(labels == 0)[0]
            indices1 = np.where(labels == 1)[0]
            
            sample0 = np.random.choice(indices0, min(len(indices0), sample_size//2), replace=False)
            sample1 = np.random.choice(indices1, min(len(indices1), sample_size//2), replace=False)
            indices = np.concatenate([sample0, sample1])
            
            # Add some additional points with neighbors from the opposite class
            if len(indices) < sample_size and self.use_connectivity and connectivity_matrix is not None:
                boundary_candidates = []
                for i in range(len(labels)):
                    if i not in indices:
                        neighbors = np.where(connectivity_matrix[i] > 0)[0]
                        if any(labels[neighbors] != labels[i]):
                            boundary_candidates.append(i)
                
                if boundary_candidates:
                    additional = np.random.choice(
                        boundary_candidates, 
                        min(len(boundary_candidates), sample_size - len(indices)), 
                        replace=False
                    )
                    indices = np.concatenate([indices, additional])
        else:
            indices = np.arange(len(X))
            
        # For each point in the sample, estimate the effect of flipping its label
        for idx in indices:
            # Create temporary labels with this point's label flipped
            temp_labels = labels.copy()
            temp_labels[idx] = 1 - temp_labels[idx]  # Flip binary label
            
            # Calculate new score
            new_score, _ = self.calculate_ensemble_score(X, temp_labels, connectivity_matrix)
            
            # Contribution is the change in score (negative means score would improve)
            contributions[idx] = base_score - new_score
            
        # For unsampled points, use a nearest neighbor approach combined with local density
        if len(X) > sample_size:
            # Use nearest neighbor interpolation with inverse distance weighting
            sampled_points = X[indices]
            sampled_contributions = contributions[indices]
            
            # Find k nearest sampled points for each unsampled point
            k = min(5, len(indices))
            nbrs = NearestNeighbors(n_neighbors=k).fit(sampled_points)
            unsampled_indices = np.setdiff1d(np.arange(len(X)), indices)
            
            if len(unsampled_indices) > 0:
                distances, nn_indices = nbrs.kneighbors(X[unsampled_indices])
                
                # Inverse distance weighting
                for i, ui in enumerate(unsampled_indices):
                    # Avoid division by zero
                    weights = 1.0 / (distances[i] + 1e-10)
                    weights = weights / np.sum(weights)  # Normalize weights
                    
                    # Calculate weighted average contribution
                    contributions[ui] = np.sum(weights * sampled_contributions[nn_indices[i]])
                    
                    # Add small penalty for points with opposite-class neighbors
                    if self.use_connectivity and connectivity_matrix is not None:
                        neighbors = np.where(connectivity_matrix[ui] > 0)[0]
                        if len(neighbors) > 0:
                            opposite_ratio = np.mean(labels[neighbors] != labels[ui])
                            contributions[ui] -= opposite_ratio * 0.01  # Small push to consider flipping boundary points
        
        return contributions
        
    def fit_refine(self, X, initial_labels=None, y_true=None, major_ratio=0.5):
        """
        Perform iterative refinement using the ensemble approach.
        
        Parameters:
        -----------
        X : array-like
            Input data features
        initial_labels : array-like or None
            Initial cluster labels. If None, SilhouetteClassifier is used
        y_true : array-like or None
            True labels for evaluation only (not used in refinement)
        major_ratio : float
            Ratio for the major class if using SilhouetteClassifier
            
        Returns:
        --------
        refined_labels : array-like
            The refined cluster labels
        """
        # Step 1: Get initial clustering if not provided
        if initial_labels is None:
            classifier = SilhouetteClassifier()
            current_labels = classifier.fit_predict(X, major_ratio=major_ratio)
        else:
            current_labels = np.array(initial_labels)
        
        # Step 2: Precompute connectivity matrix if needed
        connectivity_matrix = None
        if self.use_connectivity:
            # Create connectivity matrix using k-nearest neighbors
            nbrs = NearestNeighbors(n_neighbors=min(self.n_neighbors + 1, len(X))).fit(X)
            distances, indices = nbrs.kneighbors(X)
            connectivity_matrix = np.zeros((len(X), len(X)))
            
            # Fill connectivity matrix (1 if connected, 0 otherwise)
            for i in range(len(indices)):
                connectivity_matrix[i, indices[i]] = 1
                
            # Remove self-connections
            np.fill_diagonal(connectivity_matrix, 0)
        
        # Step 3: Calculate initial ensemble score
        ensemble_score, individual_scores = self.calculate_ensemble_score(
            X, current_labels, connectivity_matrix)
        
        # Store best solution seen so far
        best_labels = current_labels.copy()
        best_score = ensemble_score
        best_iteration = 0
        
        # Record initial state
        self.history['iteration'].append(0)
        self.history['ensemble_score'].append(ensemble_score)
        self.history['swapped_points'].append(0)
        
        # Record individual metric scores
        for metric, score in individual_scores.items():
            if metric not in self.history:
                self.history[metric] = []
            self.history[metric].append(score)
        
        # Ensure all history lists have initial values
        for key in ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'connectivity', 'density_ratio']:
            if key not in self.history:
                self.history[key] = [None]
        
        # Print initial state
        print(f"Initial ensemble score: {ensemble_score:.4f}")
        for metric, score in individual_scores.items():
            print(f"Initial {metric} score: {score:.4f}")
        
        # Evaluate initial clustering if true labels are provided (for reporting only)
        if y_true is not None:
            self._evaluate_clustering(current_labels, y_true, "Initial")
            
        # Main refinement loop
        iteration = 0
        consecutive_no_improvement = 0
        while iteration < self.max_iterations and consecutive_no_improvement < self.early_stopping:
            iteration += 1
            print(f"\nIteration {iteration}:")
            
            # Estimate point contributions to ensemble score
            contributions = self.estimate_point_contributions(X, current_labels, connectivity_matrix)
            
            # Identify points with negative contributions (flipping would improve score)
            candidates = np.where(contributions < -self.threshold)[0]
            
            if len(candidates) == 0:
                print("No points found that would improve score. Trying smaller threshold...")
                # Try with a smaller threshold
                candidates = np.where(contributions < 0)[0]
                
                if len(candidates) == 0:
                    print("Still no candidate points found. Refinement complete.")
                    consecutive_no_improvement += 1
                    continue
            
            # Sort candidates by potential improvement (most negative contributions first)
            candidates = candidates[np.argsort(contributions[candidates])]
            
            # Process candidates in batches for efficiency
            batch_size = min(self.batch_size, len(candidates))
            temp_labels = current_labels.copy()
            flipped_count = 0
            temp_score = ensemble_score
            
            for i in range(0, len(candidates), batch_size):
                batch = candidates[i:i+batch_size]
                
                # Flip labels for this batch
                for idx in batch:
                    temp_labels[idx] = 1 - temp_labels[idx]
                    flipped_count += 1
                
                # Calculate new score with batch flips
                batch_score, batch_metrics = self.calculate_ensemble_score(X, temp_labels, connectivity_matrix)
                
                # If score improved, keep these flips, otherwise revert
                if batch_score > temp_score:
                    temp_score = batch_score
                    # Keep the flips (already in temp_labels)
                else:
                    # Revert this batch
                    for idx in batch:
                        temp_labels[idx] = 1 - temp_labels[idx]
                        flipped_count -= 1
            
            # If any points were successfully flipped
            if flipped_count > 0:
                new_score, new_individual_scores = self.calculate_ensemble_score(
                    X, temp_labels, connectivity_matrix)
                
                print(f"Flipped {flipped_count} points")
                print(f"New ensemble score: {new_score:.4f} (previous: {ensemble_score:.4f})")
                
                # Update current solution
                current_labels = temp_labels
                
                # Update best solution if improved
                if new_score > best_score:
                    improvement = new_score - best_score
                    best_score = new_score
                    best_labels = current_labels.copy()
                    best_iteration = iteration
                    print(f"New best score: {best_score:.4f} (improved by {improvement:.6f})")
                    consecutive_no_improvement = 0
                    
                    # Update adaptive weights based on improvement
                    self.update_adaptive_weights(iteration, new_individual_scores, improvement)
                else:
                    consecutive_no_improvement += 1
                    
                # Always update ensemble score to current value
                ensemble_score = new_score
                
                # Record this iteration
                self.history['iteration'].append(iteration)
                self.history['ensemble_score'].append(new_score)
                self.history['swapped_points'].append(flipped_count)
                
                # Record individual metric scores
                for metric, score in new_individual_scores.items():
                    self.history[metric].append(score)
                
                # Ensure all history lists have values for this iteration
                for key in ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'connectivity', 'density_ratio']:
                    if key not in new_individual_scores and key in self.history:
                        self.history[key].append(None)
            else:
                print("No improvements found in this iteration.")
                consecutive_no_improvement += 1
                
            # Report early stopping progress
            if consecutive_no_improvement > 0:
                print(f"No improvement for {consecutive_no_improvement}/{self.early_stopping} iterations")
        
        # End of refinement
        if consecutive_no_improvement >= self.early_stopping:
            print("\nEarly stopping: No improvement for several iterations.")
        else:
            print("\nReached maximum iterations.")
            
        print(f"Best results found at iteration {best_iteration}")
        print(f"Final ensemble score: {best_score:.4f}")
        
        # Use the best labels we found
        refined_labels = best_labels
        
        # Final evaluation if true labels provided (for reporting only)
        if y_true is not None:
            self._evaluate_clustering(refined_labels, y_true, "Final")
            
        # Print final metric weights if adaptive
        if self.adaptive_weights:
            print("\nFinal ensemble weights:")
            for metric, weight in self.ensemble_weights.items():
                print(f"  {metric}: {weight:.4f}")
                
        return refined_labels
    
    def _evaluate_clustering(self, labels, y_true, prefix=""):
        """
        Evaluate clustering results against ground truth (for reporting only).
        This does not affect the unsupervised refinement process.
        
        Parameters:
        -----------
        labels : array-like
            Cluster labels to evaluate
        y_true : array-like
            Ground truth labels
        prefix : str
            Prefix for printed messages (e.g., "Initial", "Final")
        """
        # Find the best label mapping for accuracy
        accuracy_0 = accuracy_score(y_true, labels)
        accuracy_1 = accuracy_score(y_true, 1-labels)
        
        if accuracy_1 > accuracy_0:
            print("[EVALUATION ONLY] Inverting labels for accuracy calculation...")
            labels_aligned = 1 - labels
            accuracy = accuracy_1
        else:
            labels_aligned = labels.copy()
            accuracy = accuracy_0
            
        # Calculate major and minor class F1 scores
        unique_classes, counts = np.unique(y_true, return_counts=True)
        if len(unique_classes) == 2:
            major_class = unique_classes[np.argmax(counts)]
            minor_class = unique_classes[np.argmin(counts)]
            
            # Calculate F1 scores
            f1_scores = f1_score(y_true, labels_aligned, average=None)
            major_idx = np.where(unique_classes == major_class)[0][0]
            minor_idx = np.where(unique_classes == minor_class)[0][0]
            f1_major = f1_scores[major_idx]
            f1_minor = f1_scores[minor_idx]
            
            print(f"[EVALUATION ONLY] {prefix} accuracy: {accuracy:.4f}")
            print(f"[EVALUATION ONLY] {prefix} F1 (major class {major_class}): {f1_major:.4f}")
            print(f"[EVALUATION ONLY] {prefix} F1 (minor class {minor_class}): {f1_minor:.4f}")
        else:
            print(f"[EVALUATION ONLY] {prefix} accuracy: {accuracy:.4f}")
            
        # Print confusion matrix
        cm = confusion_matrix(y_true, labels_aligned)
        print(f"\n[EVALUATION ONLY] {prefix} Confusion Matrix:")
        print(cm)
    
    def get_history(self):
        """Return the history of refinement metrics"""
        return pd.DataFrame(self.history)


# Example usage
if __name__ == "__main__":
    # Load the dataset
    print("Loading dataset...")
    data = pd.read_csv('reduced.csv')
    print(f"Dataset loaded successfully. Shape: {data.shape}")
    
    # Extract features and target
    if 'vital.status' in data.columns:
        X = data.drop('vital.status', axis=1)
        y = data['vital.status']
    else:
        print("Error: 'vital.status' column not found in the dataset.")
        exit(1)
    
    # Create binary mapping - exactly as in test.py
    unique_values = sorted(y.unique())
    binary_mapping = {val: i for i, val in enumerate(unique_values)}
    
    print("\nTarget distribution:")
    for val in unique_values:
        count = (y == val).sum()
        percentage = 100 * count / len(y)
        print(f"  {val}: {count} ({percentage:.2f}%)")
    
    print(f"Binary mapping: {unique_values[0]} -> 0, {unique_values[1]} -> 1")
    
    # Convert to binary values
    y_binary = y.map(binary_mapping)
    
    # Determine major and minor classes
    class_counts = y_binary.value_counts()
    major_class = class_counts.idxmax()
    major_ratio = class_counts[major_class] / len(y_binary)
    
    print(f"Major class is '{major_class}' with ratio: {major_ratio:.3f}")
    
    # Perform initial clustering with SilhouetteClassifier - EXACTLY as in test.py
    print("\nInitializing SilhouetteClassifier...")
    classifier = SilhouetteClassifier(n_neighbors=7)  # Use the same n_neighbors=7 as in test.py
    
    # Fit and predict - EXACTLY as in test.py
    initial_labels = classifier.fit_predict(X, major_ratio=major_ratio, major_class=major_class)
    
    # Calculate initial metrics (for evaluation only)
    accuracy = accuracy_score(y_binary, initial_labels)
    conf_matrix = confusion_matrix(y_binary, initial_labels)
    f1_scores = classifier.calculate_f1_scores(y_binary)
    
    print(f"\n[EVALUATION ONLY] Initial Accuracy: {accuracy:.3f}")
    print(f"[EVALUATION ONLY] Initial F1 score (major class {major_class}): {f1_scores[f'F1_class_{major_class}']:.3f}")
    print(f"[EVALUATION ONLY] Initial F1 score (minor class {1-major_class}): {f1_scores[f'F1_class_{1-major_class}']:.3f}")
    
    print("\n[EVALUATION ONLY] Initial Confusion Matrix:")
    print(conf_matrix)
    
    # Now perform refinement using these identical initial labels
    print("\nStarting refinement process...")
    refiner = EnsembleRefinement(
        max_iterations=30,  # Increased from 15 to 30
        threshold=0.0005,   # Decreased for more sensitive detection of improvements
        ensemble_weights={
            'silhouette': 1.0,
            'davies_bouldin': -0.7,
            'calinski_harabasz': 0.5,
            'connectivity': 0.8,
            'density_ratio': 0.6   # Added new density-based metric
        },
        n_neighbors=10,
        adaptive_weights=True,  # Enable adaptive weight adjustment
        batch_size=50,          # Process 50 points per batch
        early_stopping=5        # Stop after 5 iterations with no improvement
    )
    
    # Pass y_binary for evaluation only - refinement is purely unsupervised
    refined_labels = refiner.fit_refine(X, initial_labels=initial_labels, y_true=y_binary)
    
    # Print refinement history
    history_df = refiner.get_history()
    print("\nRefinement history:")
    print(history_df)
    
    # Compare initial and refined predictions
    changes = np.sum(initial_labels != refined_labels)
    print(f"\nPoints with changed labels: {changes} ({changes/len(X):.2%} of total)")
    
    # Save results to CSV
    results = pd.DataFrame({
        'true_label': y_binary,
        'initial_label': initial_labels,
        'refined_label': refined_labels,
        'changed': initial_labels != refined_labels
    })
    
    # Add feature values for reference
    for col in X.columns:
        results[col] = X[col].values
    
    results.to_csv('refinement_results.csv', index=False)
    print("\nResults saved to 'refinement_results.csv'")
    
    # Summarize improvement (for evaluation only)
    print("\n[EVALUATION ONLY] ===== SUMMARY =====")
    print(f"Initial accuracy: {accuracy:.4f}")
    final_accuracy = accuracy_score(y_binary, refined_labels)
    print(f"Final accuracy: {final_accuracy:.4f}")
    
    accuracy_improvement = final_accuracy - accuracy
    print(f"Accuracy improvement: {accuracy_improvement:.4f} ({accuracy_improvement*100:.2f}%)")
    
    # Calculate final F1 scores
    final_f1_scores = {}
    for cls in [0, 1]:
        final_f1_scores[f'F1_class_{cls}'] = f1_score(
            y_binary, refined_labels, pos_label=cls, average='binary')
    
    print(f"\nInitial F1 (major class {major_class}): {f1_scores[f'F1_class_{major_class}']:.4f}")
    print(f"Final F1 (major class {major_class}): {final_f1_scores[f'F1_class_{major_class}']:.4f}")
    f1_major_improvement = final_f1_scores[f'F1_class_{major_class}'] - f1_scores[f'F1_class_{major_class}']
    print(f"F1 major improvement: {f1_major_improvement:.4f} ({f1_major_improvement*100:.2f}%)")
    
    print(f"\nInitial F1 (minor class {1-major_class}): {f1_scores[f'F1_class_{1-major_class}']:.4f}")
    print(f"Final F1 (minor class {1-major_class}): {final_f1_scores[f'F1_class_{1-major_class}']:.4f}")
    f1_minor_improvement = final_f1_scores[f'F1_class_{1-major_class}'] - f1_scores[f'F1_class_{1-major_class}']
    print(f"F1 minor improvement: {f1_minor_improvement:.4f} ({f1_minor_improvement*100:.2f}%)")
