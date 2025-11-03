import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class SilhouetteClassifier:
    """
    A class for classifying instances based on improved silhouette scores.
    Supports both binary and multi-class classification.
    """
    
    def __init__(self, n_neighbors=15, scale_data=True):
        """
        Initialize the SilhouetteClassifier.
        
        Parameters:
        -----------
        n_neighbors : int, default=15
            Number of neighbors to use for cohesion and separation calculations.
        scale_data : bool, default=True
            Whether to scale the data before computing silhouette scores.
        """
        self.instance_data = None
        self.n_neighbors = n_neighbors
        self.scale_data = scale_data
        self.scaler = StandardScaler() if scale_data else None
    
    def calculate_scores(self, X):
        """
        Calculate improved cohesion and separation scores for each instance.
        
        Parameters:
        -----------
        X : array-like
            The input data matrix.
        
        Returns:
        --------
        tuple
            Cohesion scores, separation scores, and combined silhouette scores
        """
        # Scale the data if requested
        if self.scale_data:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
            
        n_samples = X_scaled.shape[0]
        
        # Use NearestNeighbors for efficient distance calculation
        nn = NearestNeighbors(n_neighbors=min(n_samples, self.n_neighbors+1))
        nn.fit(X_scaled)
        
        # Get distances and indices of nearest neighbors for all points
        distances, indices = nn.kneighbors(X_scaled)
        
        # Remove self-distances (first column)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        
        # Initialize scores
        cohesion_scores = np.zeros(n_samples)
        separation_scores = np.zeros(n_samples)
        
        # Calculate cohesion as inverse of average distance to k nearest neighbors
        for i in range(n_samples):
            # Average distance to neighbors
            avg_distance = np.mean(distances[i])
            
            # Cohesion is inverse of average distance with appropriate scaling
            cohesion_scores[i] = 1.0 / (1.0 + avg_distance)
        
        # For separation, use distance to points beyond the immediate neighborhood
        # This is similar to the class-based separation in the successful code
        nn_all = NearestNeighbors(n_neighbors=n_samples)
        nn_all.fit(X_scaled)
        all_distances, all_indices = nn_all.kneighbors(X_scaled)
        
        for i in range(n_samples):
            # We want to measure separation as the distance to points outside the neighborhood
            # Get the average distance to points beyond the k nearest neighbors
            far_points_distances = all_distances[i][self.n_neighbors+1:]
            
            if len(far_points_distances) > 0:
                # Use the average of the first few distant points for better separation measure
                # This is more stable than just taking the minimum
                num_distant = min(len(far_points_distances), 10)
                separation_scores[i] = np.mean(far_points_distances[:num_distant])
            else:
                # If all points are considered neighbors, use the maximum distance
                separation_scores[i] = np.max(all_distances[i])
        
        # Normalize separation scores to [0,1]
        if n_samples > 1:
            max_separation = np.max(separation_scores)
            min_separation = np.min(separation_scores)
            if max_separation > min_separation:  # Avoid division by zero
                separation_scores = (separation_scores - min_separation) / (max_separation - min_separation)
        
        # Calculate final silhouette score using a weighted approach
        # This gives more weight to cohesion which proved more effective in the successful code
        cohesion_weight = 0.7  # Higher weight to cohesion
        separation_weight = 0.3  # Lower weight to separation
        
        silhouette_scores = (cohesion_scores ** cohesion_weight) * (separation_scores ** separation_weight)
        
        return cohesion_scores, separation_scores, silhouette_scores
    
    def fit_predict(self, X, y=None, major_ratio=0.5, major_class=0, class_ratios=None, class_labels=None):
        """
        Fit the model and predict labels.
        
        Parameters:
        -----------
        X : array-like
            The input data matrix.
        y : array-like or None, default=None
            Optional target labels. If provided, used to determine class ratios.
        major_ratio : float, default=0.5
            For binary classification, ratio of instances to assign to the major class.
        major_class : int, default=0
            For binary classification, the label to use for the major class.
        class_ratios : list or None, default=None
            List of ratios for each class. Must sum to 1.0. 
            If provided, overrides major_ratio.
        class_labels : list or None, default=None
            List of class labels to use. If None, sequential integers will be used.
            
        Returns:
        --------
        array
            Predicted labels.
        """
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
        
        n_samples = X_values.shape[0]
        
        # If y is provided, use it to determine class ratios
        if y is not None:
            # Count instances per class
            unique_values, counts = np.unique(y, return_counts=True)
            counts_dict = dict(zip(unique_values, counts))
            
            # Determine major class
            major_class = max(counts_dict, key=counts_dict.get)
            major_ratio = counts_dict[major_class] / len(y)
            
            # If class_ratios is not specified, derive from y
            if class_ratios is None:
                # For binary classification
                if len(unique_values) == 2:
                    minor_class = min(counts_dict, key=counts_dict.get)
                    minor_ratio = counts_dict[minor_class] / len(y)
                    
                    # Ensure proper order based on major_class
                    if major_class == 0:
                        class_ratios = [major_ratio, minor_ratio]
                        class_labels = [0, 1]
                    else:
                        class_ratios = [minor_ratio, major_ratio]
                        class_labels = [0, 1]
                
                # For multi-class
                else:
                    class_ratios = [counts_dict[label] / len(y) for label in sorted(unique_values)]
                    class_labels = sorted(unique_values)
        
        # Determine if we're using binary or multi-class mode
        if class_ratios is None:
            # Binary mode with major_ratio
            minor_class = 1 if major_class == 0 else 0
            class_ratios = [major_ratio, 1 - major_ratio]
            class_labels = [major_class, minor_class]
            n_classes = 2
        else:
            # Multi-class mode with class_ratios
            n_classes = len(class_ratios)
            # Validate that ratios sum to approximately 1
            if not np.isclose(sum(class_ratios), 1.0):
                raise ValueError("Class ratios must sum to 1.0")
            
            # Determine class labels if not provided
            if class_labels is None:
                class_labels = list(range(n_classes))
            elif len(class_labels) != n_classes:
                raise ValueError("Number of class labels must match number of class ratios")
        
        # Step 1: Give unique ID to all instances
        instance_ids = np.arange(n_samples)
        
        # Step 2: Calculate cohesion and separation scores
        cohesion_scores, separation_scores, final_scores = self.calculate_scores(X_values)
        
        # Step 3: Create a list of instances with their IDs and scores
        self.instance_data = pd.DataFrame({
            'instance_id': instance_ids,
            'cohesion_score': cohesion_scores,
            'separation_score': separation_scores,
            'silhouette_score': final_scores
        })
        
        # Sort the list by final score (descending) - this is key for the assignment
        sorted_data = self.instance_data.sort_values(by='silhouette_score', ascending=False).copy()
        
        # Initialize predicted labels array
        predicted_labels = np.zeros(n_samples, dtype=int)
        
        # Calculate cutoff points for each class
        cutoffs = [0]
        cumulative_ratio = 0
        for i in range(n_classes - 1):  # Only need n-1 cutoffs
            cumulative_ratio += class_ratios[i]
            cutoffs.append(int(np.round(cumulative_ratio * n_samples)))
        cutoffs.append(n_samples)  # Add final cutoff
        
        # Create a mapping from instance IDs to their positions in the sorted array
        sorted_instance_ids = sorted_data['instance_id'].values
        
        # Assign labels based on position in sorted array
        for i in range(n_classes):
            start_idx = cutoffs[i]
            end_idx = cutoffs[i+1]
            # Get instance IDs in this chunk
            chunk_instance_ids = sorted_instance_ids[start_idx:end_idx]
            # Assign the appropriate class label to these instances
            predicted_labels[chunk_instance_ids] = class_labels[i]
        
        # Store the labels in the instance data
        self.instance_data['predicted_label'] = predicted_labels
        
        return predicted_labels
    
    def get_instance_data(self):
        """
        Get all instance data with IDs, scores, and labels.
        
        Returns:
        --------
        DataFrame
            DataFrame with instance details.
        """
        if self.instance_data is None:
            raise ValueError("Model has not been fitted yet.")
        
        return self.instance_data
    
    def get_sorted_instances(self):
        """
        Get instances sorted by silhouette score.
        
        Returns:
        --------
        DataFrame
            DataFrame with instances sorted by silhouette score.
        """
        if self.instance_data is None:
            raise ValueError("Model has not been fitted yet.")
        
        return self.instance_data.sort_values(by='silhouette_score', ascending=False)
    
    def calculate_f1_scores(self, true_labels):
        """
        Calculate F1 scores for each class, including major and minor class F1 scores.
        
        Parameters:
        -----------
        true_labels : array-like
            True labels to compare against predictions.
            
        Returns:
        --------
        dict
            Dictionary containing F1 scores for each class, including major and minor class.
        """
        if self.instance_data is None:
            raise ValueError("Model has not been fitted yet.")
            
        pred_labels = self.instance_data['predicted_label'].values
        
        # Get unique labels
        unique_labels = np.unique(np.concatenate([true_labels, pred_labels]))
        results = {}
        
        # Count instances per class to identify major and minor classes
        class_counts = {}
        for label in unique_labels:
            class_counts[label] = np.sum(true_labels == label)
        
        # Identify major and minor classes
        if len(class_counts) >= 2:
            major_class = max(class_counts, key=class_counts.get)
            minor_class = min(class_counts, key=class_counts.get)
        else:
            # Only one class in the data
            major_class = list(class_counts.keys())[0]
            minor_class = major_class
        
        for label in unique_labels:
            # Calculate TP, FP, FN
            tp = np.sum((pred_labels == label) & (true_labels == label))
            fp = np.sum((pred_labels == label) & (true_labels != label))
            fn = np.sum((pred_labels != label) & (true_labels == label))
            
            # Calculate precision and recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # Calculate F1
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Store result
            results[f'F1_class_{label}'] = f1
            
            # Also store as major or minor class if applicable
            if label == major_class:
                results['F1_major'] = f1
            if label == minor_class:
                results['F1_minor'] = f1
        
        # Add accuracy
        accuracy = np.mean(pred_labels == true_labels)
        results['accuracy'] = accuracy
        
        return results
