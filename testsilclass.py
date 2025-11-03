import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from silclass import SilhouetteClassifier


def main():
    print("Loading dataset 'reduced.csv'...")
    try:
        data = pd.read_csv('reduced.csv')
        print(f"Dataset loaded successfully. Shape: {data.shape}")
        
        # Separate features and target using the correct target column
        if 'vital.status' in data.columns:
            X = data.drop('vital.status', axis=1)
            y = data['vital.status']
        else:
            print("Error: 'vital.status' column not found in the dataset.")
            return
        
        # Create binary mapping
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
        
        print("\nInitializing SilhouetteClassifier...")
        classifier = SilhouetteClassifier(n_neighbors=7)
        
        # Fit and predict
        predicted_labels = classifier.fit_predict(X, major_ratio=major_ratio, major_class=major_class)
        
        # Calculate metrics
        accuracy = accuracy_score(y_binary, predicted_labels)
        conf_matrix = confusion_matrix(y_binary, predicted_labels)
        f1_scores = classifier.calculate_f1_scores(y_binary)
        
        print(f"\nAccuracy: {accuracy:.3f}")
        print(f"F1 score (major class 0): {f1_scores['F1_class_0']:.3f}")
        print(f"F1 score (minor class 1): {f1_scores['F1_class_1']:.3f}")
        
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
