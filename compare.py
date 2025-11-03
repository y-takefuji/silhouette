import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, MeanShift, OPTICS
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import umap

# -----------------------------
# Utility functions
# -----------------------------

def to_binary_labels(y):
    """
    Convert labels to {0,1}. If already numeric with two unique values, map to {0,1}.
    Otherwise, map unique sorted labels to {0,1}.
    """
    unique = pd.Series(y).unique()
    if len(unique) != 2:
        raise ValueError(f"Target must be binary. Found {len(unique)} unique labels: {unique}")
    if set(unique) == {0, 1}:
        return pd.Series(y).astype(int).values
    mapping = {u: i for i, u in enumerate(sorted(unique))}
    return pd.Series(y).map(mapping).astype(int).values

def map_clusters_to_classes(cluster_labels, true_binary_labels):
    """
    Map largest cluster to majority class, smallest cluster to minority class.
    For >2 clusters, all clusters except the largest are mapped to the minority class.
    """
    counts_true = pd.Series(true_binary_labels).value_counts()
    majority_class = counts_true.idxmax()
    minority_class = 1 - majority_class

    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_values(ascending=False)
    largest_cluster = cluster_sizes.index[0]

    mapping = {c: minority_class for c in cluster_sizes.index}
    mapping[largest_cluster] = majority_class

    y_pred = pd.Series(cluster_labels).map(mapping).values
    return y_pred, mapping, majority_class, minority_class

def clustering_metrics(features, true_labels, method):
    """
    Run a clustering method and compute accuracy, confusion matrix,
    F1 for majority and minority classes.
    """
    X = features

    if method == 'kmeans':
        model = KMeans(n_clusters=2, random_state=42, n_init='auto')
        cluster_labels = model.fit_predict(X)

    elif method == 'umap_kmeans':
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        X_emb = reducer.fit_transform(X)
        model = KMeans(n_clusters=2, random_state=42, n_init='auto')
        cluster_labels = model.fit_predict(X_emb)

    elif method == 'dbscan':
        model = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = model.fit_predict(X)
        # Fallback if DBSCAN is degenerate
        if len(set(cluster_labels)) <= 1:
            model_fallback = KMeans(n_clusters=2, random_state=42, n_init='auto')
            cluster_labels = model_fallback.fit_predict(X)

    elif method == 'meanshift':
        model = MeanShift()
        cluster_labels = model.fit_predict(X)
        if len(set(cluster_labels)) == 1:
            model_fallback = KMeans(n_clusters=2, random_state=42, n_init='auto')
            cluster_labels = model_fallback.fit_predict(X)

    elif method == 'optics':
        model = OPTICS(min_samples=5)
        cluster_labels = model.fit_predict(X)
        if len(set(cluster_labels)) <= 1:
            model_fallback = KMeans(n_clusters=2, random_state=42, n_init='auto')
            cluster_labels = model_fallback.fit_predict(X)
    else:
        raise ValueError(f"Unknown method: {method}")

    y_true = to_binary_labels(true_labels)
    y_pred, mapping, majority_class, minority_class = map_clusters_to_classes(cluster_labels, y_true)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # F1 scores for majority and minority classes
    f1_major = f1_score(y_true, y_pred, pos_label=majority_class)
    f1_minor = f1_score(y_true, y_pred, pos_label=minority_class)

    return acc, cm, f1_major, f1_minor, cluster_labels, y_pred, mapping, majority_class, minority_class

# -----------------------------
# Main script
# -----------------------------
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("reduced.csv")

    # Separate features and target
    if 'vital.status' not in df.columns:
        raise ValueError("Column 'vital.status' not found in reduced.csv")
    y = df['vital.status']
    X = df.drop(columns=['vital.status'])

    # Standardize features for distance-based clustering methods
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    methods = ['kmeans', 'umap_kmeans', 'dbscan', 'meanshift', 'optics']
    results = {}

    for m in methods:
        acc, cm, f1_major, f1_minor, cluster_labels, y_pred, mapping, majority_class, minority_class = clustering_metrics(X_scaled, y, m)
        results[m] = {
            'accuracy': acc,
            'mapping': mapping,
            'unique_clusters': sorted(list(set(cluster_labels))),
            'confusion_matrix': cm,
            'f1_major': f1_major,
            'f1_minor': f1_minor,
            'majority_class': majority_class,
            'minority_class': minority_class
        }
        print(f"Method: {m}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Cluster -> Class mapping: {mapping}")
        print(f"  Unique clusters found: {results[m]['unique_clusters']}")
        print("  Confusion Matrix (rows=true [0,1], cols=pred [0,1]):")
        print(results[m]['confusion_matrix'])
        print(f"  F1 (major={majority_class}): {f1_major:.4f}")
        print(f"  F1 (minor={minority_class}): {f1_minor:.4f}")
        print("")

    # Save predictions for each method
    out = pd.DataFrame({'vital.status_true': to_binary_labels(y)})
    for m in methods:
        acc, cm, f1_major, f1_minor, cluster_labels, y_pred, mapping, majority_class, minority_class = clustering_metrics(X_scaled, y, m)
        out[f'pred_{m}'] = y_pred
    out.to_csv("unsupervised_predictions.csv", index=False)
    print("Saved predictions to unsupervised_predictions.csv")
