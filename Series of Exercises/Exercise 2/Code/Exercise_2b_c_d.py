import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Load the dataset
data = pd.read_csv('iris.csv')
data = data.drop(columns=['Id'])

# Encode Species to numerical values for comparison
label = LabelEncoder()
data['SpeciesEncoded'] = label.fit_transform(data['Species'])

# K-means algorithm implementation


def k_means(X, k, epsilon=1e-5, max_iterations=100):
    # Random initialization of centroids
    np.random.seed(42)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for iteration in range(max_iterations):
        # Assign points to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.array(
            [X[clusters == i].mean(axis=0) for i in range(k)])
        # Convergence criterion
        if np.linalg.norm(new_centroids - centroids) < epsilon:
            print("\nConvergence reached!")
            break

        centroids = new_centroids

    return clusters, centroids

# Function to remap clusters to match true labels
def remap_clusters(true_labels, clusters):
    # Create a mapping from clusters to true labels based on majority vote
    mapping = {}
    for cluster_id in np.unique(clusters):
        # Find true labels corresponding to points in this cluster
        true_labels_in_cluster = true_labels[clusters == cluster_id]
        # Get the most common true label for this cluster
        most_common_label = Counter(
            true_labels_in_cluster).most_common(1)[0][0]
        mapping[cluster_id] = most_common_label

    # Remap the clusters
    remapped_clusters = np.array([mapping[cluster] for cluster in clusters])
    return remapped_clusters


# (b) Apply k-means with all features
X_full = data[['SepalLengthCm', 'SepalWidthCm',
               'PetalLengthCm', 'PetalWidthCm']].values
clusters_full, centroids_full = k_means(X_full, k=3)

# Map cluster labels to the true labels for comparison
true_labels = data['SpeciesEncoded'].values
clusters_full = remap_clusters(true_labels, clusters_full)

conf_matrix_full = confusion_matrix(true_labels, clusters_full)
accuracy_full = accuracy_score(true_labels, clusters_full)


print("\n=== (b) Full Feature Set ===")
print("Confusion Matrix:")
print(conf_matrix_full)
print(f"\nAccuracy (Full Features): {accuracy_full * 100:.2f}%")

# (c) Apply k-means with only two features
X_two = data[['PetalLengthCm', 'PetalWidthCm']].values
clusters_two, centroids_two = k_means(X_two, k=3)
clusters_two = remap_clusters(true_labels, clusters_two)
conf_matrix_two = confusion_matrix(true_labels, clusters_two)
accuracy_two = accuracy_score(true_labels, clusters_two)


print("\n=== (c) Two Features ===")
print("Confusion Matrix:")
print(conf_matrix_two)
print(f"Accuracy: {accuracy_two * 100:.2f}%")

# (d) Compare results
print("\n=== (d) Comparison ===")
print(f"Accuracy with Full Features: {accuracy_full * 100:.2f}%")
print(f"Accuracy with Two Features: {accuracy_two * 100:.2f}%")
print("Observations:")
if accuracy_full > accuracy_two:
    print("- Using all features results in better clustering accuracy.")
else:
    print("- Using only two features (Petal Length and Petal Width) is sufficient for good clustering accuracy.")


# Plotting for (c)
plt.figure(figsize=(8, 6))
for i in range(3):
    cluster_points = X_two[clusters_two == i]
    plt.scatter(cluster_points[:, 0],
                cluster_points[:, 1], label=f'Cluster {i+1}')
plt.scatter(centroids_two[:, 0], centroids_two[:, 1],
            color='black', marker='x', s=100, label='Centroids')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('(c) K-means Clustering on Two Features')
plt.legend()
plt.show()
