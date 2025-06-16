import numpy as np

# Data
points = np.array([[2, 3], [3, 2], [1, 2], [4, 5],
                   [5, 4], [3, 4], [6, 4], [6, 5]])

# Initialization of centroids
centroids = np.array([[3, 3], [4, 4]])

def euclidean_distance(point, centroid):
    """Calculation of the Euclidean distance between a point and a centroid"""
    return np.sqrt(np.sum((point - centroid) ** 2))

def assign_points_to_clusters(points, centroids):
    """Assign each point to the nearest centroid"""
    clusters = []
    for point in points:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster = np.argmin(distances)  # Returns the index of the nearest centroid
        clusters.append(cluster)
    return np.array(clusters)

def update_centroids(points, clusters, k):
    """Calculation of new centroids as the mean of the points in each cluster"""
    new_centroids = []
    for i in range(k):
        cluster_points = points[clusters == i]
        if len(cluster_points) > 0:  # If there are points in the cluster
            new_centroid = np.mean(cluster_points, axis=0)
        else:  # If there are no points, retain the old centroid
            new_centroid = centroids[i]
        new_centroids.append(new_centroid)
    return np.array(new_centroids)

# Execution of the algorithm
iteration = 0
k = len(centroids)
while True:
    print(f"\n=== Iteration {iteration} ===")
    # Calculation of distances and assignment
    clusters = assign_points_to_clusters(points, centroids)
    for i, point in enumerate(points):
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        print(
            f"Point: ({point[0]}, {point[1]}), Distances: {distances}, Assigned to Cluster: {clusters[i] + 1}"
        )
    
    # Calculation of new centroids
    new_centroids = update_centroids(points, clusters, k)
    print(f"New Centroids: {new_centroids}")
    
    # Convergence check
    if np.allclose(centroids, new_centroids):
        print("\nConvergence reached.")
        break
    centroids = new_centroids
    iteration += 1

# Final output
print("\n=== Final Output ===")
print("Final Centroids:")
for i, centroid in enumerate(centroids, 1):
    print(f"  Cluster {i}: {centroid}")
print("\nFinal Cluster Assignments:")
for i, cluster in enumerate(clusters, 1):
    print(f"  Point {i}: Assigned to Cluster {cluster + 1}")

