import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform

# Data
X = np.array([[1, 5], [3, 4], [0, 2], [5, 4], [2, 6], [3, 3], [2, 3], [4, 2]])

# D(X)
D = X

# Function to compute the proximity matrix using dc(x, y) = 1 - cos(theta_xy)
def proximity_matrix(X):
    n = X.shape[0]
    P = np.zeros((n, n))  # Initialize proximity matrix
    for i in range(n):
        for j in range(n):
            if i != j:
                # Compute cosine similarity and transform to proximity
                cos_theta = np.dot(X[i], X[j]) / (np.linalg.norm(X[i]) * np.linalg.norm(X[j]))
                P[i, j] = 1 - cos_theta
    return P

# P(X)
P = proximity_matrix(X)

# Convert matrices to pandas DataFrames for better formatting

P_df = pd.DataFrame(P, index=[f"Point {i+1}" for i in range(X.shape[0])],
                    columns=[f"Point {i+1}" for i in range(X.shape[0])])

# Print the matrices in a nicely formatted way
print("\nD(X) - Distance Matrix:")
print(D) 

print("\nP(X) - Proximity Matrix:")
print(P_df.round(3))  # Round to 3 decimal places for better readability


# Extract all values from P and their corresponding indices
values = P.flatten()
indices = np.unravel_index(np.argsort(-values), P.shape)  # Sort in descending order

# Pair sorted values with their indices
sorted_values = values[np.argsort(-values)]
sorted_indices = list(zip(indices[0], indices[1]))
# Display the sorted values and their indices
print("\nValues in P from highest to lowest:")
for value, (i, j) in zip(sorted_values, sorted_indices):
    print(f"Value: {value:.3f}, Index: ({i+1}, {j+1})")



# Convert the proximity matrix to a condensed form
P_condensed = squareform(P)

# Hierarchical Clustering with Single Linkage
linkage_matrix_single = linkage(P_condensed, method='single')

# Plot dendrogram for Single Linkage
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix_single, labels=np.arange(1, X.shape[0] + 1))
plt.title('Dendrogram (Single Linkage)')
plt.xlabel('Data Points')
plt.ylabel('Proximity')
plt.show()

# Hierarchical Clustering with Complete Linkage
linkage_matrix_complete = linkage(P_condensed, method='complete')

# Plot dendrogram for Complete Linkage
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix_complete, labels=np.arange(1, X.shape[0] + 1))
plt.title('Dendrogram (Complete Linkage)')
plt.xlabel('Data Points')
plt.ylabel('Proximity')
plt.show()