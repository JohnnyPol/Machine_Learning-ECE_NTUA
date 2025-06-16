import numpy as np
import matplotlib.pyplot as plt

# Define parameters for each class
mean1 = np.array([3, -2])
mean2 = np.array([4, 3])
mean3 = np.array([6, 7])
cov_matrix = np.array([[1, 0.5], [0.5, 1]])  # Shared covariance matrix
priors = np.array([0.2, 0.5, 0.3])

# Generate 500 points for each class
np.random.seed(0)  # For reproducibility
p_x_given_w1 = np.random.multivariate_normal(mean1, cov_matrix, 500)
p_x_given_w2 = np.random.multivariate_normal(mean2, cov_matrix, 500)
p_x_given_w3 = np.random.multivariate_normal(mean3, cov_matrix, 500)

# Define decision boundaries as lines based on given equations
x_vals = np.linspace(-3, 20, 1000)

# Boundary line between ω1 and ω2: 3x1 - 9x2 = 7.3744
boundary12_y = (3 * x_vals - 7.3744) / 9

# Boundary line between ω2 and ω3: x2 = 5.1277 (horizontal line)
boundary23_y = np.full_like(x_vals, 5.1277)

# Boundary line between ω1 and ω3: -3x1 + 15x2 = 23.3918
boundary13_y = (3 * x_vals + 23.3918) / 15

# Plotting
plt.figure(figsize=(10, 8))

# Scatter plot for each class
plt.scatter(p_x_given_w1[:, 0], p_x_given_w1[:, 1], color='red', label='Class ω1', alpha=0.6)
plt.scatter(p_x_given_w2[:, 0], p_x_given_w2[:, 1], color='blue', label='Class ω2', alpha=0.6)
plt.scatter(p_x_given_w3[:, 0], p_x_given_w3[:, 1], color='green', label='Class ω3', alpha=0.6)

# Plot decision boundaries
plt.plot(x_vals, boundary12_y, 'm--', label='Boundary ω1-ω2')
plt.plot(x_vals, boundary23_y, 'k--', label='Boundary ω2-ω3')
plt.plot(x_vals, boundary13_y, 'c--', label='Boundary ω1-ω3')

# Plot centers of each class
plt.plot(mean1[0], mean1[1], 'ro', marker='x', markersize=10, label='Center ω1')
plt.plot(mean2[0], mean2[1], 'bo', marker='x', markersize=10, label='Center ω2')
plt.plot(mean3[0], mean3[1], 'go', marker='x', markersize=10, label='Center ω3')

# Additional plot settings
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Class Distributions with Decision Boundaries')
plt.legend()
plt.grid(True)
plt.xlim(-3, 17.84126)
plt.ylim(-5, 10)
plt.show()
