import numpy as np
from scipy.stats import multivariate_normal
from scipy.integrate import dblquad

# Given parameters
mean_w2 = [4, 3]
cov_matrix = [[1, 0.5], [0.5, 1]]

# Define the distribution for p(x|w2)
rv_w2 = multivariate_normal(mean=mean_w2, cov=cov_matrix)

# Define the boundaries
boundary12 = lambda x1: (3 * x1 - 7.3744) / 9  # Boundary ω1-ω2
boundary23 = 5.1277  # Boundary ω2-ω3

# Misclassification region for ω3 (x2 > boundary23)
def integrand_w3(x2, x1):
    return rv_w2.pdf([x1, x2])

# Misclassification region for ω1 (x2 < boundary12)
def integrand_w1(x2, x1):
    return rv_w2.pdf([x1, x2])

# Integration limits
x1_min, x1_max = -np.inf, 17.84126

# Compute the probability for ω3 region
P_error_w3, _ = dblquad(
    integrand_w3,
    x1_min, x1_max,
    lambda x1: boundary23,  # Lower bound for x2
    lambda x1: np.inf       # Upper bound for x2
)

# Compute the probability for ω1 region
P_error_w1, _ = dblquad(
    integrand_w1,
    x1_min, x1_max,
    lambda x1: -np.inf,            # Lower bound for x2
    lambda x1: boundary12(x1)      # Upper bound for x2
)

# Total misclassification probability
P_misclassification = P_error_w1 + P_error_w3
print(f"Probability of Misclassification is: {P_misclassification}")
