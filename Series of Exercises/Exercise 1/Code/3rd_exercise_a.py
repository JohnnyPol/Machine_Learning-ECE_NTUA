import numpy as np
from scipy.stats import multivariate_normal

# Given data
mu1 = np.array([3, 2])
mu2 = np.array([4, 3])
mu3 = np.array([6, 7])
cov_matrix = np.array([[1, 0.5], [0.5, 1]])
priors = np.array([0.2, 0.5, 0.3])
x = np.array([3, 5])

# Calculate likelihoods
p_x_given_w1 = multivariate_normal.pdf(x, mean=mu1, cov=cov_matrix)
p_x_given_w2 = multivariate_normal.pdf(x, mean=mu2, cov=cov_matrix)
p_x_given_w3 = multivariate_normal.pdf(x, mean=mu3, cov=cov_matrix)

# Calculate posteriors using Bayes' rule
evidence = p_x_given_w1 * priors[0] + p_x_given_w2 * priors[1] + p_x_given_w3 * priors[2]
p_w1_given_x = (p_x_given_w1 * priors[0]) / evidence
p_w2_given_x = (p_x_given_w2 * priors[1]) / evidence
p_w3_given_x = (p_x_given_w3 * priors[2]) / evidence

print("Posterior probabilities:")
print(f"P(ω1 | x) = {p_w1_given_x:.4f}")
print(f"P(ω2 | x) = {p_w2_given_x:.4f}")
print(f"P(ω3 | x) = {p_w3_given_x:.4f}")
