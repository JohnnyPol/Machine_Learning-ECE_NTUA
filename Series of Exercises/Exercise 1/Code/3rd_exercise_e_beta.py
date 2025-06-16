import numpy as np
from scipy.stats import multivariate_normal

# Δεδομένα της άσκησης
mu1 = np.array([3, -2])
mu2 = np.array([4, 3])
mu3 = np.array([6, 7])
sigma = np.array([[1, 0.5], [0.5, 1]])
priors = [0.2, 0.5, 0.3]

# Creating 10000 samples 
num_samples = 10000
samples = multivariate_normal.rvs(mean=mu2, cov=sigma, size=num_samples)

# Miclassification initialization
misclassified_count = 0

# For each sample we computer the posterio probability and classify them with Bayes 
for x in samples:
    # Computation of PDF for every distribution
    p_x_given_w1 = multivariate_normal.pdf(x, mean=mu1, cov=sigma) * priors[0]
    p_x_given_w2 = multivariate_normal.pdf(x, mean=mu2, cov=sigma) * priors[1]
    p_x_given_w3 = multivariate_normal.pdf(x, mean=mu3, cov=sigma) * priors[2]
    
    
    # Classification with Bayes classifier
    predicted_class = np.argmax([p_x_given_w1, p_x_given_w2, p_x_given_w3])
    
    # If the classifier didn't say that the sample was in class w2 (class 1) then we increase the misclassified count
    if predicted_class != 1:
        misclassified_count += 1

# Final Output
error_probability = misclassified_count / num_samples
print("Probability of misclassification of Class w2 is:", error_probability)
