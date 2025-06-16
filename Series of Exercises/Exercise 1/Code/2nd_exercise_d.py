import numpy as np
import matplotlib.pyplot as plt

# Define the function based on the given equation
def contour_function(x, y):
    term1 = ((x - 5) / 3) ** 2
    term2 = -1.2 * ((x - 5) / 3) * ((y - 10) / 4)
    term3 = ((y - 10) / 4) ** 2
    return term1 + term2 + term3

# Set up the grid of x and y values
x_vals = np.linspace(-5, 15, 400)
y_vals = np.linspace(0, 20, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Calculate Z values on the grid
Z = contour_function(X, Y)

# Define the contour level based on the given constant
contour_level = 0.31111111

# Plotting
plt.figure(figsize=(8, 6))
contour = plt.contour(X, Y, Z, levels=[contour_level], colors='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title(r'Contour plot for $\left(\frac{x-5}{3}\right)^{2}-1.2\left(\frac{x-5}{3}\cdot\frac{y-10}{4}\right)+\left(\frac{y-10}{4}\right)^{2}=0.3111$')
plt.xlim([2, 8])
plt.ylim([6.5,13.5])
plt.show()
