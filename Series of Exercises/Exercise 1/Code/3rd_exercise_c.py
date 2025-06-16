import numpy as np
import matplotlib.pyplot as plt

# Define decision boundaries as lines based on given equations
x_vals = np.linspace(0, 30, 1000)

# Boundary line between ω1 and ω2: 3x1 - 9x2 = 7.3744
boundary12_y = (3 * x_vals - 7.3744) / 9

# Boundary line between ω2 and ω3: x2 = 5.1277 (horizontal line)
boundary23_y = np.full_like(x_vals, 5.1277)

# Boundary line between ω1 and ω3: -3x1 + 15x2 = 23.3918
boundary13_y = (3 * x_vals + 23.3918) / 15

# Plotting
plt.figure(figsize=(10, 8))
# Plot decision boundaries
plt.plot(x_vals, boundary12_y, 'r', label='Boundary ω1-ω2')
plt.plot(x_vals, boundary23_y, 'g', label='Boundary ω2-ω3')
plt.plot(x_vals, boundary13_y, 'b', label='Boundary ω1-ω3')

# Additional plot settings
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Class Distributions with Decision Boundaries')
plt.legend()
plt.grid(True)
plt.xlim(0, 30)
plt.ylim(-5, 10)


# Plotting
plt.figure(figsize=(10, 8))
# Plot decision boundaries
plt.plot(x_vals, boundary12_y, 'm', label='Boundary ω1-ω2')
plt.plot(x_vals, boundary23_y, 'k', label='Boundary ω2-ω3')
plt.plot(x_vals, boundary13_y, 'c', label='Boundary ω1-ω3')

# # Fill areas based on classification regions
plt.fill_between(x_vals, boundary23_y, 10, color='lightcoral', alpha=0.3, label='Class ω3')   # Area above ω2-ω3 boundary
plt.fill_between(x_vals, boundary12_y, boundary23_y, where=(boundary12_y <= boundary23_y),
                 color='lightblue', alpha=0.3, label='Class ω2')  # Area between ω1-ω2 and ω2-ω3 boundaries
plt.fill_between(x_vals, -10, boundary12_y, color='lightgreen', alpha=0.3, label='Class ω1')  # Area below ω1-ω2 boundary

# Additional plot settings
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Class Distributions with Decision Boundaries')
plt.legend()
plt.grid(True)
plt.xlim(0, 17.84126)
plt.ylim(-5, 10)
plt.show()