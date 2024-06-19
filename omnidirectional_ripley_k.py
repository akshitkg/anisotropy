import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Omnidirectional K-function with edge correction
def omnidirectional_k_function(points, h_values, domain_size):
    N = len(points)
    domain_area = domain_size[0] * domain_size[1]
    K_values = np.zeros_like(h_values, dtype=float)
    
    for idx, h in enumerate(h_values):
        neighbors_count = 0
        for i in range(N):
            distances = np.linalg.norm(points - points[i], axis=1)
            inside_circle = (distances <= h) & (distances > 0)
            weights = np.ones(N)
            for j in range(N):
                if inside_circle[j]:
                    weights[j] = calculate_weight(points[j], h, domain_size)
            neighbors_count += np.sum(inside_circle * weights)
        K_values[idx] = (domain_area / (N * N)) * neighbors_count
    
    return K_values

# Edge correction weight calculation
def calculate_weight(point, h, domain_size):
    x, y = point
    R = h
    area = np.pi * R**2
    
    if x - R < 0 or x + R > domain_size[0] or y - R < 0 or y + R > domain_size[1]:
        overlap_area = calculate_overlap_area(x, y, R, domain_size)
        return area / overlap_area
    return 1

def calculate_overlap_area(x, y, R, domain_size):
    # Calculate the area of the circle within the rectangular domain
    area = np.pi * R**2

    if x - R < 0: area -= R**2 * np.arccos((x - R) / R) - (x - R) * np.sqrt(R**2 - (x - R)**2)
    if x + R > domain_size[0]: area -= R**2 * np.arccos((domain_size[0] - x) / R) - (domain_size[0] - x) * np.sqrt(R**2 - (domain_size[0] - x)**2)
    if y - R < 0: area -= R**2 * np.arccos((y - R) / R) - (y - R) * np.sqrt(R**2 - (y - R)**2)
    if y + R > domain_size[1]: area -= R**2 * np.arccos((domain_size[1] - y) / R) - (domain_size[1] - y) * np.sqrt(R**2 - (domain_size[1] - y)**2)

    return area

# Monte Carlo simulations for confidence intervals
def monte_carlo_simulation(num_simulations, h_values, domain_size, num_points):
    K_simulations = np.zeros((num_simulations, len(h_values)))

    for i in range(num_simulations):
        random_points = np.random.rand(num_points, 2) * domain_size
        K_simulations[i, :] = omnidirectional_k_function(random_points, h_values, domain_size)
    
    K_mean = np.mean(K_simulations, axis=0)
    K_lower = np.percentile(K_simulations, 2.5, axis=0)
    K_upper = np.percentile(K_simulations, 97.5, axis=0)
    
    return K_mean, K_lower, K_upper

# Plotting the results
def plot_k_function(h_values, K_values, K_mean, K_lower, K_upper):
    plt.figure(figsize=(12, 6))
    plt.plot(h_values, K_values, label='K(h)', color='blue')
    plt.fill_between(h_values, K_lower, K_upper, color='gray', alpha=0.5, label='95% Confidence Interval')
    plt.plot(h_values, K_mean, linestyle='--', color='red', label='Mean K(h)')
    plt.xlabel('h')
    plt.ylabel('K(h)')
    plt.title('Omnidirectional K-function with Confidence Intervals')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("/results/omni_k_result.py")

# Example usage
domain_size = (10, 8)  # Rectangular study domain dimensions
num_points = 1000
h_values = np.linspace(0.1, 5, 50)
points = np.random.rand(num_points, 2) * domain_size

# Calculate K-function
K_values = omnidirectional_k_function(points, h_values, domain_size)

# Monte Carlo simulation
num_simulations = 100
K_mean, K_lower, K_upper = monte_carlo_simulation(num_simulations, h_values, domain_size, num_points)

# Plot the results
plot_k_function(h_values, K_values, K_mean, K_lower, K_upper)
