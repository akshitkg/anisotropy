import numpy as np
import matplotlib.pyplot as plt

def directional_k_function(points, h, theta, delta_theta, domain_area):
    """
    Calculate the directional K-function for a given set of points.
    
    Args:
    points (ndarray): Array of point coordinates (N x 2).
    h (float): Distance parameter for the K-function.
    theta (float): Azimuth angle in degrees.
    delta_theta (float): Azimuth tolerance in degrees.
    domain_area (float): Area of the study domain.
    
    Returns:
    K_theta (float): Value of the directional K-function.
    """
    
    # Number of points
    N = len(points)
    
    # Convert angles from degrees to radians
    theta_rad = np.deg2rad(theta)
    delta_theta_rad = np.deg2rad(delta_theta)
    
    # Initialize sum for K-function
    K_theta = 0
    
    # Iterate through all pairs of points
    for i in range(N):
        for j in range(N):
            if i != j:
                # Distance between points i and j
                dist_ij = np.linalg.norm(points[i] - points[j])
                
                if dist_ij <= h:
                    # Angle between points i and j
                    angle_ij = np.arctan2(points[j][1] - points[i][1], points[j][0] - points[i][0])
                    
                    # Check if angle_ij falls within the azimuth tolerance
                    if (theta_rad - delta_theta_rad <= angle_ij <= theta_rad + delta_theta_rad) or \
                       (theta_rad - delta_theta_rad <= angle_ij + 2 * np.pi <= theta_rad + delta_theta_rad):
                        
                        # Apply edge correction factor (assuming domain boundary effects are negligible)
                        omega_i = 1  # Simplification: No edge correction
                        
                        # Increment K-function sum
                        K_theta += omega_i
    
    # Normalize K-function value
    K_theta *= domain_area / (N**2 * h * delta_theta_rad)
    
    return K_theta

def plot_directional_k_function(points, h_max, theta, delta_theta, domain_area, num_steps=100):
    """
    Plot the directional K-function for a range of distance parameters.
    
    Args:
    points (ndarray): Array of point coordinates (N x 2).
    h_max (float): Maximum distance parameter for the K-function.
    theta (float): Azimuth angle in degrees.
    delta_theta (float): Azimuth tolerance in degrees.
    domain_area (float): Area of the study domain.
    num_steps (int): Number of steps for the distance parameter.
    """
    
    h_values = np.linspace(0, h_max, num_steps)
    K_values = [directional_k_function(points, h, theta, delta_theta, domain_area) for h in h_values]
    
    plt.plot(h_values, K_values)
    plt.xlabel('Distance (h)')
    plt.ylabel('K(theta)')
    plt.title(f'Directional K-function (theta={theta}°, delta_theta={delta_theta}°)')
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Define some example points
    points = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6]
    ])
    
    # Set parameters
    h_max = 10
    theta = 45
    delta_theta = 10
    domain_area = 100  # Example domain area
    
    # Plot the directional K-function
    plot_directional_k_function(points, h_max, theta, delta_theta, domain_area)
