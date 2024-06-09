import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def extract_white_pixels(image_path):
    """
    Extract coordinates of white pixels from a binary image.
    
    Parameters:
    image_path (str): Path to the input image file.
    
    Returns:
    points (np.ndarray): Array of coordinates of white pixels.
    """
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    binary_image = np.array(image) > 128  # Convert to binary (white pixels)
    y_indices, x_indices = np.where(binary_image)
    points = np.column_stack((x_indices, y_indices))
    return points

def directional_k_function(points, study_area, h_values, azimuth, delta_theta, num_simulations=100):
    """
    Compute the directional Ripley K-function with edge correction and generate confidence intervals.
    
    Parameters:
    points (np.ndarray): Array of point coordinates.
    study_area (tuple): Dimensions of the study area (width, height).
    h_values (np.ndarray): Array of distance values at which to compute the K-function.
    azimuth (float): Azimuth angle of interest.
    delta_theta (float): Azimuth tolerance.
    num_simulations (int): Number of Monte Carlo simulations for confidence intervals.
    
    Returns:
    k_values (np.ndarray): Computed K-function values.
    conf_intervals (np.ndarray): Confidence intervals for the K-function.
    """
    
    def edge_correction(point, h, study_area):
        """
        Calculate edge correction weight for a point at a distance h.
        """
        x, y = point
        width, height = study_area
        r = h
        
        w_x = 1
        w_y = 1
        
        if x - r < 0: 
            w_x = np.pi * r**2 / (np.pi * r**2 - 0.5 * (r**2 * np.arccos((x - r) / r) - (x - r) * np.sqrt(r**2 - (x - r)**2)))
        
        if x + r > width: 
            w_x = np.pi * r**2 / (np.pi * r**2 - 0.5 * (r**2 * np.arccos((x + r - width) / r) - (x + r - width) * np.sqrt(r**2 - (x + r - width)**2)))
        
        if y - r < 0: 
            w_y = np.pi * r**2 / (np.pi * r**2 - 0.5 * (r**2 * np.arccos((y - r) / r) - (y - r) * np.sqrt(r**2 - (y - r)**2)))
        
        if y + r > height: 
            w_y = np.pi * r**2 / (np.pi * r**2 - 0.5 * (r**2 * np.arccos((y + r - height) / r) - (y + r - height) * np.sqrt(r**2 - (y + r - height)**2)))
        
        return w_x * w_y
    
    def directional_weight(theta_ij, azimuth, delta_theta):
        """
        Weight function for directional K-function.
        """
        return 1 if azimuth - delta_theta <= theta_ij < azimuth + delta_theta else 0
    
    n_points = len(points)
    area = study_area[0] * study_area[1]
    intensity = n_points / area
    
    k_values = np.zeros_like(h_values)
    
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                d_ij = np.linalg.norm(points[i] - points[j])
                theta_ij = np.degrees(np.arctan2(points[j, 1] - points[i, 1], points[j, 0] - points[i, 0]))
                if theta_ij < 0:
                    theta_ij += 360
                
                for k, h in enumerate(h_values):
                    if d_ij <= h:
                        w = edge_correction(points[i], h, study_area)
                        k_values[k] += directional_weight(theta_ij, azimuth, delta_theta) * w
    
    k_values /= (intensity * n_points)
    
    # Monte Carlo simulations for confidence intervals
    simulated_k_values = np.zeros((num_simulations, len(h_values)))
    for sim in range(num_simulations):
        simulated_points = np.column_stack((np.random.uniform(0, study_area[0], n_points), 
                                            np.random.uniform(0, study_area[1], n_points)))
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    d_ij = np.linalg.norm(simulated_points[i] - simulated_points[j])
                    theta_ij = np.degrees(np.arctan2(simulated_points[j, 1] - simulated_points[i], simulated_points[j, 0] - simulated_points[i]))
                    if theta_ij.any() < 0:
                        theta_ij += 360
                    
                    for k, h in enumerate(h_values):
                        if d_ij <= h:
                            w = edge_correction(simulated_points[i], h, study_area)
                            simulated_k_values[sim, k] += directional_weight(theta_ij, azimuth, delta_theta) * w
        
        simulated_k_values[sim] /= (intensity * n_points)
    
    conf_intervals = np.percentile(simulated_k_values, [2.5, 97.5], axis=0)
    
    return k_values, conf_intervals

# Example usage
image_path = 'random_sample.png'
points = extract_white_pixels(image_path)
study_area = (points[:, 0].max(), points[:, 1].max())
h_values = np.linspace(1, 50, 50)
azimuth = 45
delta_theta = 15
num_simulations = 100

k_values, conf_intervals = directional_k_function(points, study_area, h_values, azimuth, delta_theta, num_simulations)

plt.plot(h_values, k_values, label='Directional K-function')
plt.fill_between(h_values, conf_intervals[0], conf_intervals[1], color='gray', alpha=0.5, label='95% Confidence Interval')
plt.xlabel('Distance (h)')
plt.ylabel('K(h)')
plt.legend()
plt.show()
