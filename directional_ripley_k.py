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

def sector_area(radius, angle):
    """
    Calculate the area of a sector.
    
    Parameters:
    radius (float): Radius of the sector.
    angle (float): Angle of the sector in radians.
    
    Returns:
    float: Area of the sector.
    """
    return 0.5 * radius ** 2 * angle

def boundary_intersection(x, y, radius, angle, width, height):
    """
    Find intersection points of the sector's radius with the study area boundaries.
    
    Parameters:
    x, y (float): Center of the sector.
    radius (float): Radius of the sector.
    angle (float): Angle of the sector in radians.
    width, height (float): Dimensions of the study area.
    
    Returns:
    list of tuples: Intersection points with the boundaries.
    """
    intersections = []
    # Right boundary (x = width)
    if angle >= 0 and angle <= np.pi:
        if x + radius * np.cos(angle) > width:
            y_int = y + (width - x) * np.tan(angle)
            if 0 <= y_int <= height:
                intersections.append((width, y_int))
    
    # Left boundary (x = 0)
    if angle >= np.pi or angle <= 0:
        if x - radius * np.cos(angle) < 0:
            y_int = y - x * np.tan(angle)
            if 0 <= y_int <= height:
                intersections.append((0, y_int))
    
    # Top boundary (y = 0)
    if angle <= 0.5 * np.pi or angle >= 1.5 * np.pi:
        if y - radius * np.sin(angle) < 0:
            x_int = x - y / np.tan(angle)
            if 0 <= x_int <= width:
                intersections.append((x_int, 0))
    
    # Bottom boundary (y = height)
    if angle >= 0.5 * np.pi and angle <= 1.5 * np.pi:
        if y + radius * np.sin(angle) > height:
            x_int = x + (height - y) / np.tan(angle)
            if 0 <= x_int <= width:
                intersections.append((x_int, height))
    
    return intersections

def intersection_area(x, y, radius, angle_start, angle_end, width, height):
    """
    Calculate the area of the sector that intersects with the study area boundaries.
    
    Parameters:
    x, y (float): Center of the sector.
    radius (float): Radius of the sector.
    angle_start, angle_end (float): Start and end angles of the sector in radians.
    width, height (float): Dimensions of the study area.
    
    Returns:
    float: Area of the sector that intersects with the study area boundaries.
    """
    def integrate_area(r, theta1, theta2):
        """
        Integrate the area of a circular segment.
        """
        return 0.5 * r ** 2 * (theta2 - theta1) - 0.5 * r ** 2 * (np.sin(theta2 - theta1))
    
    total_intersection_area = 0.0
    
    for angle in np.linspace(angle_start, angle_end, 100):
        intersections = boundary_intersection(x, y, radius, angle, width, height)
        for intersection in intersections:
            x_int, y_int = intersection
            theta1 = np.arctan2(y_int - y, x_int - x)
            if theta1 < 0:
                theta1 += 2 * np.pi
            theta2 = theta1 + 0.01  # Small increment to approximate the segment
            segment_area = integrate_area(radius, theta1, theta2)
            total_intersection_area += segment_area
    
    return total_intersection_area

def edge_correction(point, h, study_area, azimuth, tolerance):
    """
    Calculate edge correction weight for a point within a directional search window.
    
    Parameters:
    point (tuple): Coordinates of the point (x, y).
    h (float): Radius of the search window.
    study_area (tuple): Dimensions of the study area (width, height).
    azimuth (float): Azimuth angle in degrees.
    tolerance (float): Tolerance angle in degrees.
    
    Returns:
    float: Edge correction weight.
    """
    x, y = point
    width, height = study_area
    
    # Convert angles to radians
    azimuth_rad = np.radians(azimuth)
    tolerance_rad = np.radians(tolerance)
    
    # Total area of two sectors
    total_sector_angle = 2 * tolerance_rad
    total_area = 2 * sector_area(h, total_sector_angle)
    
    # Calculate the areas of the two sectors inside the study area
    sector1_area_inside = sector_area(h, total_sector_angle) - intersection_area(
        x, y, h, azimuth_rad - tolerance_rad, azimuth_rad + tolerance_rad, width, height)
    sector2_area_inside = sector_area(h, total_sector_angle) - intersection_area(
        x, y, h, azimuth_rad + np.pi - tolerance_rad, azimuth_rad + np.pi + tolerance_rad, width, height)
    
    area_inside = sector1_area_inside + sector2_area_inside
    
    # Edge correction weight
    edge_correction_weight = total_area / area_inside if area_inside > 0 else 0
    
    return edge_correction_weight

def directional_weight(theta_ij, azimuth, delta_theta):
    """
    Weight function for directional K-function.
    """
    return 1 if azimuth - delta_theta <= theta_ij < azimuth + delta_theta else 0

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
    n_points = len(points)
    area = study_area[0] * study_area[1]
    intensity = n_points / area
    
    k_values = np.zeros_like(h_values)
    
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                d_ij = np.linalg.norm(points[i] - points[j])
                theta_ij = float(np.degrees(np.arctan2(points[j, 1] - points[i, 1], points[j, 0] - points[i, 0])))
                print(type(theta_ij), theta_ij)
                if theta_ij < 0:
                    theta_ij += 360
                
                for k, h in enumerate(h_values):
                    if d_ij <= h:
                        w = edge_correction(points[i], h, study_area, azimuth, delta_theta)
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
                    theta_ij = float(np.degrees(np.arctan2(simulated_points[j, 1] - simulated_points[i], simulated_points[j, 0] - simulated_points[i]))[0])
                    print(type(theta_ij), theta_ij)
                    if theta_ij < 0:
                        theta_ij += 360
                    
                    for k, h in enumerate(h_values):
                        if d_ij <= h:
                            w = edge_correction(simulated_points[i], h, study_area, azimuth, delta_theta)
                            simulated_k_values[sim, k] += directional_weight(theta_ij, azimuth, delta_theta) * w
        
        simulated_k_values[sim] /= (intensity * n_points)
    
    conf_intervals = np.percentile(simulated_k_values, [2.5, 97.5], axis=0)
    
    return k_values, conf_intervals

# Example usage
image_path = 'random_sample.png'  # Replace with the path to your image file
points = extract_white_pixels(image_path)
study_area = (points[:, 0].max() + 1, points[:, 1].max() + 1)
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
plt.savefig("directional_result.png")
plt.show()

print("finished executing the file")