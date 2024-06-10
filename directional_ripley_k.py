import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from read_binary import extract_coordinates_from_image

# Define the function for Directional Ripley's K
def directional_ripley_k(data, radii, azimuth_direction, tolerance_angle):
    n = len(data)
    print("1", n)
    k_values = np.zeros(len(radii))
    
    # Calculate the unit vector corresponding to the azimuth direction
    azimuth_rad = np.deg2rad(azimuth_direction)
    azimuth_rad_180plus = np.deg2rad(azimuth_direction+180)
    azimuth_vector = np.array([np.cos(azimuth_rad), np.sin(azimuth_rad)])
    azimuth_vector_180plus = np.array([np.cos(azimuth_rad_180plus), np.sin(azimuth_rad_180plus)])
    print(azimuth_direction, azimuth_rad, azimuth_rad_180plus)

    # Calculate the sector boundaries
    angle_tolerance = np.deg2rad(tolerance_angle)
    sector_start = azimuth_rad - angle_tolerance
    sector_end = azimuth_rad + angle_tolerance
    sector_start_180plus=azimuth_rad_180plus-angle_tolerance
    sector_end_180plus = azimuth_rad_180plus + angle_tolerance
    
    # Create a k-d tree for fast nearest-neighbor search
    kdtree = KDTree(data)
    
    for i in range(n):
        point = data[i]
        print(point)
        
        # Find points within the specified radius
        indices = kdtree.query_ball_point(point, radii[-1])
        print(indices)
        
        # Count points in the sector
        sector_count = 0
        for j in indices:
            dx, dy = data[j][0] - point[0], data[j][1] - point[1]
            # dx,dy=[point[0],data[j][0]],[point[1],data[j][1]]
            if dx!=0:
                tan_theta=dy/dx
                print("tan_theta ",tan_theta)
                
            else:
                tan_theta=0    
            angle = np.arctan(tan_theta)
            print("tan_theta: ",dx,dy)
            print()
            if (sector_start <= angle <= sector_end):
                sector_count += 1
        

        for r_idx, r in enumerate(radii):
            sector_count = 0
            for j in indices:
                dx, dy = data[j][0] - point[0], data[j][1] - point[1]
                angle = np.arctan2(dy, dx)
                 # Debug print statements
                print(f"Point: {data[j]}, Angle: {np.degrees(angle)}")
                if (sector_start <= angle <= sector_end):
                    sector_count += 1
                if (sector_start_180plus<=angle<=sector_end_180plus):
                    sector_count+=1
            
            k_values[r_idx] += sector_count
            
            print(f"Radius {r}: {sector_count} points in sector")

        # After the loop
        print("Final K values:", k_values)  

    
    # Apply edge correction
    for r_idx, r in enumerate(radii):  # Fixed the iteration here
        k_values[r_idx] /= (n * n)  # Normalize by the total number of points
    
    return k_values

# Sample data (replace with your actual point coordinates)
np.random.seed(42)
n_points = 500
data = extract_coordinates_from_image("random_sample.png")
print(len(data))

# Define the radii at which to compute K
radii = np.linspace(0, 0.5, 50)

# Specify the azimuth direction (in degrees) and tolerance angle (in degrees)
azimuth_direction = 45  # For example, 45 degrees
tolerance_angle = 15  # For example, 15 degrees

# Define the boundary of the study area (adjust as needed)
# boundary = [0, 1, 0, 1]

# Compute Directional Ripley's K values
directional_ripley_k(data, radii, azimuth_direction, tolerance_angle)

# Calculate 95% confidence intervals (example based on normal approximation)
# num_simulations = 10  # Number of Monte Carlo simulations
# conf_intervals = []

# for r in radii:
#     sim_values = []

#     for _ in range(num_simulations):
#         # Generate random points within the boundary
#         random_points = np.random.rand(n_points, 2)
        
#         # Calculate the Directional Ripley's K for the random points
#         sim_k_values = directional_ripley_k(random_points, [r], azimuth_direction, tolerance_angle, boundary)
        
#         sim_values.append(sim_k_values[0])
    
#     # Calculate the mean and standard deviation of the simulated K values
#     mean_sim = np.mean(sim_values)
#     std_sim = np.std(sim_values, ddof=1)
    
#     # Calculate the 95% confidence interval using the normal approximation
#     alpha = 0.05
#     z = 1.96  # For a 95% confidence interval
#     conf_interval = (mean_sim - z * std_sim / np.sqrt(num_simulations), mean_sim + z * std_sim / np.sqrt(num_simulations))
#     conf_intervals.append(conf_interval)

# Plot the results with confidence intervals
# print(k_values)
# plt.plot(radii, k_values, label="Observed K")

# # for r_idx, r in enumerate(radii):
#     # plt.fill_between([r],  alpha=0.3)

# plt.xlabel("Radius (r)")
# plt.ylabel("Directional Ripley's K Value")
# plt.title("Directional Ripley's K Function with Confidence Intervals")
# plt.legend()
# plt.show()






















