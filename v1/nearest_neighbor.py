import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Step 1: Load the data and extract latitude and longitude
# Replace 'your_file.xlsx' with the actual path to your Excel file
file_path = 'data.xlsx'
data = pd.read_excel(file_path)

# Assuming your latitude and longitude columns are named 'Latitude' and 'Longitude'
latitude = data['Lat']
longitude = data['Long']

# Step 2: Convert latitude and longitude to pixel coordinates
# Choose an appropriate scale factor to convert degrees to pixels, adjust it based on your data
scale_factor = 100
x = (longitude - longitude.min()) * scale_factor
y = (latitude - latitude.min()) * scale_factor

# Step 3: Calculate distances between points
d_matrix = cdist(np.column_stack((x, y)), np.column_stack((x, y)))

# Step 4: Calculate nearest neighbor distances
def nearest_neighbor_distances(d_matrix):
    n = len(d_matrix)
    min_distances = []

    for i in range(n):
        sorted_dists = np.sort(d_matrix[i])
        min_dist = sorted_dists[1]  # Exclude the point itself
        min_distances.append(min_dist)

    return min_distances

# Calculate nearest neighbor distances
nearest_distances = nearest_neighbor_distances(d_matrix)

# Step 5: Visualize nearest neighbor distances (zoomed in)
plt.figure(figsize=(8, 6))
plt.hist(nearest_distances, bins=30, density=True, alpha=0.7, color='blue', label='Nearest Neighbor Distances')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.title('Nearest Neighbor Analysis')
plt.xlim(0, np.max(nearest_distances))
plt.legend()
plt.grid(True)
plt.show()
