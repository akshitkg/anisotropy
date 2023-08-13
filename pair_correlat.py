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

# Step 4: Calculate pair correlation function (Ripley's K-function)
def pair_correlation(r, r_max, n):
    area = np.pi * (r_max**2)
    n_pairs = n * (n - 1) / 2  # Number of possible point pairs
    
    k_values = []
    
    for r_val in r:
        condition = (d_matrix > 0) & (d_matrix <= r_val)
        num_pairs_in_disk = np.sum(condition) / 2  # Divide by 2 to avoid double counting
        k = num_pairs_in_disk / (n_pairs * area)
        k_values.append(k)
    
    return k_values

# Define distance bins for the pair correlation function
r_max = np.sqrt((x.max() - x.min())**2 + (y.max() - y.min())**2)
r = np.linspace(0, r_max, 100)

# Calculate pair correlation function values
n = len(x)  # Total number of points
k_values = pair_correlation(r, r_max, n)

# Step 5: Visualize the pair correlation function
plt.figure(figsize=(8, 6))
plt.plot(r, k_values, 'b', label='Pair Correlation Function (K)')
plt.xlabel('Distance (r)')
plt.ylabel('K(r)')
plt.title('Pair Correlation Function (Ripley\'s K-function)')
plt.legend()
plt.grid(True)
plt.show()
