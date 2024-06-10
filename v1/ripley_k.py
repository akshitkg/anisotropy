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

# Step 3: Calculate Ripley's K-function
def ripley_k(x, y, r):
    n = len(x)
    d_matrix = cdist(np.column_stack((x, y)), np.column_stack((x, y)))
    k_values = []

    for r_val in r:
        condition = d_matrix <= r_val
        num_points_in_disk = np.sum(condition, axis=1) - 1  # Exclude the point itself
        k = np.sum(num_points_in_disk) / n
        k_values.append(k)

    return k_values

# Define distance bins for the K-function
r = np.linspace(0, np.sqrt((x.max() - x.min())**2 + (y.max() - y.min())**2), 100)

# Calculate Ripley's K-function values
k_values = ripley_k(x, y, r)

# Convert K-function to L-function
L_values = np.sqrt(np.array(k_values) / np.pi)

# Step 4: Visualize Ripley's K-function
plt.figure(figsize=(8, 6))
plt.plot(r, L_values - r, 'b', label='L(r) - r')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Distance (r)')
plt.ylabel('L(r) - r')
plt.title("Ripley's K-function")
plt.legend()
plt.grid(True)
plt.show()
