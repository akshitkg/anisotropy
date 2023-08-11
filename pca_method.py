import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Step 2: Load the data and extract latitude and longitude
# Replace 'your_file.xlsx' with the actual path to your Excel file
file_path = 'Linement_Intersection_SRTM.xlsx'
data = pd.read_excel(file_path)


# Assuming your latitude and longitude columns are named 'Latitude' and 'Longitude'
latitude = data['Lat']
longitude = data['Long']
# print(latitude,longitude)
# Step 3: Convert latitude and longitude to pixel coordinates (similar to the previous method)
# Choose an appropriate scale factor to convert degrees to pixels, adjust it based on your data
scale_factor = 100
x = (latitude - latitude.min()) * scale_factor
y = (longitude - longitude.min()) * scale_factor

# Stack the coordinates into a single array for PCA
coordinates = np.column_stack((x, y))

# Step 4: Perform PCA to find the anisotropy direction
# Create a PCA object and fit it to the data
pca = PCA(n_components=2)
pca.fit(coordinates)

# Get the principal components and eigenvalues
eigenvectors = pca.components_
eigenvalues = pca.explained_variance_

# The anisotropy direction is the eigenvector corresponding to the largest eigenvalue
anisotropy_direction = eigenvectors[np.argmax(eigenvalues)]

# Step 5: Visualize the anisotropy direction as an ellipse
# Plot the scatterplot with the anisotropy direction
plt.figure(figsize=(8, 6))
plt.scatter(x, y, c='b', marker='.', label='Data Points')
plt.xlabel('Latitude (Pixel Coordinates)')
plt.ylabel('Longitude (Pixel Coordinates)')
plt.title('Scatter Plot with Anisotropy Direction')
plt.legend()
plt.grid(True)

# Draw the anisotropy direction as a line from the center
center_x, center_y = np.mean(x), np.mean(y)
plt.plot([center_x, center_x + anisotropy_direction[0]], [center_y, center_y + anisotropy_direction[1]], 'r', label='Anisotropy Direction')

# Draw the ellipse representation
ellipse = Ellipse((center_x, center_y), 2 * np.sqrt(eigenvalues[0]), 2 * np.sqrt(eigenvalues[1]), np.arctan2(*anisotropy_direction[::-1]) * 180 / np.pi, edgecolor='r', facecolor='none', label='Anisotropy Ellipse')
plt.gca().add_patch(ellipse)

plt.axis('equal')
plt.show()
