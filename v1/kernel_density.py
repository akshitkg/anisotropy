import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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

# Step 3: Create a KDE object
positions = np.vstack([x, y])
kde = gaussian_kde(positions)

# Step 4: Define the grid for evaluating KDE
x_grid, y_grid = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
grid_positions = np.vstack([x_grid.ravel(), y_grid.ravel()])

# Step 5: Evaluate KDE on the grid
density = kde(grid_positions)

# Step 6: Visualize KDE as a contour plot
plt.figure(figsize=(10, 8))
plt.contourf(x_grid, y_grid, density.reshape(x_grid.shape), cmap='viridis')
plt.colorbar(label='Density')
plt.scatter(x, y, color='red', marker='.', alpha=0.5)
plt.xlabel('Longitude (Pixel Coordinates)')
plt.ylabel('Latitude (Pixel Coordinates)')
plt.title('Kernel Density Estimation (KDE)')
plt.grid(True)
plt.show()
