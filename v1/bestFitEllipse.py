import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Replace 'your_file.xlsx' with the actual path to your Excel file
file_path = 'data.xlsx'
data = pd.read_excel(file_path)

# Assuming your latitude and longitude columns are named 'Latitude' and 'Longitude'
latitude = data['Lat']
longitude = data['Long']

plt.figure(figsize=(8, 6))
plt.scatter(longitude, latitude, c='b', marker='.', label='Data Points')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Scatter Plot of Spatial Data')
plt.legend()
plt.grid(True)
# plt.show()


from scipy.optimize import minimize

def fit_ellipse(x, y):
    """Fits an ellipse to the data points (x, y) using least-squares optimization."""
    def ellipse_residuals(params, x, y):
        a, b, h, k = params
        return ((x - h) ** 2) / (a ** 2) + ((y - k) ** 2) / (b ** 2) - 1

    # Initial guess for ellipse parameters (semi-major and semi-minor axes, center coordinates)
    initial_guess = [1.0, 1.0, np.mean(x), np.mean(y)]

    # Perform the least-squares optimization to find the best fit ellipse parameters
    result = minimize(ellipse_residuals, initial_guess, args=(x, y))

    return result.x

# Fit the ellipse to the data
ellipse_params = fit_ellipse(longitude, latitude)

# Extract the semi-major and semi-minor axes and center coordinates of the fitted ellipse
semi_major_axis, semi_minor_axis, center_x, center_y = ellipse_params

# Create an ellipse with the fitted parameters
theta = np.linspace(0, 2 * np.pi, 100)
ellipse_x = center_x + semi_major_axis * np.cos(theta)
ellipse_y = center_y + semi_minor_axis * np.sin(theta)

# Plot the scatterplot with the fitted ellipse
plt.figure(figsize=(8, 6))
plt.scatter(longitude, latitude, c='b', marker='.', label='Data Points')
plt.plot(ellipse_x, ellipse_y, 'r', label='Best Fit Ellipse')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Scatter Plot with Best Fit Ellipse')
plt.legend()
plt.grid(True)
plt.show()
