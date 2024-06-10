import cv2
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt

# Define the function for Ripley's K
def ripley_k(data, radii, boundary):
    n = len(data)
    k_values = []
    
    for r in radii:
        count = 0
        
        for i in range(n):
            point = data[i]
            neighbors = []
            
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(np.array(data[j]) - np.array(point))
                    if dist <= r:
                        neighbors.append(dist)
            
            # Apply edge correction
            area_factor = len(neighbors) / n
            corrected_count = area_factor / (np.pi * r**2)
            
            count += corrected_count
        
        k_values.append(count)
    
    return k_values

# Load the image
image = cv2.imread("random_sample.png")  # Replace with your image path
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the image to isolate points (adjust the threshold as needed)
_, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# Find contours to extract point coordinates
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
points  = []
for contour in contours:
    for point in contour:
        x, y = point[0]
        points.append([x, y])

# Define the radii at which to compute K
radii = np.linspace(0, 100, 100)  # Adjust the range and step size as needed

# Define the boundary of the study area (adjust as needed)
boundary = [0, image.shape[1], 0, image.shape[0]]

# Compute Ripley's K values
k_values = ripley_k(points, radii, boundary)

# Plot the results
plt.plot(radii, k_values)
plt.xlabel("Radius (r)")
plt.ylabel("Ripley's K Value")
plt.title("Ripley's K Function on Image")
plt.show()
# plt.savefig("Ripley's K Function on Image.png")
plt.savefig("ripley k plot.png", dpi='figure', format=None)