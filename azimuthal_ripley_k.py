import numpy as np
from scipy.spatial import distance
import math
import matplotlib.pyplot as plt
from PIL import Image

def calculate_directional_ripley_k(image_path, azimuth_angle, tolerance_angle):
    # Load the image and extract coordinates of white pixels
    image = np.array(Image.open(image_path))
    white_pixel_coordinates = np.argwhere(image == 255)
    print(len(white_pixel_coordinates))
    
    # Calculate the distance matrix
    distances = distance.squareform(distance.pdist(white_pixel_coordinates))
    max_distance = np.max(distances)

    
    # Define the radii for Ripley-K function
    radii = np.linspace(0, max_distance, num=int(max_distance)+1)
    
    # Define the angles for sectors
    num_sectors = int(180 / (2 * tolerance_angle))
    
    # Initialize Ripley-K function
    ripley_k = np.zeros_like(radii)
    
    # Loop over each pixel as the center of the sector
    for center_pixel in white_pixel_coordinates:
        for i in range(num_sectors):
            # Define sector angles
            sector_start_angle = azimuth_angle + i * 2 * tolerance_angle
            sector_end_angle = sector_start_angle + 2 * tolerance_angle
            
            # Get pixels within the sector
            pixels_within_sector = []
            for pixel in white_pixel_coordinates:
                if pixel[0] == center_pixel[0] and pixel[1] == center_pixel[1]:
                    continue
                angle = math.degrees(math.atan2(pixel[0]-center_pixel[0], pixel[1]-center_pixel[1]))
                if angle < 0:
                    angle += 360
                if sector_start_angle <= angle < sector_end_angle:
                    pixels_within_sector.append(pixel)
            # Calculate distances from center pixel to pixels within the sector
            distances_within_sector = [distance.euclidean(center_pixel, pixel) for pixel in pixels_within_sector]
            
            # Update Ripley-K function
            for r in radii:
                k_value = sum(1 for d in distances_within_sector if r - d >= 0)
                ripley_k[int(r)] += k_value
    
    # Normalize Ripley-K function by the number of pixels and sectors
    ripley_k /= (len(white_pixel_coordinates) * num_sectors)
    
    return radii, ripley_k



def plot_ripley_k(radii, ripley_k):
    plt.plot(radii, ripley_k)
    plt.xlabel ('r')
    plt.ylabel('K(r)')
    plt.title('Directional Ripley-K Function')
    plt.grid(True)
    plt.show()

# Example usage
image_path = "binary_image.png"
azimuth_angle = 45  # in degrees
tolerance_angle = 15  # in degrees

radii, ripley_k = calculate_directional_ripley_k(image_path, azimuth_angle, tolerance_angle)
plot_ripley_k(radii, ripley_k)

