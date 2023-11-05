import numpy as np
import cv2
import random

# Create a blank white image
image_size = (512, 512)
binary_image = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)

# Generate random points
num_points = 1000
coordinates=[]
for _ in range(num_points):
    x = random.randint(0, image_size[1] - 1)
    y = random.randint(0, image_size[0] - 1)
    binary_image[y, x] = 255  # Set the pixel at (y, x) to white
    coordinates.append((y,x))

# Display the image (optional)
cv2.imshow("Random Points Image", binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the image to a file (optionalco)
cv2.imwrite("random_sample.png", binary_image)
print(len(coordinates))