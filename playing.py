import numpy as np
import cv2
import random

image_size = (512, 512)
binary_image = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)

binary_image[0,0] = 255

cv2.imshow("Random Points Image", binary_image)
cv2.waitKey(0)