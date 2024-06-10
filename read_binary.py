import cv2


def extract_coordinates_from_image(image_path):
    # Read the image
    binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if binary_image is None:
        print("Error: Unable to read the image.")
        return 

    # Threshold the image to obtain a binary image
    _, binary_image = cv2.threshold(binary_image, 128, 255, cv2.THRESH_BINARY)

    # Get the coordinates of the white pixels (points)
    point_coordinates = []
    rows, cols = binary_image.shape

    for y in range(rows):
        for x in range(cols):
            if binary_image[y, x] == 255:
                point_coordinates.append((x, y))
        # print(point_coordinates)

    return point_coordinates

# print("hello")
# Example usage
# image_path = "random_sample.png"
# point_coordinates = extract_coordinates_from_image(image_path)



# print(point_coordinates)