import numpy as np
import cv2

def process_image(image_path):
    """
    Parameters
    ----------
    image_path : str
        The path to the image file.

    Returns
    -------
    tuple
        A tuple containing two float values:
        - normalized_distance_bottom: Closeness to bottom of the image. Returns 1 at bottom, 0 at top.
        - normalized_distance_side: Closeness to right of the image. Returns 1 at the right, 0 at the left.

    - Returns 0 in both cases if no box is found.
    """
        
    # Load the image
    image = cv2.imread(image_path)

    # Resize the image to speed up processing
    scale_percent = 50  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

    # Define range for green color and create a mask
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get image dimensions
    image_height, image_width = resized_image.shape[:2]

    closest_contour = None
    min_distance_from_bottom = float('inf')

    for contour in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        box_bottom = y + h

        # Calculate the distance to the bottom of the image
        distance_from_bottom = image_height - box_bottom

        if distance_from_bottom < min_distance_from_bottom:
            min_distance_from_bottom = distance_from_bottom
            closest_contour = contour

    if closest_contour is not None:
        # Get the bounding box of the closest contour
        x, y, w, h = cv2.boundingRect(closest_contour)
        box_bottom = y + h
        box_center_x = x + w / 2

        # Calculate the distance from the bottom of the image
        distance_from_bottom = image_height - box_bottom

        # Normalize the distance to a value between 0 and 1
        normalized_distance_bottom = 1 - (distance_from_bottom / image_height)

        # Normalize the horizontal position to a value between 0 and 1
        normalized_distance_right = box_center_x / image_width

        return normalized_distance_bottom, normalized_distance_right
    else:
        return 0, 0

# Example usage:
# image_path = "test_pic.jpg"
# normalized_distance_bottom, normalized_distance_vertical = process_image(image_path)
# if normalized_distance_bottom is not None and normalized_distance_vertical is not None:
#     print(f"Normalized distance from the bottom: {normalized_distance_bottom:.2f}")
#     print(f"Normalized distance from the vertical center: {normalized_distance_vertical:.2f}")
