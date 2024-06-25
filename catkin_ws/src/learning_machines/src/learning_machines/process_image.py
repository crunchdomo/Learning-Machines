import numpy as np
import cv2

def process_image(image_path, colour='green'):
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

    if colour == 'green':
        # Define range for green color and create a mask
        lower_green = np.array([40, 100, 100])
        # lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

    if colour == 'red':
        if True: #isinstance(rob, SimulationRobobo):
            # Define range for blue color and create a mask
            lower_blue = np.array([110, 50, 50])
            upper_blue = np.array([130, 255, 255])
            
            # Create mask for blue color
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
        else:
            # Define range for red color and create a mask
            # Red wraps around the HSV color space, so we need two ranges
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            # Create masks for both ranges
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            
            # Combine the masks
            mask = cv2.bitwise_or(mask1, mask2)

        # Split the image into a 3x3 grid
    height, width = mask.shape
    grid_h, grid_w = height // 3, width // 3
    
    grid_percentages = []
    
    for i in range(3):
        for j in range(3):
            # Extract the grid
            grid = mask[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
            
            # Calculate the percentage of the grid filled with the color
            total_pixels = grid.size
            colored_pixels = np.count_nonzero(grid)
            percentage = colored_pixels / total_pixels
            
            grid_percentages.append(percentage)

            grid_number = i * 3 + j
            text = f"{grid_number}: {percentage:.2f}"
            text_x = j * grid_w + 10
            text_y = i * grid_h + 30
            cv2.putText(resized_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw grid lines on the image for debugging
    for i in range(1, 3):
        # Draw horizontal lines
        cv2.line(resized_image, (0, i * grid_h), (width, i * grid_h), (0, 255, 0), 2)
        # Draw vertical lines
        cv2.line(resized_image, (i * grid_w, 0), (i * grid_w, height), (0, 255, 0), 2)

    # Display the image with grid overlay
    cv2.imshow('Image with Grid', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return grid_percentages

# Example usage:
image_path = "test_photo.png"
color_percentages = process_image(image_path, colour='green')

print("Color percentages for each grid (0 to 1):")
for i, percentage in enumerate(color_percentages):
    print(f"Grid {i}: {percentage:.2f}")
