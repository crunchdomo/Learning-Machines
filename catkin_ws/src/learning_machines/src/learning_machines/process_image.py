import numpy as np
import cv2
import matplotlib.pyplot as plt

def process_image(image_path, colour='green'):
    # Load the image
    image = cv2.imread(image_path)
    
    # Resize the image to speed up processing
    scale_percent = 50 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
    
    if colour == 'green':
        # Define range for green color and create a mask
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
    elif colour == 'red':
        # Define range for red color and create a mask
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
    
    height, width = mask.shape
    column_w = width // 3
    column_percentages = []
    
    # Create a copy of the resized image for drawing
    output_image = resized_image.copy()
    
    for j in range(3):
        # Extract the column
        column = mask[:, j*column_w:(j+1)*column_w]
        
        # Calculate the percentage of the column filled with the color
        total_pixels = column.size
        colored_pixels = np.count_nonzero(column)
        percentage = colored_pixels / total_pixels
        column_percentages.append(percentage)
        
        # Draw rectangle and text on the output image
        cv2.rectangle(output_image, (j*column_w, 0), ((j+1)*column_w, height), (0, 255, 0), 2)
        cv2.putText(output_image, f"{percentage:.2f}", (j*column_w + 10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return column_percentages, output_image, mask

# Example usage:
image_path = "test_photo.png"
color_percentages, output_image, mask = process_image(image_path, colour='green')

# Display results
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title("Image with Columns and Percentages")

plt.subplot(132)
plt.imshow(mask, cmap='gray')
plt.title("Mask")

plt.subplot(133)
plt.bar(range(3), color_percentages)
plt.title("Color Percentages")
plt.xlabel("Column")
plt.ylabel("Percentage")

plt.tight_layout()
plt.show()

print("Color percentages for each grid (0 to 1):")
for i, percentage in enumerate(color_percentages):
    print(f"Grid {i}: {percentage:.2f}")
