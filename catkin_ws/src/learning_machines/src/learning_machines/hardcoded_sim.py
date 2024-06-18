import cv2
import time
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from data_files import FIGRURES_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)

'''
THis is a backup of the original hardcoded stuff as reference

'''


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
        - normalized_distance_bottom: Proximity to the bottom of the image. Returns 1 and bottom, 0 at top.
        - normalized_distance_vertical: Proximity to the vertical center of the image. Returns 1 and the middle, 0 at top.

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
    vertical_center = image_width / 2

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

        # Calculate the distance from the vertical center
        distance_from_vertical_center = abs(box_center_x - vertical_center)

        # Normalize the distance to a value between 0 and 1, where 1 is at the center and 0 at the edges
        normalized_distance_vertical = 1 - (distance_from_vertical_center / (image_width / 2))

        return normalized_distance_bottom, normalized_distance_vertical
    else:
        print("No green box found.")
        return 0, 0

# # Example usage:
# image_path = "test_pic.jpg"
# normalized_distance_bottom, normalized_distance_vertical = process_image(image_path)
# if normalized_distance_bottom is not None and normalized_distance_vertical is not None:
#     print(f"Normalized distance from the bottom: {normalized_distance_bottom:.2f}")
#     print(f"Normalized distance from the vertical center: {normalized_distance_vertical:.2f}")


def test_emotions(rob: IRobobo):
    rob.set_emotion(Emotion.HAPPY)
    rob.talk("Hello")
    rob.play_emotion_sound(SoundEmotion.PURR)
    rob.set_led(LedId.FRONTCENTER, LedColor.GREEN)

def test_move_and_wheel_reset(rob):
    demo = True
    ir_data = []
    start_time = time.time()

    def record_ir_data():
        current_time = time.time() - start_time
        ir_values = rob.read_irs()
        print(f"IR Values: {ir_values}")  # Debugging: Print IR sensor values
        if len(ir_values) != 8:
            print(f"Unexpected number of IR sensor readings: {len(ir_values)}")
        ir_data.append([current_time] + ir_values)

    # Find wall and turn to side
    x = 0
    if demo:
        while rob.read_irs()[4] < 20 and x < 200:
            record_ir_data()
            x+=1
            rob.move(60, 60, 100)
        record_ir_data()
        rob.move_blocking(0, -100, 800)
        record_ir_data()
        rob.move_blocking(100, 100, 1000)
        record_ir_data()
    else:
        # Touch wall and back off
        while rob.read_irs()[4] < 200 and x < 100:
            record_ir_data()
            x+=1
            rob.move(100, 100, 100)
        record_ir_data()
        rob.move_blocking(-100, -100, 1000)
        record_ir_data()

    # Convert IR data to DataFrame
    if ir_data:  # Ensure ir_data is not empty
        expected_columns = 9  # 1 for time + 8 IR sensors
        if len(ir_data[0]) != expected_columns:
            print(f"Unexpected data structure: {len(ir_data[0])} columns found.")

        columns = ['Time'] + [f'IR_{i}' for i in range(8)]
        ir_df = pd.DataFrame(ir_data, columns=columns)

        # Plot the IR data  
        plt.figure(figsize=(10, 6))
        for i in range(1, len(columns)):
            sns.lineplot(x=ir_df['Time'], y=ir_df[columns[i]], label=columns[i])
        plt.title('IR Sensor Data Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('IR Sensor Reading')
        plt.legend()
        plt.grid(True)

        # Save the plot
        plt.savefig(FIGRURES_DIR / 'ir_sensor_data_plot.png')
        plt.close()

        # Plot the IR data
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=ir_df.drop(columns=['Time']))
        plt.title('IR Sensor Data Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('IR Sensor Reading')
        plt.grid(True)

        # Save the plot
        plt.savefig(FIGRURES_DIR / 'ir_sensor_data.png')
        plt.close()
    else:
        print("No IR data recorded.")

def test_sensors(rob: IRobobo):
    print("IRS data: ", rob.read_irs())
    image = rob.get_image_front()
    if isinstance(rob, SimulationRobobo):
        image = cv2.flip(image, 0)
    cv2.imwrite(str(FIGRURES_DIR / "photo.png"), image)
    print("Phone pan: ", rob.read_phone_pan())
    print("Phone tilt: ", rob.read_phone_tilt())
    print("Current acceleration: ", rob.read_accel())
    print("Current orientation: ", rob.read_orientation())

def test_phone_movement(rob: IRobobo):
    rob.set_phone_tilt_blocking(100, 100)
    time.sleep(1)

def test_sim(rob: SimulationRobobo):
    print(rob.get_sim_time())
    # print("is running?: ", rob.is_running())
    rob.stop_simulation()
    print(rob.get_sim_time())
    # print("is running?: ", rob.is_running())
    rob.play_simulation()
    print(rob.get_sim_time())
    print(rob.get_position())

def test_hardware(rob: HardwareRobobo):
    print("Phone battery level: ", rob.read_phone_battery())
    print("Robot battery level: ", rob.read_robot_battery())


def run_all_actions(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    # test_emotions(rob)
    test_phone_movement(rob)
    test_sensors(rob)
    fig_path = FIGRURES_DIR / "photo.png"
    b, m = process_image(fig_path)
    # test_move_and_wheel_reset(rob)
    # if isinstance(rob, SimulationRobobo):
    #     test_sim(rob)

    # if isinstance(rob, HardwareRobobo):
    #     test_hardware(rob)

    # test_phone_movement(rob)
    print(f"Normalized distance from the bottom: {b:.2f}")
    print(f"Normalized distance from the vertical center: {m:.2f}")

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()



