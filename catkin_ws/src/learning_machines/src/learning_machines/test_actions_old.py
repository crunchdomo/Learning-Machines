import cv2
import time
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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
    cv2.imwrite(str(FIGRURES_DIR / "photo.png"), image)
    print("Phone pan: ", rob.read_phone_pan())
    print("Phone tilt: ", rob.read_phone_tilt())
    print("Current acceleration: ", rob.read_accel())
    print("Current orientation: ", rob.read_orientation())


def test_phone_movement(rob: IRobobo):
    rob.set_phone_pan_blocking(20, 100)
    print("Phone pan after move to 20: ", rob.read_phone_pan())
    rob.set_phone_tilt_blocking(50, 100)
    print("Phone tilt after move to 50: ", rob.read_phone_tilt())


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
    test_sensors(rob)
    test_move_and_wheel_reset(rob)
    if isinstance(rob, SimulationRobobo):
        test_sim(rob)

    # if isinstance(rob, HardwareRobobo):
    #     test_hardware(rob)

    # test_phone_movement(rob)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
