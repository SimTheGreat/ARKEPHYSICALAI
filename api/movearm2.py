
import cv2
import numpy as np
import requests
import time
from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower
#import webrtcvad
import numpy as np
import queue
import threading
import time
import os


def send_action_to_vm(action):
    url = "http://192.168.89.143:5005/receive_action"
    try:
        r = requests.post(url, json={"action": action})
        print("✅ Sent:", action, "➜ VM responded:", r.json())
    except Exception as e:
        print("❌ Failed to send action:", e)
# Configure your robot with safety limitsn
config = SO101FollowerConfig(
    port="/dev/tty.usbmodem5AE70453521",
    id="jd_follower_arm",
    use_degrees=True,  # Use degree values instead of -100 to 100 range
)

import numpy as np
import requests
import time

def send_action_to_vm(action):
    url = "http://192.168.89.143:5005/receive_action"
    try:
        r = requests.post(url, json={"action": action})
        print("✅ Sent:", action, "➜ VM responded:", r.json())
    except Exception as e:
        print("❌ Failed to send action:", e)

send_action_to_vm("STEPFORWARD")
time.sleep(0.5)
send_action_to_vm("STEPFORWARD")
time.sleep(0.5)
send_action_to_vm("STEPFORWARD")
time.sleep(0.5)
send_action_to_vm("STEPFORWARD")
time.sleep(0.5)
send_action_to_vm("STEPFORWARD")
time.sleep(0.5)
send_action_to_vm("STEPFORWARD")
time.sleep(0.5)
send_action_to_vm("STEPFORWARD")
time.sleep(0.5)

send_action_to_vm("STEPRIGHT")

time.sleep(4.5)


# Initialize robot
robot = SO101Follower(config)
robot.connect()


print("Robot connected. Configuring for slow movement...\n")
r = requests.get("http://127.0.0.1:5000/distance")
print(r.json()["distance"])
# Set velocity for all motors
# Goal_Velocity range is typically 0-3000, lower = slower
# Acceleration range is 0-254, lower = slower acceleration
SLOW_VELOCITY = 200  # Moderate speed for full range movement
SLOW_ACCELERATION = 20  # Moderate acceleration

for motor in robot.bus.motors:
    print(f"Setting {motor} to slow speed...")
    robot.bus.write("Goal_Velocity", motor, SLOW_VELOCITY)
    robot.bus.write("Acceleration", motor, SLOW_ACCELERATION)

time.sleep(0.5)

# Read current position first
print("\nCurrent position:")
current_obs = robot.get_observation()
for motor, value in current_obs.items():
    if motor.endswith(".pos"):
        print(f"  {motor:20s}: {value:7.2f}")

# Define target position for each joint
#target_position = {
#    "shoulder_pan.pos": 0.00,     # Pan joint
#    "shoulder_lift.pos": -93.58,  # Lift joint
#    "elbow_flex.pos": 80.09,      # Elbow joint
#    "wrist_flex.pos": 38.07,      # Wrist flex
#    "wrist_roll.pos": 0.31,       # Wrist roll
#    "gripper.pos": 24.21,         # Gripper
#}

target_position1 = {
    "shoulder_pan.pos": 1.14,     # Pan joint

    "shoulder_lift.pos":  -102.24,  # Lift joint

    "elbow_flex.pos": 94.42,      # Elbow joint
    "wrist_flex.pos": 3.69,      # Wrist flex
    "wrist_roll.pos": 3.12,       # Wrist roll
    "gripper.pos": 0.00,         # Gripper
}

target_position2 = {
    "shoulder_pan.pos":  79.74,     # Pan joint

    "shoulder_lift.pos":  31.65,  # Lift joint

    "elbow_flex.pos":  34.46,      # Elbow joint
    "wrist_flex.pos":  -49.85,      # Wrist flex
    "wrist_roll.pos": -2.51,       # Wrist roll
    "gripper.pos": 60.94,         # Gripper
}
target_position3 = {
    "shoulder_pan.pos":  84.92,     # Pan joint

    "shoulder_lift.pos":  17.67,  # Lift joint

    "elbow_flex.pos":  -19.96,      # Elbow joint
    "wrist_flex.pos":  5.36,      # Wrist flex
    "wrist_roll.pos": -2.07,       # Wrist roll
    "gripper.pos": 16,         # Gripper
}
# Show what will change
target_position4 = {
    "shoulder_pan.pos": 1.14,     # Pan joint

    "shoulder_lift.pos":  -102.24,  # Lift joint

    "elbow_flex.pos": 94.42,      # Elbow joint
    "wrist_flex.pos": 3.69,      # Wrist flex
    "wrist_roll.pos": 3.12,       # Wrist roll
    "gripper.pos": 16,         # Gripper
}

print("\nTarget position and differences:")
for key, target_val in target_position1.items():
    current_val = current_obs[key]
    diff = target_val - current_val
    print(f"  {key:20s}: {current_val:7.2f} → {target_val:7.2f} (Δ {diff:+7.2f}°)")

# Safety confirmation

print("\nMoving to target position SLOWLY...")


sent_action = robot.send_action(target_position1)
print(f"Sent action: {sent_action}")

time.sleep(6.0)
sent_action = robot.send_action(target_position2)
print(f"Sent action: {sent_action}")
time.sleep(6.0)
r = requests.get("http://127.0.0.1:5000/distance")
print(r.json()["distance"])

current_obs = robot.get_observation()

print("\nCurrent position:")
current_obs = robot.get_observation()
for motor, value in current_obs.items():
    if motor.endswith(".pos"):
        print(f"  {motor:20s}: {value:7.2f}")

current_obs = robot.get_observation()

shoulder_pan = current_obs["shoulder_pan.pos"]
print(shoulder_pan)
time.sleep(6.0)


if shoulder_pan is not None:
    print(f"\nShoulder Pan: {shoulder_pan:.2f}")
else:
    print("Shoulder pan value not found.")
print(r.json()["distance"])
current_obs = robot.get_observation()

shoulder_pan = current_obs["shoulder_pan.pos"]
print(shoulder_pan)
while r.json()["distance"] > 50:
    print("\nDistance is greater than 40, moving shoulder pan...")
    # 1️⃣ Get latest joint position FIRST
    current_obs = robot.get_observation()
    shoulder_pan = current_obs["shoulder_pan.pos"]

    # 2️⃣ Compute new target
    target_positionimprov = {
        "shoulder_pan.pos": shoulder_pan + 3,
    "shoulder_lift.pos":   77.98,  # Lift joint

    "elbow_flex.pos":   -49.85,      # Elbow joint
    "wrist_flex.pos":  -20.92,      # Wrist flex
    "wrist_roll.pos": -3.65,       # Wrist roll
    "gripper.pos": 61.30, 
    }

    # 3️⃣ Send action
    robot.send_action(target_positionimprov)

    print("Moving shoulder pan")

    time.sleep(4.0)

    # 4️⃣ Update distance AFTER movement
    r = requests.get("http://127.0.0.1:5000/distance")
    print("Distance:", r.json()["distance"])
    
while r.json()["distance"] < -50:
    print("Too far left, moving right...")
    # 1️⃣ Get latest joint position FIRST
    current_obs = robot.get_observation()
    shoulder_pan = current_obs["shoulder_pan.pos"]

    # 2️⃣ Compute new target
    target_positionimprov = {
        "shoulder_pan.pos": shoulder_pan - 3,
    "shoulder_lift.pos":   77.98,  # Lift joint

    "elbow_flex.pos":   -49.85,      # Elbow joint
    "wrist_flex.pos":  -20.92,      # Wrist flex
    "wrist_roll.pos": -3.65,       # Wrist roll
    "gripper.pos": 61.30, 
    }

    # 3️⃣ Send action
    robot.send_action(target_positionimprov)

    print("Moving shoulder pan")

    time.sleep(1.0)

    # 4️⃣ Update distance AFTER movement
    r = requests.get("http://127.0.0.1:5000/distance")
    print("Distance:", r.json()["distance"])


print("\nMoving to target position SLOWLY...")

    
time.sleep(6.0)   




print("\nMoving to target position SLOWLY...")


# Send the action to move the robot
#sent_action = robot.send_action(target_position3)
print(f"Sent action: ")


print("\nWaiting for movement to complete...")




current_obs = robot.get_observation()
shoulder_pan = current_obs["shoulder_pan.pos"]
shoulder_lift = current_obs["shoulder_lift.pos"]
elbow_flex = current_obs["elbow_flex.pos"]
wrist_flex = current_obs["wrist_flex.pos"]
wrist_roll = current_obs["wrist_roll.pos"]
gripper = current_obs["gripper.pos"]

target_positiondown = {
        "shoulder_pan.pos": shoulder_pan,
        "shoulder_lift.pos": shoulder_lift,
        "elbow_flex.pos": elbow_flex,
        "wrist_flex.pos": wrist_flex,
        "wrist_roll.pos": wrist_roll,
        "gripper.pos": 16,
    }

sent_action = robot.send_action(target_positiondown)
time.sleep(5.0)


print(f"Sent action: {sent_action}")





sent_action = robot.send_action(target_position3)
print(f"Sent action: {sent_action}")

time.sleep(5.0)
sent_action = robot.send_action(target_position4)



# print("\nWaitfor movement to complete...")
# time.sleep(5.0)
# sent_action = robot.send_action(target_position4)
# print(f"Sent action: {sent_action}")
# print("\nWaiting for movement to complete...")
# time.sleep(5.0)
# sent_action = robot.send_action(target_position3)
# print(f"Sent action: {sent_action}")
# print("\nWaiting for movement to complete...")
# time.sleep(5.0)
# sent_action = robot.send_action(target_position2)
# print(f"Sent action: {sent_action}")
# time.sleep(5.0)
# sent_action = robot.send_action(target_position1)
# print(f"Sent action: {sent_action}")
# time.sleep(50.0)
# robot.disconnect()
# Disconnect
time.sleep(15)
send_action_to_vm("STEPLEFT")
time.sleep(0.5)
send_action_to_vm("STEPBACK")
time.sleep(0.5)
send_action_to_vm("STEPBACK")
time.sleep(0.5)
send_action_to_vm("STEPBACK")
time.sleep(0.5)
send_action_to_vm("STEPBACK")
time.sleep(0.5)
send_action_to_vm("STEPBACK")

time.sleep(0.5)

robot.disconnect()
# print("\nRobot disconnected.")
time.sleep(800.5)
