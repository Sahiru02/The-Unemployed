import cv2
import numpy as np
from pymavlink import mavutil
import time
from pupil_apriltags import Detector

# --- CONFIGURATION ---
CONNECTION_STRING = 'udpin:0.0.0.0:14550' 
VIDEO_SOURCE = "udp://127.0.0.1:5599"
TARGET_ALTITUDE = 1.5
FORWARD_SPEED = 0.5
STEER_GAIN = 0.006 

# --- INITIALIZATION ---
print(f"Connecting to SITL on {CONNECTION_STRING}...")
master = mavutil.mavlink_connection(CONNECTION_STRING)
master.wait_heartbeat()
print(f"Heartbeat received! (System {master.target_system})")

at_detector = Detector(families='tag36h11')

def send_heartbeat():
    master.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)

def fix_prearm_errors():
    """Sets required parameters to bypass ACRO_BAL errors"""
    print("Fixing PreArm ACRO parameters...")
    # Set ACRO_BAL_ROLL and ACRO_BAL_PITCH to 1.0 (default) to satisfy PreArm
    params = [('ACRO_BAL_ROLL', 1.0), ('ACRO_BAL_PITCH', 1.0)]
    
    for p_name, p_val in params:
        master.mav.param_set_send(
            master.target_system, master.target_component,
            p_name.encode('utf-8'), p_val, mavutil.mavlink.MAV_PARAM_TYPE_REAL32
        )
        time.sleep(0.2)
    print("Parameters updated.")

def wait_for_ekf():
    print("Waiting for EKF/GPS fix...")
    while True:
        send_heartbeat()
        msg = master.recv_match(type='EKF_STATUS_REPORT', blocking=True, timeout=0.1)
        if msg and (msg.flags & 1 and msg.flags & 2):
            print("EKF Ready!")
            break
        time.sleep(0.5)

def arm_and_takeoff(target_alt):
    # 1. Fix the specific error you saw
    fix_prearm_errors()
    
    # 2. Wait for physics/GPS to settle
    wait_for_ekf()
    
    print("Setting mode to GUIDED...")
    master.set_mode('GUIDED')
    time.sleep(1)
    
    print("Arming...")
    # Forcing arming command
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 21196, 0, 0, 0, 0, 0)
    
    master.motors_armed_wait()
    print("Armed! Taking off...")
    
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, target_alt)

    while True:
        send_heartbeat()
        msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=0.1)
        if msg:
            alt = msg.relative_alt / 1000.0
            if alt >= target_alt * 0.95:
                print(f"Reached altitude: {alt:.2f}m")
                break
        time.sleep(0.2)

def move_velocity(vx, vy):
    master.mav.set_position_target_local_ned_send(
        0, master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000111111000111, 
        0, 0, 0, vx, vy, 0, 0, 0, 0, 0, 0)

# --- MAIN LOOP ---
try:
    arm_and_takeoff(TARGET_ALTITUDE)
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    phase = "LINE_1"

    while True:
        send_heartbeat()
        ret, frame = cap.read()
        if not ret:
            move_velocity(0, 0)
            continue

        # Simple Yellow Detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
        
        h, w = mask.shape
        roi = mask[int(h*0.7):h, :]
        M = cv2.moments(roi)

        # AprilTag Detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = at_detector.detect(gray)
        
        if tags:
            tag_id = tags[0].tag_id
            if phase == "LINE_1":
                print(f"Tag {tag_id} found - Turning...")
                master.mav.command_long_send(
                    master.target_system, master.target_component,
                    mavutil.mavlink.MAV_CMD_CONDITION_YAW, 0, 90, 0, 1, 1, 0, 0, 0)
                time.sleep(2)
                phase = "LINE_2"
            else:
                master.set_mode('LAND')
                break

        # Steering
        if M["m00"] > 500:
            cx = int(M["m10"] / M["m00"])
            err = cx - (w / 2)
            move_velocity(FORWARD_SPEED, err * STEER_GAIN)
        else:
            move_velocity(0, 0)

except Exception as e:
    print(f"Error: {e}")
finally:
    master.set_mode('RTL')
    if 'cap' in locals(): cap.release()