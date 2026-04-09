import jetson_inference
import jetson_utils
import cv2
import numpy as np
import time
import os
import random

# --- ARDUINO-STYLE SERVO IMPORTS ---
import board
import busio
import pwmio
from adafruit_motor import servo

# --- Configuration ---
CAM_B_ID = "/dev/v4l/by-id/usb-Ingenic_Semiconductor_Co._Ltd_HD_Web_Camera_1234567890-video-index0"
CAM_A_ID = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_Camera_SN0001-video-index0"
WINDOW_NAME = "Farmonaut GearX Control"

# Use the same Pins (Physical Board Numbering)
SERVOLEFTPIN, SERVORIGHTPIN = 32, 33
SERVO1_ACTIVE, SERVO1_INACTIVE = 170, 90
SERVO2_ACTIVE, SERVO2_INACTIVE = 0, 80

# --- SERVO INITIALIZATION ---
# Pin 32 is PWM0 (board.D12 in Blinka)
# Pin 33 is PWM1 (board.D13 in Blinka)
pwm_l = pwmio.PWMOut(board.D12, duty_cycle=0, frequency=50)
pwm_r = pwmio.PWMOut(board.D13, duty_cycle=0, frequency=50)

# Create servo objects (MG995 usually likes 500us to 2500us pulse width)
servo_l = servo.Servo(pwm_l, min_pulse=500, max_pulse=2500)
servo_r = servo.Servo(pwm_r, min_pulse=500, max_pulse=2500)

# Initial Positions (Arduino style: servo.angle = 90)
servo_l.angle = SERVO1_INACTIVE
servo_r.angle = SERVO2_INACTIVE
# ----------------------------------------------------

last_servo_time = 0
prev_bits = [0, 0]
width, height = 1280, 720
TARGET_FPS = 20
FRAME_TIME = 1.0 / TARGET_FPS

# [Points and Model Initializations remain the same]
LEFT_POINTS = [(150, 325), (250, 325), (350, 325), (450, 325), (550, 325)]
RIGHT_POINTS = [(730, 325), (830, 325), (930, 325), (1030, 325), (1130, 325)]
ALL_POINTS = LEFT_POINTS + RIGHT_POINTS

net_depth = jetson_inference.depthNet("fcn-mobilenet")
net_safety = jetson_inference.detectNet("ssd-mobilenet-v2", threshold=0.4)
depth_field = None

current_screen = "working" 
selected_mode = "idle"
is_running = True
tank_val, tank_cap = 60.0, 100.0 
human_bits = [0] * 10
left_spray_bit = 0
right_spray_bit = 0
mx, my, m_clicked = -1, -1, False

def on_mouse_click(event, x, y, flags, param):
    global mx, my, m_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        mx, my = x, y
        m_clicked = True

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback(WINDOW_NAME, on_mouse_click)

def get_video_index(path):
    if not os.path.exists(path): return None
    return int(''.join(filter(str.isdigit, os.path.realpath(path).split('/')[-1])))

def open_camera(path, target_w, target_h):
    idx = get_video_index(path)
    if idx is None: return None
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
    return cap

cam_a = open_camera(CAM_A_ID, 960, 540)
cam_b = open_camera(CAM_B_ID, 640, 360)

def get_nozzle_mask(detections, points):
    bits = [0] * len(points)
    for d in detections:
        for i, pt in enumerate(points):
            if d.Left <= pt[0] <= d.Right and d.Top <= pt[1] <= d.Bottom:
                bits[i] = 1
    return bits

def draw_working_screen(img, frame_a, frame_b, safety_dets):
    if frame_a is not None: img[180:461, 100:600] = cv2.resize(frame_a, (500, 281))
    if frame_b is not None: img[180:461, 680:1180] = cv2.resize(frame_b, (500, 281))
    cv2.rectangle(img, (100, 180), (600, 461), (255, 255, 255), 2)
    cv2.rectangle(img, (680, 180), (1180, 461), (255, 255, 255), 2)
    cv2.putText(img, "Farmonaut GearX", (480, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.putText(img, f"MODE: {selected_mode.upper()}", (540, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    for d in safety_dets:
        cv2.rectangle(img, (int(d.Left), int(d.Top)), (int(d.Right), int(d.Bottom)), (0, 0, 255), 2)
    for i, pt in enumerate(ALL_POINTS):
        side_bit = left_spray_bit if i < 5 else right_spray_bit
        color = (0, 255, 0) if side_bit == 1 else (0, 0, 255)
        if human_bits[i] == 1: cv2.circle(img, pt, 22, (0, 165, 255), 3) 
        cv2.rectangle(img, (pt[0]-15, pt[1]-15), (pt[0]+15, pt[1]+15), color, -1 if side_bit else 2)
    fill_pct = tank_val / tank_cap
    cv2.rectangle(img, (1210, 150), (1240, 450), (255, 255, 255), 2)
    cv2.rectangle(img, (1212, 450 - int(300*fill_pct)), (1238, 448), (180, 180, 180), -1)
    cv2.rectangle(img, (30, 30), (120, 80), (50, 50, 200), -1); cv2.putText(img, "QUIT", (45, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.rectangle(img, (580, 550), (710, 650), (255, 255, 255), 2); cv2.putText(img, "MODE", (610, 610), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img

# --- Main Loop ---
while is_running:
    start_time = time.time()
    ui_frame = np.zeros((height, width, 3), dtype=np.uint8)

    if current_screen == "working":
        ret_a, frame_a = cam_a.read(); ret_b, frame_b = cam_b.read()
        if not (ret_a and ret_b): continue
        
        panorama = np.hstack((cv2.resize(frame_a, (640, 480)), cv2.resize(frame_b, (640, 480))))
        cuda_img = jetson_utils.cudaFromNumpy(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGBA))
        
        safety_dets = net_safety.Detect(cuda_img)
        human_bits = get_nozzle_mask([d for d in safety_dets if net_safety.GetClassDesc(d.ClassID) == "person"], ALL_POINTS)
        
        mode_bits = [0] * 10
        if selected_mode in ["weed", "broadcast"]:
            hsv = cv2.cvtColor(panorama, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
            mode_bits = [1 if mask[pt[1]-150, pt[0]] > 0 else 0 for pt in ALL_POINTS]
        elif selected_mode == "target":
            gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blurred, 30, 100) 
            frame_a = cv2.cvtColor(edges[0:480, 0:640], cv2.COLOR_GRAY2BGR)
            frame_b = cv2.cvtColor(edges[0:480, 640:1280], cv2.COLOR_GRAY2BGR)
            for i, pt in enumerate(ALL_POINTS):
                px, py = pt[0], pt[1] - 150 
                if 0 <= py < 480:
                    roi = edges[max(0, py-10):min(480, py+10), max(0, px-10):min(1280, px+10)]
                    if cv2.countNonZero(roi) > 5: mode_bits[i] = 1
        elif selected_mode == "simple": mode_bits = [1] * 10
        elif selected_mode == "idle":
            if depth_field is None: depth_field = jetson_utils.cudaAllocMapped(width=1280, height=480, format="rgba8")
            net_depth.Process(cuda_img, depth_field, "viridis")
            depth_numpy = cv2.cvtColor(jetson_utils.cudaToNumpy(depth_field), cv2.COLOR_RGBA2BGR)
            ui_frame[180:461, 100:600] = cv2.resize(depth_numpy[:, 0:640], (500, 281))
            ui_frame[180:461, 680:1180] = cv2.resize(depth_numpy[:, 640:1280], (500, 281))

        left_spray_bit = 1 if (not any(human_bits[0:5]) and sum(mode_bits[0:5]) > 2) else 0
        right_spray_bit = 1 if (not any(human_bits[5:10]) and sum(mode_bits[5:10]) > 2) else 0

        ui_frame = draw_working_screen(ui_frame, frame_a if selected_mode != "idle" else None, frame_b if selected_mode != "idle" else None, safety_dets)
        
        # --- NEW SERVO OUTPUT (Library Driven) ---
        if time.time() - last_servo_time > 0.2:
            if [left_spray_bit, right_spray_bit] != prev_bits:
                l_angle = SERVO1_ACTIVE if left_spray_bit else SERVO1_INACTIVE
                r_angle = SERVO2_ACTIVE if right_spray_bit else SERVO2_INACTIVE
                
                # Simple Arduino-style write commands
                servo_l.write(l_angle)
                servo_r.write(r_angle)
                
                prev_bits = [left_spray_bit, right_spray_bit]
                last_servo_time = time.time()
    
    else:
        # Menu Screen
        cv2.rectangle(ui_frame, (50, 150), (250, 550), (60, 60, 180), -1)
        modes = [["target", "broadcast"], ["weed", "simple"]]
        for r in range(2):
            for c in range(2):
                x1, y1 = 350 + (c * 400), 150 + (r * 200)
                cv2.rectangle(ui_frame, (x1, y1), (x1+380, y1+180), (80, 80, 80), -1)
                cv2.putText(ui_frame, modes[r][c].upper(), (x1 + 80, y1 + 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    if m_clicked:
        if current_screen == "working":
            if 30 <= mx <= 120 and 30 <= my <= 80: is_running = False
            elif 580 <= mx <= 710 and 550 <= my <= 650: current_screen = "menu"
        elif current_screen == "menu":
            if 50 <= mx <= 250 and 150 <= my <= 550: current_screen = "working"
            elif mx > 350:
                col, row = int((mx-350)//400), int((my-150)//200)
                if 0 <= row < 2 and 0 <= col < 2:
                    selected_mode = [["target", "broadcast"], ["weed", "simple"]][row][col]
                    current_screen = "working"
        m_clicked = False

    cv2.imshow(WINDOW_NAME, ui_frame)
    if cv2.waitKey(1) & 0xFF == 27: break
    time.sleep(max(0, FRAME_TIME - (time.time() - start_time)))

# --- Cleanup ---
if cam_a: cam_a.release()
if cam_b: cam_b.release()
# Stopping servos
pwm_l.deinit()
pwm_r.deinit()
cv2.destroyAllWindows()