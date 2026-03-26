import cv2
import numpy as np
import time
import random
import os

# --- Configuration ---
CAM_A_ID = "/dev/v4l/by-id/usb-Ingenic_Semiconductor_Co._Ltd_HD_Web_Camera_1234567890-video-index0"
CAM_B_ID = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_Camera_SN0001-video-index0"
WINDOW_NAME = "Farmonaut GearX Control"

width, height = 1280, 720
TARGET_FPS = 24
FRAME_TIME = 1.0 / TARGET_FPS

# States
current_screen = "working" 
selected_mode = "idle"
is_running = True
tank_val, tank_cap = 156.0, 200.0

# Mouse Global Variables
mx, my = -1, -1
m_clicked = False

def on_mouse_click(event, x, y, flags, param):
    global mx, my, m_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        mx, my = x, y
        m_clicked = True

# Initialize OpenCV Window
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback(WINDOW_NAME, on_mouse_click)

def get_video_index(dev_id_path):
    if not os.path.exists(dev_id_path): return None
    real_path = os.path.realpath(dev_id_path)
    try:
        return int(''.join(filter(str.isdigit, real_path.split('/')[-1])))
    except ValueError: return None

def open_camera(path):
    idx = get_video_index(path)
    if idx is None: return None
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

cam_a = open_camera(CAM_A_ID)
cam_b = open_camera(CAM_B_ID)

def draw_working_screen(img, frame_a, frame_b):
    # --- Render Cam Feeds ---
    if frame_a is not None:
        img[150:500, 100:600] = cv2.resize(frame_a, (500, 350))
    if frame_b is not None:
        img[150:500, 680:1180] = cv2.resize(frame_b, (500, 350))
    
    cv2.rectangle(img, (100, 150), (600, 500), (255, 255, 255), 2)
    cv2.rectangle(img, (680, 150), (1180, 500), (255, 255, 255), 2)

    # UI Elements
    cv2.putText(img, "Farmonaut GearX", (480, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.putText(img, f"MODE: {selected_mode.upper()}", (540, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # --- Gun Status (New Content) ---
    # Left Gun Status
    l_status = "ON" if selected_mode != "idle" else "OFF"
    l_color = (0, 255, 0) if l_status == "ON" else (0, 0, 255)
    cv2.putText(img, f"gun1mode: {l_status}", (100, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.7, l_color, 1)
    cv2.putText(img, f"gun1: {random.randint(0, 180)} angle", (100, 575), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # Right Gun Status
    r_status = "ON" if selected_mode != "idle" else "OFF"
    r_color = (0, 255, 0) if r_status == "ON" else (0, 0, 255)
    cv2.putText(img, f"gun2mode: {r_status}", (680, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.7, r_color, 1)
    cv2.putText(img, f"gun2: {random.randint(0, 180)} angle", (680, 575), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # Tank UI
    cv2.rectangle(img, (1210, 150), (1240, 450), (255, 255, 255), 2)
    fill = int(300 * (tank_val / tank_cap))
    cv2.rectangle(img, (1212, 450 - fill), (1238, 448), (180, 180, 180), -1)
    cv2.putText(img, f"{int(100*tank_val/tank_cap)}%", (1205, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Buttons
    cv2.rectangle(img, (30, 30), (120, 80), (50, 50, 200), -1) # QUIT
    cv2.putText(img, "QUIT", (45, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.rectangle(img, (580, 550), (710, 650), (255, 255, 255), 2) # MODE
    cv2.putText(img, "MODE", (610, 610), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if selected_mode != "idle":
        cv2.rectangle(img, (1170, 580), (1270, 650), (0, 0, 255), -1) # STOP
        cv2.putText(img, "STOP", (1190, 625), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img

def draw_menu_screen(img):
    cv2.rectangle(img, (50, 150), (250, 550), (60, 60, 180), -1)
    cv2.putText(img, "BACK", (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    modes = [["target", "broadcast"], ["weed", "simple"]]
    for r in range(2):
        for c in range(2):
            x1, y1 = 350 + (c * 400), 150 + (r * 200)
            cv2.rectangle(img, (x1, y1), (x1+380, y1+180), (80, 80, 80), -1)
            cv2.putText(img, modes[r][c].upper(), (x1 + 80, y1 + 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img

# --- Main Loop ---
while is_running:
    start_time = time.time()
    ui_frame = np.zeros((height, width, 3), dtype=np.uint8)

    if current_screen == "working":
        ret_a, frame_a = cam_a.read() if cam_a else (False, None)
        ret_b, frame_b = cam_b.read() if cam_b else (False, None)
        ui_frame = draw_working_screen(ui_frame, frame_a, frame_b)
    else:
        ui_frame = draw_menu_screen(ui_frame)

    # --- Button Logic ---
    if m_clicked:
        if current_screen == "working":
            if 30 <= mx <= 120 and 30 <= my <= 80: # QUIT
                is_running = False
            elif 580 <= mx <= 710 and 550 <= my <= 650: # MODE
                current_screen = "menu"
            elif selected_mode != "idle" and 1170 <= mx <= 1270 and 580 <= my <= 650: # STOP
                selected_mode = "idle"
        
        elif current_screen == "menu":
            if 50 <= mx <= 250 and 150 <=UI_main my <= 550: # BACK
                current_screen = "working"
            elif mx > 350:
                col, row = int((mx - 350) // 400), int((my - 150) // 200)
                if 0 <= row < 2 and 0 <= col < 2:
                    selected_mode = [["target", "broadcast"], ["weed", "simple"]][row][col]
                    current_screen = "working"
        m_clicked = False

    cv2.imshow(WINDOW_NAME, ui_frame)
    if cv2.waitKey(1) & 0xFF == 27: break
    elapsed = time.time() - start_time
    if elapsed < FRAME_TIME: time.sleep(FRAME_TIME - elapsed)

if cam_a: cam_a.release()
if cam_b: cam_b.release()
cv2.destroyAllWindows()