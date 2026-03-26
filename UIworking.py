import jetson_inference
import jetson_utils
import cv2
import numpy as np
import time
import os
import serial

# --- Configuration ---
CAM_A_ID = "/dev/v4l/by-id/usb-Ingenic_Semiconductor_Co._Ltd_HD_Web_Camera_1234567890-video-index0"
CAM_B_ID = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_Camera_SN0001-video-index0"
WINDOW_NAME = "Farmonaut GearX Control"
SERIAL_PORT = "/dev/ttyTHS1"

width, height = 1280, 720
TARGET_FPS = 24
FRAME_TIME = 1.0 / TARGET_FPS

# 10 Nozzle Coordinates
LEFT_POINTS = [(150, 325), (250, 325), (350, 325), (450, 325), (550, 325)]
RIGHT_POINTS = [(730, 325), (830, 325), (930, 325), (1030, 325), (1130, 325)]
ALL_POINTS = LEFT_POINTS + RIGHT_POINTS


# Initialize Depth Network for Idle Mode
# "fcn-mobilenet" is optimized for Jetson Orin Nano
net_depth = jetson_inference.depthNet("fcn-mobilenet")
# Allocate CUDA memory for the depth visualization
depth_field = None


# States
current_screen = "working" 
selected_mode = "idle"
is_running = True
tank_val, tank_cap = 0.0, 1.0 
spray_bits = [0] * 10 
human_bits = [0] * 10

# Interaction
mx, my, m_clicked = -1, -1, False

def on_mouse_click(event, x, y, flags, param):
    global mx, my, m_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        mx, my = x, y
        m_clicked = True

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback(WINDOW_NAME, on_mouse_click)

# AI Models
net_safety = jetson_inference.detectNet("ssd-mobilenet-v2", threshold=0.4)
net_weed = None 

try:
    ser = serial.Serial(SERIAL_PORT, 9600, timeout=0.1)
except:
    ser = None

def get_video_index(path):
    if not os.path.exists(path): return None
    return int(''.join(filter(str.isdigit, os.path.realpath(path).split('/')[-1])))

def open_camera(path, target_w, target_h):
    idx = get_video_index(path)
    if idx is None: return None
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    # Set MJPG to handle higher bandwidth at these resolutions
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
    return cap

# Set specific resolutions as requested
cam_a = open_camera(CAM_A_ID, 960, 540) # Ingenic 1080p -> 960x540
cam_b = open_camera(CAM_B_ID, 640, 360) # Sonic 720p -> 640x360

def get_nozzle_mask(detections, points):
    bits = [0] * len(points)
    for d in detections:
        for i, pt in enumerate(points):
            if d.Left <= pt[0] <= d.Right and d.Top <= pt[1] <= d.Bottom:
                bits[i] = 1
    return bits

def draw_working_screen(img, frame_a, frame_b, safety_dets, mode_dets):
    # Adjusted proportions: 16:9 feeds (500x281) to fit UI spacing
    # Previously 500x350 (4:3 approx), now using standard widescreen height
    if frame_a is not None: img[180:461, 100:600] = cv2.resize(frame_a, (500, 281))
    if frame_b is not None: img[180:461, 680:1180] = cv2.resize(frame_b, (500, 281))
    
    # Updated rectangles to match new feed height
    cv2.rectangle(img, (100, 180), (600, 461), (255, 255, 255), 2)
    cv2.rectangle(img, (680, 180), (1180, 461), (255, 255, 255), 2)

    cv2.putText(img, "Farmonaut GearX", (480, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.putText(img, f"MODE: {selected_mode.upper()}", (540, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    for d in safety_dets + mode_dets:
        label = "person" if d in safety_dets else selected_mode
        color = (0, 0, 255) if label == "person" else (255, 255, 0)
        cv2.rectangle(img, (int(d.Left), int(d.Top)), (int(d.Right), int(d.Bottom)), color, 2)
        cv2.putText(img, label, (int(d.Left), int(d.Top)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for i, pt in enumerate(ALL_POINTS):
        color = (0, 255, 0) if spray_bits[i] == 1 else (0, 0, 255)
        if human_bits[i] == 1:
            cv2.circle(img, pt, 22, (0, 165, 255), 3) 
        cv2.rectangle(img, (pt[0]-15, pt[1]-15), (pt[0]+15, pt[1]+15), color, -1 if spray_bits[i] else 2)

    # Tank UI logic
    fill_pct = tank_val / tank_cap if tank_cap > 0 else 0
    fill_h = int(300 * min(max(fill_pct, 0), 1))
    cv2.rectangle(img, (1210, 150), (1240, 450), (255, 255, 255), 2)
    cv2.rectangle(img, (1212, 450 - fill_h), (1238, 448), (180, 180, 180), -1)
    cv2.putText(img, f"{int(fill_pct*100)}%", (1205, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.rectangle(img, (30, 30), (120, 80), (50, 50, 200), -1)
    cv2.putText(img, "QUIT", (45, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.rectangle(img, (580, 550), (710, 650), (255, 255, 255), 2)
    cv2.putText(img, "MODE", (610, 610), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    if selected_mode != "idle":
        cv2.rectangle(img, (1170, 580), (1270, 650), (0, 0, 255), -1) 
        cv2.putText(img, "STOP", (1190, 625), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img

while is_running:
    start_time = time.time()
    
    # Serial Inbound
    if ser and ser.in_waiting > 0:
        try:
            line = ser.readline().decode().strip()
            if line.startswith("read:") and line.endswith(";"):
                parts = line[5:-1].split(',')
                if len(parts) == 2:
                    tank_val, tank_cap = float(parts[0]), float(parts[1])
        except: pass

    ui_frame = np.zeros((height, width, 3), dtype=np.uint8)

    if current_screen == "working":
        ret_a, frame_a = cam_a.read(); ret_b, frame_b = cam_b.read()
        if not (ret_a and ret_b): continue
        
        # Internal 1280x480 panorama (Stretched internally for analysis)
        panorama = np.hstack((cv2.resize(frame_a, (640, 480)), cv2.resize(frame_b, (640, 480))))
        cuda_img = jetson_utils.cudaFromNumpy(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGBA))
        
        safety_dets = net_safety.Detect(cuda_img)
        human_bits = get_nozzle_mask([d for d in safety_dets if net_safety.GetClassDesc(d.ClassID) == "person"], ALL_POINTS)
        
        mode_dets, mode_bits = [], [0] * 10
     
        if selected_mode == "target":
            gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            kernel = np.ones((5,5), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            
            ui_frame = draw_working_screen(ui_frame, frame_a, frame_b, safety_dets, mode_dets)
            
            for i, pt in enumerate(ALL_POINTS):
                px, py = pt[0], pt[1] - 150 
                if 0 <= py < 480:
                    roi = dilated_edges[max(0, py-20):min(480, py+20), max(0, px-20):min(1280, px+20)]
                    if cv2.countNonZero(roi) > 15: 
                        mode_bits[i] = 1

            view_a_mask = cv2.resize(dilated_edges[:, 0:640], (500, 281))
            view_b_mask = cv2.resize(dilated_edges[:, 640:1280], (500, 281))
            ui_frame[180:461, 100:600][view_a_mask > 0] = [0, 0, 0]
            ui_frame[180:461, 680:1180][view_b_mask > 0] = [0, 0, 0]
            
        elif selected_mode == "weed":
            if net_weed is None: 
                net_weed = jetson_inference.detectNet(model="weed.onnx", labels="weed_labels.txt", input_blob="input_0", output_cvg="scores", output_bbox="boxes")
            mode_dets = net_weed.Detect(cuda_img)
            mode_bits = get_nozzle_mask(mode_dets, ALL_POINTS)
            ui_frame = draw_working_screen(ui_frame, frame_a, frame_b, safety_dets, mode_dets)

        elif selected_mode == "broadcast":
            hsv = cv2.cvtColor(panorama, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
            mode_bits = [1 if mask[pt[1]-150, pt[0]] > 0 else 0 for pt in ALL_POINTS]
            ui_frame = draw_working_screen(ui_frame, frame_a, frame_b, safety_dets, mode_dets)

        elif selected_mode == "simple":
            mode_bits = [1] * 10
            ui_frame = draw_working_screen(ui_frame, frame_a, frame_b, safety_dets, mode_dets)
        elif selected_mode == "idle":
            
            if depth_field is None:
                # Allocate once to match panorama resolution (1280x480)
                depth_field = jetson_utils.cudaAllocMapped(width=1280, height=480, format="rgba8")
            
            net_depth.Process(cuda_img, depth_field, "viridis") # Viridis provides clear depth contrast
            
            depth_numpy = cv2.cvtColor(jetson_utils.cudaToNumpy(depth_field), cv2.COLOR_RGBA2BGR)
            
            spray_bits = [0] * 10
            ui_frame = draw_working_screen(ui_frame, None, None, safety_dets, mode_dets)
            
            ui_frame[180:461, 100:600] = cv2.resize(depth_numpy[:, 0:640], (500, 281))
            ui_frame[180:461, 680:1180] = cv2.resize(depth_numpy[:, 640:1280], (500, 281))
            
        else:
            ui_frame = draw_working_screen(ui_frame, frame_a, frame_b, safety_dets, mode_dets)
            
        spray_bits = [mode_bits[i] if human_bits[i] == 0 else 0 for i in range(10)]
        
        if ser:
            l_str = ",".join(map(str, spray_bits[:5]))
            r_str = ",".join(map(str, spray_bits[5:]))
            ser.write(f"write:{l_str};{r_str}\r\n".encode())

    else:
        cv2.rectangle(ui_frame, (50, 150), (250, 550), (60, 60, 180), -1)
        cv2.putText(ui_frame, "BACK", (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
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
            elif selected_mode != "idle" and 1170 <= mx <= 1270 and 580 <= my <= 650:
                selected_mode = "idle"; spray_bits = [0]*10
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
    elapsed = time.time() - start_time
    if elapsed < FRAME_TIME: time.sleep(FRAME_TIME - elapsed)

if cam_a: cam_a.release()
if cam_b: cam_b.release()
cv2.destroyAllWindows()