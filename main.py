#!/usr/bin/python3
import jetson_inference
import jetson_utils
import cv2
import serial
import numpy as np
import os
from shapely.geometry import box, Polygon

# --- 1. CONFIGURATION ---
SERIAL_PORT = "/dev/ttyTHS1"
BAUD_RATE = 9600
# Update these paths if your symlinks change
CAM_A_ID = "/dev/v4l/by-id/usb-Ingenic_Semiconductor_Co._Ltd_HD_Web_Camera_1234567890-video-index0"
CAM_B_ID = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_Camera_SN0001-video-index0"

INPUT_W, INPUT_H = 640, 480 
CANVAS_W, CANVAS_H = 1280, 480

ZONE_LEFT = Polygon([(50, 200), (300, 200), (300, 450), (50, 450)])
ZONE_RIGHT = Polygon([(340, 200), (600, 200), (600, 450), (340, 450)])

LOWER_GREEN = np.array([35, 40, 40])
UPPER_GREEN = np.array([85, 255, 255])

# --- 2. GLOBAL STATE & MODELS ---
state = 'DEPTH'
WINDOW_NAME = "Jetson Master Control"

# Initialize Models Globally to avoid lag spikes
net_human = jetson_inference.detectNet("ssd-mobilenet-v2", threshold=0.35)
net_depth = jetson_inference.depthNet("fcn-mobilenet")

# Pre-allocate CUDA memory for depth
depth_mem_l = jetson_utils.cudaAllocMapped(width=INPUT_W, height=INPUT_H, format="rgba8")
depth_mem_r = jetson_utils.cudaAllocMapped(width=INPUT_W, height=INPUT_H, format="rgba8")

# Button locations
btn_w, btn_h = 240, 80
water_btn = (CANVAS_W // 2 - 260, CANVAS_H // 2 - 40, btn_w, btn_h)
fert_btn  = (CANVAS_W // 2 + 20, CANVAS_H // 2 - 40, btn_w, btn_h)
back_btn  = (CANVAS_W // 2 - 60, 10, 120, 40)

# --- 3. HELPERS ---
def get_video_index(dev_id_path):
    if not os.path.exists(dev_id_path): return 0
    return int(''.join(filter(str.isdigit, os.path.realpath(dev_id_path).split('/')[-1])))

def on_mouse_click(event, x, y, flags, param):
    global state
    if event == cv2.EVENT_LBUTTONDOWN:
        rect = cv2.getWindowImageRect(WINDOW_NAME)
        s_w, s_h = max(rect[2], 1), max(rect[3], 1)
        scale_x = x * (CANVAS_W / s_w)
        scale_y = y * (CANVAS_H / s_h)

        if state == 'DEPTH':
            if water_btn[0] < scale_x < water_btn[0] + water_btn[2] and \
               water_btn[1] < scale_y < water_btn[1] + water_btn[3]:
                state = 'GREEN'
            elif fert_btn[0] < scale_x < fert_btn[0] + fert_btn[2] and \
                 fert_btn[1] < scale_y < fert_btn[1] + fert_btn[3]:
                state = 'HUMAN'
        
        if back_btn[0] < scale_x < back_btn[0] + back_btn[2] and \
           back_btn[1] < scale_y < back_btn[1] + back_btn[3]:
            state = 'DEPTH'

# --- 4. INITIALIZATION ---
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback(WINDOW_NAME, on_mouse_click)

try:
    ser = serial.Serial(SERIAL_PORT, baudrate=BAUD_RATE, timeout=0.1)
except Exception as e:
    print(f"Serial Error: {e}")
    ser = None

cap_left = cv2.VideoCapture(get_video_index(CAM_A_ID), cv2.CAP_V4L2)
cap_right = cv2.VideoCapture(get_video_index(CAM_B_ID), cv2.CAP_V4L2)
for c in [cap_left, cap_right]:
    c.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_W)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_H)

# --- 5. MAIN LOOP ---
while True:
    ret_l, img_l = cap_left.read()
    ret_r, img_r = cap_right.read()
    if not ret_l or not ret_r: break

    # Prep CUDA images once per loop
    cuda_l = jetson_utils.cudaFromNumpy(cv2.cvtColor(img_l, cv2.COLOR_BGR2RGBA))
    cuda_r = jetson_utils.cudaFromNumpy(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGBA))

    if state == 'DEPTH':
        net_depth.Process(cuda_l, depth_mem_l, "mask")
        net_depth.Process(cuda_r, depth_mem_r, "mask")
        out_l = cv2.cvtColor(jetson_utils.cudaToNumpy(depth_mem_l), cv2.COLOR_RGBA2BGR)
        out_r = cv2.cvtColor(jetson_utils.cudaToNumpy(depth_mem_r), cv2.COLOR_RGBA2BGR)

    elif state == 'HUMAN':
        out_l, out_r = img_l.copy(), img_r.copy()
        for frame, cuda_img, zone, prefix in [(out_l, cuda_l, ZONE_LEFT, 'a'), (out_r, cuda_r, ZONE_RIGHT, 'b')]:
            detections = net_human.Detect(cuda_img)
            hit = False
            for d in detections:
                if net_human.GetClassDesc(d.ClassID) == 'person':
                    if box(d.Left, d.Top, d.Right, d.Bottom).intersects(zone):
                        hit = True
                        break
            
            if ser: ser.write(f";{prefix},{'0' if hit else '90'};\n".encode())
            color = (0, 255, 0) if hit else (0, 0, 255)
            cv2.polylines(frame, [np.array(list(zone.exterior.coords), np.int32)], True, color, 3)

    elif state == 'GREEN':
        out_l, out_r = img_l.copy(), img_r.copy()
        for frame, zone, prefix in [(out_l, ZONE_LEFT, 'a'), (out_r, ZONE_RIGHT, 'b')]:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
            poly_mask = np.zeros((INPUT_H, INPUT_W), dtype=np.uint8)
            cv2.fillPoly(poly_mask, [np.array(list(zone.exterior.coords), np.int32)], 255)
            
            green_pixels = cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=poly_mask))
            hit = green_pixels > (cv2.countNonZero(poly_mask) * 0.05)
            
            if ser: ser.write(f";{prefix},{'0' if hit else '90'};\n".encode())
            cv2.polylines(frame, [np.array(list(zone.exterior.coords), np.int32)], True, (0, 255, 0) if hit else (255, 0, 0), 3)

    # --- UI & RENDERING ---
    canvas = np.hstack((out_l, out_r))
    
    if state == 'DEPTH':
        # Draw Menu Buttons
        cv2.rectangle(canvas, (water_btn[0], water_btn[1]), (water_btn[0]+btn_w, water_btn[1]+btn_h), (0, 180, 0), -1)
        cv2.putText(canvas, "WATER (GREEN)", (water_btn[0]+20, water_btn[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.rectangle(canvas, (fert_btn[0], fert_btn[1]), (fert_btn[0]+btn_w, fert_btn[1]+btn_h), (0, 0, 180), -1)
        cv2.putText(canvas, "FERT (HUMAN)", (fert_btn[0]+20, fert_btn[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    else:
        # Draw Back Button only when not in menu
        cv2.rectangle(canvas, (back_btn[0], back_btn[1]), (back_btn[0]+back_btn[2], back_btn[1]+back_btn[3]), (50, 50, 50), -1)
        cv2.putText(canvas, "BACK", (back_btn[0]+35, back_btn[1]+28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # Fullscreen resize
    rect = cv2.getWindowImageRect(WINDOW_NAME)
    final_view = cv2.resize(canvas, (max(rect[2], 1280), max(rect[3], 480)))
    cv2.imshow(WINDOW_NAME, final_view)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()