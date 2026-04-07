import cv2
import numpy as np
import random
import sys
import os
import tkinter as tk
from tkinter import simpledialog

# --- STEP 1: COLLECT ALL INPUTS FIRST ---
def collect_all_settings():
    """Collects all numeric ranges for both cameras before starting OpenCV."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    
    settings = {}
    for side in ["left", "right"]:
        print(f"Collecting values for {side}...")
        v1 = simpledialog.askfloat("Setup", f"{side.upper()}: Value for RED (Start):", parent=root)
        v2 = simpledialog.askfloat("Setup", f"{side.upper()}: Value for BLUE (End):", parent=root)
        
        if v1 is None or v2 is None:
            print(f"Error: Missing input for {side}. Exiting.")
            root.destroy()
            sys.exit()
        settings[side] = (v1, v2)
    
    root.destroy()
    return settings

def get_screen_res():
    root = tk.Tk()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    return sw, sh

SCREEN_W, SCREEN_H = get_screen_res()
TARGET_WIN_H = int(SCREEN_H * 0.7)

# --- CORE RANSAC LOGIC ---
def get_circle_3p(p1, p2, p3):
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3
    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(D) < 1e-6: return None
    xc = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / D
    yc = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / D
    r = np.sqrt((x1 - xc)**2 + (y1 - yc)**2)
    return xc, yc, r

def ransac_fit_circle(pts, iterations=100, threshold=2.0):
    if len(pts) < 3: return None
    best_c, max_inliers = None, -1
    for _ in range(iterations):
        sample = random.sample(pts, 3)
        circle = get_circle_3p(sample[0], sample[1], sample[2])
        if circle is None: continue
        xc, yc, r = circle
        inliers = sum(1 for p in pts if abs(np.sqrt((p[0]-xc)**2 + (p[1]-yc)**2) - r) < threshold)
        if inliers > max_inliers:
            max_inliers, best_c = inliers, circle
    return best_c

# --- UI STATE ---
arc_pixels = []
btn_triggered = None
current_delay = 250

def click_event(event, x, y, flags, param):
    global arc_pixels, btn_triggered
    if event == cv2.EVENT_LBUTTONDOWN:
        for name, (x1, y1, x2, y2) in param['btns'].items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                btn_triggered = name
                return
        if y < param['h']:
            arc_pixels.append((x, y))

def run_calibration(cam_index, side_name, manual_values):
    global arc_pixels, btn_triggered, current_delay
    v_start, v_end = manual_values
    
    arc_pixels = []
    swap_ext = False
    btn_triggered = None

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Cam {cam_index} not found.")
        return

    ret, first_frame = cap.read()
    if not ret: return
    orig_h, orig_w = first_frame.shape[:2]
    
    scale = TARGET_WIN_H / (orig_h + 100)
    disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)
    
    btns = {
        "SWAP": (10, disp_h + 20, 110, disp_h + 70),
        "RESET": (120, disp_h + 20, 220, disp_h + 70),
        "SAVE": (230, disp_h + 20, 330, disp_h + 70),
        "QUIT": (disp_w - 110, disp_h + 20, disp_w - 10, disp_h + 70)
    }

    win_name = f"Calibrate {side_name.upper()}"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, click_event, param={'h': disp_h, 'btns': btns})
    cv2.createTrackbar("Delay", win_name, current_delay, 5000, lambda x: globals().update(current_delay=x))

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        canvas = np.zeros((disp_h + 100, disp_w, 3), dtype=np.uint8)
        canvas[0:disp_h, 0:disp_w] = cv2.resize(frame, (disp_w, disp_h))

        cv2.rectangle(canvas, (0, disp_h), (disp_w, disp_h+100), (40, 40, 40), -1)
        for name, (x1, y1, x2, y2) in btns.items():
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (70, 70, 70), -1)
            cv2.putText(canvas, name, (x1+10, y1+35), 1, 1.1, (255, 255, 255), 2)

        for p in arc_pixels:
            cv2.circle(canvas, p, 4, (0, 255, 0), -1)

        result = None
        if len(arc_pixels) >= 3:
            result = ransac_fit_circle(arc_pixels)
            if result:
                xc, yc, r = result
                angles = [np.arctan2(p[1]-yc, p[0]-xc) for p in arc_pixels]
                a1, a2 = min(angles), max(angles)
                start_a, end_a = (a2, a1) if swap_ext else (a1, a2)
                
                pts = [[int(xc+r*np.cos(t)), int(yc+r*np.sin(t))] for t in np.linspace(a1, a2, 50)]
                cv2.polylines(canvas, [np.array(pts)], False, (255, 255, 0), 2)
                
                cv2.circle(canvas, (int(xc+r*np.cos(start_a)), int(yc+r*np.sin(start_a))), 8, (0, 0, 255), -1) 
                cv2.circle(canvas, (int(xc+r*np.cos(end_a)), int(yc+r*np.sin(end_a))), 8, (255, 0, 0), -1)

        cv2.imshow(win_name, canvas)
        key = cv2.waitKey(1) & 0xFF

        if btn_triggered == "RESET":
            arc_pixels = []; btn_triggered = None
        elif btn_triggered == "SWAP":
            swap_ext = not swap_ext; btn_triggered = None
        elif btn_triggered == "SAVE" and result:
            real_xc, real_yc, real_r = result[0]/scale, result[1]/scale, result[2]/scale
            nx, ny = real_xc/orig_w, real_yc/orig_h
            nr = real_r / np.sqrt(orig_w**2 + orig_h**2)

            with open(f"{side_name}_calibration.txt", "w") as f:
                f.write(f"{nx},{ny},{nr},{start_a},{end_a},{v_start},{v_end}")
            
            with open("fluid_config.txt", "w") as f:
                f.write(str(current_delay))
            
            print(f"Saved {side_name.upper()} successfully.")
            break 

        elif btn_triggered == "QUIT" or key == ord('q'):
            cap.release(); cv2.destroyAllWindows(); sys.exit()

    cap.release()
    cv2.destroyWindow(win_name)
    cv2.waitKey(100) # Give X11 a moment to breathe between windows


CAM_RIGHT_ID = "/dev/v4l/by-id/usb-Ingenic_Semiconductor_Co._Ltd_HD_Web_Camera_1234567890-video-index0"
CAM_LEFT_ID = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_Camera_SN0001-video-index0"

def get_video_index(path):
    if not os.path.exists(path): return None
    return int(''.join(filter(str.isdigit, os.path.realpath(path).split('/')[-1])))

if __name__ == "__main__":
    # Get all inputs first while the screen is clean
    user_inputs = collect_all_settings()
    
    # Now run OpenCV windows sequentially
    run_calibration(get_video_index(CAM_LEFT_ID), "left", user_inputs["left"])
    run_calibration(get_video_index(CAM_RIGHT_ID), "right", user_inputs["right"])