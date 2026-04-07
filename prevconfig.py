import cv2
import numpy as np
import sys
import os

# --- CAMERA CONFIGURATION ---
# Use the specific IDs provided to ensure Left/Right never swap
CAM_RIGHT_ID = "/dev/v4l/by-id/usb-Ingenic_Semiconductor_Co._Ltd_HD_Web_Camera_1234567890-video-index0"
CAM_LEFT_ID = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_Camera_SN0001-video-index0"

def get_video_index(path):
    """Resolves symbolic link to actual /dev/videoX index."""
    if not os.path.exists(path): return None
    return int(''.join(filter(str.isdigit, os.path.realpath(path).split('/')[-1])))

# --- DATA LOADING ---
def load_calibration(side_name):
    fname = f"{side_name}_calibration.txt"
    if not os.path.exists(fname): return None
    try:
        with open(fname, "r") as f:
            data = f.read().strip().split(',')
            return {
                "xc": float(data[0]), "yc": float(data[1]), "r": float(data[2]),
                "start_a": float(data[3]), "end_a": float(data[4]),
                "v_min": float(data[5]), "v_max": float(data[6])
            }
    except Exception: return None

def load_fluid_delay():
    if os.path.exists("fluid_config.txt"):
        try:
            with open("fluid_config.txt", "r") as f:
                return f.read().strip()
        except: return "N/A"
    return "N/A"

def denormalize(cfg, w, h):
    xc, yc = cfg['xc'] * w, cfg['yc'] * h
    diag = np.sqrt(w**2 + h**2)
    r = cfg['r'] * diag
    return xc, yc, r

# --- UI LOGIC ---
exit_flag = False
def click_event(event, x, y, flags, param):
    global exit_flag
    if event == cv2.EVENT_LBUTTONDOWN:
        # Back Button area
        if 10 <= x <= 110 and 10 <= y <= 60:
            exit_flag = True

def run_preview():
    global exit_flag
    
    # Resolve and open specific cameras
    idx_l = get_video_index(CAM_LEFT_ID)
    idx_r = get_video_index(CAM_RIGHT_ID)
    
    cap_l = cv2.VideoCapture(idx_l if idx_l is not None else 0)
    cap_r = cv2.VideoCapture(idx_r if idx_r is not None else 1)
    
    cfg_l = load_calibration("left")
    cfg_r = load_calibration("right")
    fluid_delay = load_fluid_delay()

    win_name = "Wiper Preview - Fullscreen"
    cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(win_name, click_event)

    while not exit_flag:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()

        if not ret_l: frame_l = np.zeros((480, 640, 3), dtype=np.uint8)
        if not ret_r: frame_r = np.zeros((480, 640, 3), dtype=np.uint8)

        h, w = 480, 640
        frame_l = cv2.resize(frame_l, (w, h))
        frame_r = cv2.resize(frame_r, (w, h))

        # Overlay Left
        if cfg_l:
            xc, yc, r = denormalize(cfg_l, w, h)
            pts = [[int(xc + r*np.cos(t)), int(yc + r*np.sin(t))] 
                   for t in np.linspace(cfg_l['start_a'], cfg_l['end_a'], 50)]
            cv2.polylines(frame_l, [np.array(pts)], False, (0, 255, 255), 3)
            cv2.putText(frame_l, f"L: {cfg_l['v_min']}-{cfg_l['v_max']}", (20, h-20), 1, 1.5, (0, 255, 0), 2)

        # Overlay Right
        if cfg_r:
            xc, yc, r = denormalize(cfg_r, w, h)
            pts = [[int(xc + r*np.cos(t)), int(yc + r*np.sin(t))] 
                   for t in np.linspace(cfg_r['start_a'], cfg_r['end_a'], 50)]
            cv2.polylines(frame_r, [np.array(pts)], False, (0, 255, 255), 3)
            cv2.putText(frame_r, f"R: {cfg_r['v_min']}-{cfg_r['v_max']}", (20, h-20), 1, 1.5, (0, 255, 0), 2)

        # Combine side-by-side
        combined = np.hstack((frame_l, frame_r))
        
        # UI Elements
        # Red Back Button
        cv2.rectangle(combined, (10, 10), (110, 60), (0, 0, 255), -1)
        cv2.putText(combined, "BACK", (25, 45), 1, 1.2, (255, 255, 255), 2)
        
        # Fluid Delay Display (Centered at bottom)
        delay_text = f"Fluid Delay: {fluid_delay}ms"
        cv2.putText(combined, delay_text, (520, h-50), 1, 1.5, (250, 250, 0), 2)

        cv2.imshow(win_name, combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_preview()