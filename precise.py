#!/usr/bin/python3
import jetson_inference, jetson_utils
import cv2, serial, numpy as np, os, time, subprocess
from shapely.geometry import box

# --- 1. CONFIGURATION ---
SERIAL_PORT = "/dev/ttyTHS1"
BAUD_RATE = 9600
CONFIG_FILE = "fluid_config.txt"
WINDOW_NAME = "Jetson Master Control - PRECISE"
INPUT_W, INPUT_H = 640, 480 
CANVAS_W, CANVAS_H = 1280, 480
CAM_A_ID = "/dev/v4l/by-id/usb-Ingenic_Semiconductor_Co._Ltd_HD_Web_Camera_1234567890-video-index0"
CAM_B_ID = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_Camera_SN0001-video-index0"



def load_fluid_delay():
    try:
        if not os.path.exists(CONFIG_FILE):
            raise FileNotFoundError(f"Missing {CONFIG_FILE}")
        
        with open(CONFIG_FILE, "r") as f:
            value = f.read().strip()
            if not value:
                raise ValueError("Fluid config file is empty")
            return int(value)
            
    except Exception as e:
        # Create a fatal error screen if the config is missing or broken
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "FATAL: CONFIG ERROR", (50, 200), 1, 2, (0, 0, 255), 3)
        cv2.putText(error_img, str(e), (50, 260), 1, 1, (255, 255, 255), 1)
        cv2.imshow("CRITICAL ERROR", error_img)
        cv2.waitKey(0) # Block forever until user kills process
        sys.exit(f"System Halted: {e}")

T_FLUID_STOP_MS = load_fluid_delay()

# --- 2. LAZY LOADED MODELS ---
# We use globals but initialize them to None to save memory/power at boot
net_human = None
net_depth = None
depth_mem_l = None
depth_mem_r = None

def get_human_net():
    global net_human
    if net_human is None:
        net_human = jetson_inference.detectNet("ssd-mobilenet-v2", threshold=0.35)
    return net_human

def get_depth_resources():
    global net_depth, depth_mem_l, depth_mem_r
    if net_depth is None:
        net_depth = jetson_inference.depthNet("fcn-mobilenet")
        depth_mem_l = jetson_utils.cudaAllocMapped(width=INPUT_W, height=INPUT_H, format="rgba8")
        depth_mem_r = jetson_utils.cudaAllocMapped(width=INPUT_W, height=INPUT_H, format="rgba8")
    return net_depth, depth_mem_l, depth_mem_r

# --- 3. PREDICTION ENGINE ---
class WiperPredictor:
    def __init__(self, side):
        self.side = side
        self.c = self.load_calib()
        self.last_theta = 0
        self.last_time = time.time()
        
    def load_calib(self):
        fname = f"{self.side}_calibration.txt"
        if not os.path.exists(fname): return None
        with open(fname, "r") as f:
            return [float(x) for x in f.read().split(',')]

    def get_impact_point(self, curr_theta):
        if not self.c: return (0, 0)
        now = time.time()
        dt = (now - self.last_time) * 1000 
        omega = (curr_theta - self.last_theta) / dt if dt > 0 else 0
        theta_pred = curr_theta + (omega * T_FLUID_STOP_MS)
        
        rad_scale = (self.c[4] - self.c[3]) / (self.c[6] - self.c[5])
        target_rad = self.c[3] + (theta_pred - self.c[5]) * rad_scale
        px = int(self.c[0] + self.c[2] * np.cos(target_rad))
        py = int(self.c[1] + self.c[2] * np.sin(target_rad))
        
        self.last_theta, self.last_time = curr_theta, now
        return px, py

# --- 4. GLOBAL STATE ---
state = 'DEPTH'
motor_thetas = {'left': 0.0, 'right': 0.0}
predictors = {'left': WiperPredictor('left'), 'right': WiperPredictor('right')}

btn_w, btn_h = 240, 80
water_btn = (CANVAS_W // 2 - 380, CANVAS_H // 2 - 40, btn_w, btn_h)
fert_btn  = (CANVAS_W // 2 - 120, CANVAS_H // 2 - 40, btn_w, btn_h)
calib_btn = (CANVAS_W // 2 + 140, CANVAS_H // 2 - 40, btn_w, btn_h)
back_btn  = (CANVAS_W // 2 - 60, 10, 120, 40)

# --- 3. HELPERS ---
def get_video_index(dev_id_path):
    if not os.path.exists(dev_id_path): return 0
    return int(''.join(filter(str.isdigit, os.path.realpath(dev_id_path).split('/')[-1])))

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
except:
    ser = None

def on_mouse_click(event, x, y, flags, param):
    global state
    if event == cv2.EVENT_LBUTTONDOWN:
        rect = cv2.getWindowImageRect(WINDOW_NAME)
        sx = x * (CANVAS_W / max(rect[2], 1))
        sy = y * (CANVAS_H / max(rect[3], 1))
        if state == 'DEPTH':
            if water_btn[1] < sy < water_btn[1]+btn_h:
                if water_btn[0] < sx < water_btn[0]+btn_w: state = 'GREEN'
                elif fert_btn[0] < sx < fert_btn[0]+btn_w: state = 'HUMAN'
                elif calib_btn[0] < sx < calib_btn[0]+btn_w:
                    print("Stopping pumps for calibration...")
                    if ser:
                        ser.write(b";a,0;;b,0;\n") # Force pumps OFF
                        ser.close()
                    # 1. RELEASE RESOURCES
                    print("Releasing hardware for calibration...")
                    cap_l.release()
                    cap_r.release()
                    if ser:
                        ser.close()
                    
                    cv2.destroyAllWindows() # Close the main UI

                    # 2. HANDOVER CONTROL
                    # This line blocks 'precise.py' until the calibration script is closed
                    subprocess.run(["python3", "calibrate_wiper2.py"])

                    # 3. RE-ACQUIRE RESOURCES
                    print("Calibration finished. Re-acquiring hardware...")
                    
                    # Re-open Serial
                    try:
                        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
                    except:
                        ser = None
                    
                    # Re-open Cameras
                    cap_l = cv2.VideoCapture(get_video_index(CAM_A_ID))
                    cap_r = cv2.VideoCapture(get_video_index(CAM_B_ID))
                    
                    # Re-load Predictions (The 'Lazy Loading' update)
                    predictors['left'] = WiperPredictor('left')
                    predictors['right'] = WiperPredictor('right')
                    T_FLUID_STOP_MS = load_fluid_delay()
                    

                    # Re-create Main Window
                    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                    cv2.setMouseCallback(WINDOW_NAME, on_mouse_click)
                    
        if back_btn[1] < sy < back_btn[1]+40 and back_btn[0] < sx < back_btn[0]+120:
            state = 'DEPTH'

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW_NAME, on_mouse_click)

cap_l = cv2.VideoCapture(get_video_index(CAM_A_ID), cv2.CAP_V4L2)
cap_r = cv2.VideoCapture(get_video_index(CAM_B_ID), cv2.CAP_V4L2)

# --- 5. MAIN LOOP ---
while True:
    if ser and ser.in_waiting > 0:
        line = ser.readline().decode('utf-8', errors='ignore').strip().lower()
        if ':' in line:
            side, val = line.split(':')
            if 'left' in side: motor_thetas['left'] = float(val)
            elif 'right' in side: motor_thetas['right'] = float(val)

    ret_l, img_l = cap_l.read(); ret_r, img_r = cap_r.read()
    if not ret_l or not ret_r: break

    # Move to CUDA only when needed
    cuda_l = jetson_utils.cudaFromNumpy(cv2.cvtColor(img_l, cv2.COLOR_BGR2RGBA))
    cuda_r = jetson_utils.cudaFromNumpy(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGBA))

    if state == 'DEPTH':
        # --- DEPTH MAP LOGIC ---
        net, mem_l, mem_r = get_depth_resources()
        net.Process(cuda_l, mem_l, "mask") # "mask" is the heat map
        net.Process(cuda_r, mem_r, "mask")
        out_l = cv2.cvtColor(jetson_utils.cudaToNumpy(mem_l), cv2.COLOR_RGBA2BGR)
        out_r = cv2.cvtColor(jetson_utils.cudaToNumpy(mem_r), cv2.COLOR_RGBA2BGR)
    
    else:
        out_l, out_r = img_l.copy(), img_r.copy()
        for side, frame, cuda_img, prefix in [('left', out_l, cuda_l, 'a'), ('right', out_r, cuda_r, 'b')]:
            px, py = predictors[side].get_impact_point(motor_thetas[side])
            hit = False
            
            if state == 'HUMAN':
                # --- HUMAN DETECTION LOGIC ---
                net = get_human_net()
                detections = net.Detect(cuda_img)
                vicinity = box(px-25, py-25, px+25, py+25)
                for d in detections:
                    if net.GetClassDesc(d.ClassID) == 'person':
                        if box(d.Left, d.Top, d.Right, d.Bottom).intersects(vicinity):
                            hit = True; break
                            
                            
            elif state == 'GREEN':
                # --- GREEN COLOR LOGIC WITH HYSTERESIS ---
                roi = frame[max(0, py-25):min(INPUT_H, py+25), max(0, px-25):min(INPUT_W, px+25)]
                
                is_green_now = False
                if roi.size > 0:
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv_roi, (35, 40, 40), (85, 255, 255))
                    # If more than 15% of the vicinity is green, it's a 'hit' for this frame
                    is_green_now = cv2.countNonZero(mask) > (roi.size * 0.15)

                # Update history buffer
                green_history[side].append(is_green_now)
                if len(green_history[side]) > GREEN_BUFFER_SIZE:
                    green_history[side].pop(0)

                # HYSTERESIS DECISION:
                # We only change the 'hit' status if the buffer is consistently one way.
                # Requirement: 3 out of 5 frames must agree to change state.
                if sum(green_history[side]) >= 3:
                    hit = True
                elif sum(green_history[side]) <= 1:
                    hit = False
                else:
                    # Keep previous 'hit' state to prevent chattering
                    # (Note: 'hit' is initialized to False at the start of the 'for' loop, 
                    # so you might need to persist 'hit' in a global if you want true sticky behavior)
                    pass

    # Rendering
    canvas = np.hstack((out_l, out_r))
    if state == 'DEPTH':
        for b, txt, col in [(water_btn, "WATER", (0,160,0)), (fert_btn, "FERT", (0,0,160)), (calib_btn, "CALIBr8", (160,0,0))]:
            cv2.rectangle(canvas, (b[0], b[1]), (b[0]+btn_w, b[1]+btn_h), col, -1)
            cv2.putText(canvas, txt, (b[0]+50, b[1]+55), 1, 2, (255,255,255), 3)
    else:
        cv2.rectangle(canvas, (back_btn[0], back_btn[1]), (back_btn[0]+120, back_btn[1]+40), (50,50,50), -1)
        cv2.putText(canvas, "BACK", (back_btn[0]+30, back_btn[1]+30), 1, 1.5, (255,255,255), 2)

    cv2.imshow(WINDOW_NAME, canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap_l.release(); cap_r.release(); cv2.destroyAllWindows()