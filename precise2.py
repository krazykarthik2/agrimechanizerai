import jetson_inference
import jetson_utils
import cv2
import numpy as np
import time
import os
import serial

# --- 1. CONFIGURATION & SETUP ---
# Camera IDs usually correspond to the physical USB ports
CAM_A_ID = "/dev/v4l/by-id/usb-Ingenic_Semiconductor_Co._Ltd_HD_Web_Camera_1234567890-video-index0"
CAM_B_ID = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_Camera_SN0001-video-index0"
WINDOW_NAME = "Farmonaut GearX Control"
SERIAL_PORT = "/dev/ttyTHS1"
CONFIG_FILE = "fluid_config.txt"

width, height = 1280, 720
TARGET_FPS = 24
FRAME_TIME = 1.0 / TARGET_FPS

# Initialize AI Networks
# DepthNet for idle visualization
net_depth = jetson_inference.depthNet("fcn-mobilenet")
# DetectNet for person safety (always running)
net_safety = jetson_inference.detectNet("ssd-mobilenet-v2", threshold=0.4)
net_weed = None # Lazy loaded only when 'weed' mode is selected
depth_field = None

# --- 2. PREDICTION ENGINE (ARC-BASED) ---
class WiperPredictor:
    def __init__(self, side):
        self.side = side
        self.last_theta = 0
        self.last_time = time.time()
        self.calib = self.load_normalized_calib()
        
    def load_normalized_calib(self):
        """Loads the 7 parameters saved by calibrate_wiper2.py (Normalized xc,yc,r)"""
        fname = f"{self.side}_calibration.txt"
        if not os.path.exists(fname): return None
        with open(fname, "r") as f:
            # Format: xc,yc,r,theta_start,theta_end,stream_min,stream_max
            return [float(x) for x in f.read().split(',')]

    def get_points(self, curr_theta, frame_w, frame_h):
        """
        Calculates:
        1. Current Position (Blue): Where the arm is physically located now.
        2. Predicted Position (Green): Where water hits after the fluid travel delay.
        """
        if not self.calib: return None, None
        n_xc, n_yc, n_r, t_start, t_end, s_min, s_max = self.calib
        
        # Get Fluid Delay from config file (constant pump travel time)
        try:
            with open(CONFIG_FILE, "r") as f:
                delay_ms = int(f.read().strip())
        except: delay_ms = 250
            
        # Calculate Angular Velocity (omega) to predict where the arm will be when fluid lands
        now = time.time()
        dt = (now - self.last_time) * 1000 
        omega = (curr_theta - self.last_theta) / dt if dt > 0 else 0
        
        # Predict theta for the moment of impact
        theta_pred = curr_theta + (omega * delay_ms)
        self.last_theta, self.last_time = curr_theta, now

        # Map Serial Motor values back to Arc Radians (Theta)
        def motor_to_rad(m_val):
            if abs(s_max - s_min) < 1e-6: return 0
            scale = (t_end - t_start) / (s_max - s_min)
            return t_start + (m_val - s_min) * scale
        
        # Convert Normalized coordinates to current Pixel space
        diag = np.sqrt(frame_w**2 + frame_h**2)
        xc, yc, r = n_xc * frame_w, n_yc * frame_h, n_r * diag
        
        # Current Arm Pos (Blue tracking)
        rad_now = motor_to_rad(curr_theta)
        p_now = (int(xc + r * np.cos(rad_now)), int(yc + r * np.sin(rad_now)))
        
        # Predicted Fluid Landing Pos (Green tracking)
        rad_pred = motor_to_rad(theta_pred)
        p_pred = (int(xc + r * np.cos(rad_pred)), int(yc + r * np.sin(rad_pred)))
        
        return p_now, p_pred

# --- 3. STATE & GLOBAL UTILS ---
current_screen = "working" 
selected_mode = "idle"
is_running = True
tank_val, tank_cap = 0.0, 1.0 
motor_thetas = {'left': 0.0, 'right': 0.0}
final_spray = [0, 0] # [Left Bank, Right Bank]

predictors = {'left': WiperPredictor('left'), 'right': WiperPredictor('right')}

try:
    ser = serial.Serial(SERIAL_PORT, 9600, timeout=0.1)
except:
    ser = None

# Mouse Interaction for Fullscreen UI
mx, my, m_clicked = -1, -1, False
def on_mouse_click(event, x, y, flags, param):
    global mx, my, m_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        mx, my = x, y; m_clicked = True

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

# --- 4. CORE DRAWING FUNCTION (Matches UIworking.py) ---
def draw_working_screen(img, frame_a, frame_b, safety_dets, mode_dets, pts_map):
    # Resize camera feeds to fit UI containers (16:9 widescreen height)
    if frame_a is not None: img[180:461, 100:600] = cv2.resize(frame_a, (500, 281))
    if frame_b is not None: img[180:461, 680:1180] = cv2.resize(frame_b, (500, 281))
    
    # Draw static containers and branding
    cv2.rectangle(img, (100, 180), (600, 461), (255, 255, 255), 2)
    cv2.rectangle(img, (680, 180), (1180, 461), (255, 255, 255), 2)
    cv2.putText(img, "GearX PRECISE", (480, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.putText(img, f"MODE: {selected_mode.upper()}", (540, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # DRAW ARC OVERLAY: Visualizes the arm's path and predicted impact
    for i, side in enumerate(['left', 'right']):
        p_now, p_pred = pts_map[side]
        cal = predictors[side].calib
        if cal and p_now and p_pred:
            n_xc, n_yc, n_r, t_start, t_end, _, _ = cal
            diag = np.sqrt(640**2 + 480**2)
            xc, yc, r = n_xc * 640, n_yc * 480, n_r * diag
            
            offset_x = 100 if side == 'left' else 680
            def to_ui(pt):
                # Scale coordinate from 640x480 panorama space to the 500x281 UI viewport
                ux = offset_x + int((pt[0] % 640) * (500/640))
                uy = 180 + int(pt[1] * (281/480))
                return (ux, uy)

            # Draw the faint gray trajectory arc
            arc_pts = [to_ui((xc + r*np.cos(t), yc + r*np.sin(t))) for t in np.linspace(t_start, t_end, 50)]
            cv2.polylines(img, [np.array(arc_pts, np.int32)], False, (100, 100, 100), 1)

            # Draw Current Position (Blue Circle)
            ui_now = to_ui(p_now)
            cv2.circle(img, ui_now, 6, (255, 0, 0), -1) 
            
            # Draw Landing Impact (Green Circle if spraying, Red if safety-blocked)
            ui_pred = to_ui(p_pred)
            color = (0, 255, 0) if final_spray[i] else (0, 0, 255)
            cv2.circle(img, ui_pred, 10, color, 3)

    # Tank Level Logic (Exact copy)
    fill_pct = tank_val / tank_cap if tank_cap > 0 else 0
    cv2.rectangle(img, (1210, 150), (1240, 450), (255, 255, 255), 2)
    cv2.rectangle(img, (1212, 450 - int(300*fill_pct)), (1238, 448), (180, 180, 180), -1)
    
    # UI Control Buttons (Exact copy)
    cv2.rectangle(img, (30, 30), (120, 80), (50, 50, 200), -1) # QUIT
    cv2.putText(img, "QUIT", (45, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.rectangle(img, (580, 550), (710, 650), (255, 255, 255), 2) # MODE
    cv2.putText(img, "MODE", (610, 610), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if selected_mode != "idle":
        cv2.rectangle(img, (1170, 580), (1270, 650), (0, 0, 255), -1) # STOP
        cv2.putText(img, "STOP", (1190, 625), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return img

# --- 5. MAIN LOOP ---
while is_running:
    start_time = time.time()
    
    # SERIAL INPUT: Read Arm positions and Tank sensor data
    if ser and ser.in_waiting > 0:
        try:
            line = ser.readline().decode().strip().lower()
            if line.startswith("read:"):
                parts = line[5:-1].split(',')
                tank_val, tank_cap = float(parts[0]), float(parts[1])
            elif ':' in line:
                side, val = line.split(':')
                motor_thetas[side.strip()] = float(val)
        except: pass

    ui_frame = np.zeros((height, width, 3), dtype=np.uint8)

    if current_screen == "working":
        ret_a, frame_a = cam_a.read(); ret_b, frame_b = cam_b.read()
        if not (ret_a and ret_b): continue
        
        # Merge feeds into a 1280x480 panorama for AI processing
        panorama = np.hstack((cv2.resize(frame_a, (640, 480)), cv2.resize(frame_b, (640, 480))))
        cuda_img = jetson_utils.cudaFromNumpy(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGBA))
        
        # PREDICTION: Determine Landing points based on current arm speed
        pts_map = {
            'left': predictors['left'].get_points(motor_thetas['left'], 640, 480),
            'right': predictors['right'].get_points(motor_thetas['right'], 640, 480)
        }

        # SAFETY: Always detect humans to prevent spraying near them
        safety_dets = net_safety.Detect(cuda_img)
        spray_allowed = [1, 1] 
        for d in safety_dets:
            if net_safety.GetClassDesc(d.ClassID) == "person":
                for i, side in enumerate(['left', 'right']):
                    _, p_pred = pts_map[side]
                    # SAFETY INTERLOCK: If person is at the PREDICTED impact site, lock pump
                    if p_pred and d.Left <= p_pred[0] <= d.Right and d.Top <= p_pred[1] <= d.Bottom:
                        spray_allowed[i] = 0

        mode_dets = []
        mode_bits = [0, 0]

        # MODE: WEED DETECTION (ML Based)
        if selected_mode == "weed":
            if net_weed is None:
                net_weed = jetson_inference.detectNet(model="weed.onnx", labels="weed_labels.txt", input_blob="input_0", output_cvg="scores", output_bbox="boxes")
            mode_dets = net_weed.Detect(cuda_img)
            for d in mode_dets:
                for i, side in enumerate(['left', 'right']):
                    _, p_pred = pts_map[side]
                    if p_pred and d.Left <= p_pred[0] <= d.Right and d.Top <= p_pred[1] <= d.Bottom:
                        mode_bits[i] = 1

        # MODE: BROADCAST (Color Based - Greenery)
        elif selected_mode == "broadcast":
            hsv = cv2.cvtColor(panorama, cv2.COLOR_BGR2HSV)
            # Detect standard greenery range
            mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
            for i, side in enumerate(['left', 'right']):
                _, p_pred = pts_map[side]
                if p_pred:
                    # Check if the pixel at the predicted landing point is Green
                    py, px = int(p_pred[1]), int(p_pred[0])
                    if 0 <= py < 480 and 0 <= px < 1280:
                        if mask[py, px] > 0:
                            mode_bits[i] = 1

        # MODE: SIMPLE (Continuous Spray)
        elif selected_mode == "simple":
            mode_bits = [1, 1]

        # Execute final spray decision (Object hit AND no human present)
        final_spray = [mode_bits[i] and spray_allowed[i] for i in range(2)]

        # IDLE: Show Depth Map
        if selected_mode == "idle":
            if depth_field is None:
                depth_field = jetson_utils.cudaAllocMapped(width=1280, height=480, format="rgba8")
            net_depth.Process(cuda_img, depth_field, "viridis")
            final_spray = [0, 0]
        
        # Render Working UI
        ui_frame = draw_working_screen(ui_frame, frame_a, frame_b, safety_dets, mode_dets, pts_map)
        
        # Overlay depth visuals if idle
        if selected_mode == "idle":
            depth_numpy = cv2.cvtColor(jetson_utils.cudaToNumpy(depth_field), cv2.COLOR_RGBA2BGR)
            ui_frame[180:461, 100:600] = cv2.resize(depth_numpy[:, 0:640], (500, 281))
            ui_frame[180:461, 680:1180] = cv2.resize(depth_numpy[:, 640:1280], (500, 281))

        # SERIAL OUTPUT: Send nozzle commands to Arduino
        if ser:
            ser.write(f"write:{final_spray[0]},{final_spray[1]}\r\n".encode())

    # --- MENU SCREEN (Mode Selection Grid) ---
    else:
        cv2.rectangle(ui_frame, (50, 150), (250, 550), (60, 60, 180), -1)
        cv2.putText(ui_frame, "BACK", (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        modes = [["target", "broadcast"], ["weed", "simple"]]
        for r in range(2):
            for c in range(2):
                x1, y1 = 350 + (c * 400), 150 + (r * 200)
                cv2.rectangle(ui_frame, (x1, y1), (x1+380, y1+180), (80, 80, 80), -1)
                cv2.putText(ui_frame, modes[r][c].upper(), (x1 + 80, y1 + 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # UI Interaction (Mouse clicks for fullscreen buttons)
    if m_clicked:
        if current_screen == "working":
            if 30 <= mx <= 120 and 30 <= my <= 80: is_running = False
            elif 580 <= mx <= 710 and 550 <= my <= 650: current_screen = "menu"
            elif selected_mode != "idle" and 1170 <= mx <= 1270 and 580 <= my <= 650:
                selected_mode = "idle"; final_spray = [0, 0]
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