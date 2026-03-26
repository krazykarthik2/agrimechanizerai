import cv2
import numpy as np
import random
import sys
import os
import serial  # Requires: pip install pyserial

# --- 1. SERIAL CONFIGURATION ---
# Using your specified Orin port and baud rate
SERIAL_PORT = "/dev/ttyTHS1"
BAUD_RATE = 9600

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0) # timeout=0 makes it non-blocking
except Exception as e:
    print(f"Serial Error: {e}. Check permissions or connections.")
    ser = None

# --- 2. CORE RANSAC LOGIC ---
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
    best_circle, max_inliers = None, -1
    for _ in range(iterations):
        sample = random.sample(pts, 3)
        circle = get_circle_3p(sample[0], sample[1], sample[2])
        if circle is None: continue
        xc, yc, r = circle
        inliers = sum(1 for p in pts if abs(np.sqrt((p[0]-xc)**2 + (p[1]-yc)**2) - r) < threshold)
        if inliers > max_inliers:
            max_inliers, best_circle = inliers, circle
    return best_circle

# --- 3. UI & BUTTON LOGIC ---
buttons = {
    "SWAP": (50, 500, 150, 550),
    "RESET": (200, 500, 300, 550),
    "SAVE": (350, 500, 500, 550),
    "QUIT": (550, 500, 650, 550)
}

btn_triggered = None
arc_pixels = []
serial_buffer = ""
current_delay = 250

def on_delay_change(val):
    global current_delay
    current_delay = val

def click_event(event, x, y, flags, param):
    global arc_pixels, btn_triggered
    if event == cv2.EVENT_LBUTTONDOWN:
        for name, (x1, y1, x2, y2) in buttons.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                btn_triggered = name
                return
        if y < 480: 
            arc_pixels.append((x, y))

def parse_serial_stream(side_name, current_min, current_max):
    """
    Parses the incoming serial buffer for "side:value" strings.
    Updates and returns the new min and max.
    """
    global serial_buffer
    if ser and ser.in_waiting > 0:
        try:
            # Read all available bytes and decode
            data = ser.read(ser.in_waiting).decode('utf-8')
            serial_buffer += data
            
            # Process complete lines only
            if "\n" in serial_buffer:
                lines = serial_buffer.split("\n")
                # Keep the last partial line in the buffer
                serial_buffer = lines.pop()
                
                for line in lines:
                    line = line.strip().lower()
                    if f"{side_name}:" in line:
                        # Extract value after the colon
                        _, val_str = line.split(":", 1)
                        val = float(val_str)
                        current_min = min(current_min, val)
                        current_max = max(current_max, val)
        except Exception:
            pass # Ignore malformed serial fragments
            
    return current_min, current_max

def run_calibration(cam_index, side_name):
    global arc_pixels, btn_triggered, swap_extremes, current_delay
    arc_pixels = []
    swap_extremes = False
    btn_triggered = None

    # Initialize slider value from file if it exists.
    if os.path.exists("fluid_config.txt"):
        try:
            with open("fluid_config.txt", "r") as f:
                current_delay = int(f.read().strip())
        except Exception:
            pass
    
    stream_min = float('inf')
    stream_max = float('-inf')

    cap = cv2.VideoCapture(cam_index)
    win_name = f"Calibration - {side_name.upper()}"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, click_event)
    cv2.createTrackbar("Fluid Delay (ms)", win_name, current_delay, 5000, on_delay_change)

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Real Parsing Logic
        stream_min, stream_max = parse_serial_stream(side_name, stream_min, stream_max)

        h, w = frame.shape[:2]
        canvas = np.zeros((h + 100, w, 3), dtype=np.uint8)
        canvas[0:h, 0:w] = frame

        # Draw UI Elements
        cv2.rectangle(canvas, (0, h), (w, h+100), (30, 30, 30), -1)
        cv2.putText(canvas, f"DELAY: {current_delay}ms", (w-200, 40), 1, 1.2, (255, 200, 0), 2)
        for name, (x1, y1, x2, y2) in buttons.items():
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 60, 60), -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (200, 200, 200), 2)
            cv2.putText(canvas, name, (x1+10, y1+35), 1, 1.2, (255, 255, 255), 2)

        # Show captured limits from stream
        t_display_min = stream_min if stream_min != float('inf') else 0.0
        t_display_max = stream_max if stream_max != float('-inf') else 0.0
        cv2.putText(canvas, f"STREAM: {t_display_min:.1f} to {t_display_max:.1f}", 
                    (20, 40), 1, 1.5, (0, 255, 0), 2)

        for p in arc_pixels:
            cv2.circle(canvas, p, 4, (0, 255, 0), -1)

        if len(arc_pixels) >= 3:
            result = ransac_fit_circle(arc_pixels)
            if result:
                xc, yc, r = result
                angles = [np.arctan2(p[1]-yc, p[0]-xc) for p in arc_pixels]
                actual_min_a, actual_max_a = min(angles), max(angles)
                
                start_a, end_a = (actual_max_a, actual_min_a) if swap_extremes else (actual_min_a, actual_max_a)

                # Draw Curve
                curve_pts = [ [int(xc + r * np.cos(t)), int(yc + r * np.sin(t))] 
                             for t in np.linspace(actual_min_a, actual_max_a, 100) ]
                cv2.polylines(canvas, [np.array(curve_pts, np.int32)], False, (255, 255, 0), 3)

                # Label mapping
                p_start = arc_pixels[angles.index(start_a)]
                p_end = arc_pixels[angles.index(end_a)]
                cv2.putText(canvas, f"MIN: {t_display_min:.1f}", (p_start[0], p_start[1]-10), 1, 1, (0,0,255), 2)
                cv2.putText(canvas, f"MAX: {t_display_max:.1f}", (p_end[0], p_end[1]-10), 1, 1, (255,0,0), 2)

        # Handle Events
        if btn_triggered == "SWAP":
            swap_extremes = not swap_extremes
            btn_triggered = None
        elif btn_triggered == "RESET":
            arc_pixels = []
            stream_min, stream_max = float('inf'), float('-inf')
            btn_triggered = None
        elif btn_triggered == "SAVE" and len(arc_pixels) >= 3:
            with open(f"{side_name}_calibration.txt", "w") as f:
                f.write(f"{xc},{yc},{r},{start_a},{end_a},{t_display_min},{t_display_max}")
            with open("fluid_config.txt", "w") as f:
                f.write(str(current_delay))
            print(f"Saved! Delay set to {current_delay}ms")
            btn_triggered = None
            break
        elif btn_triggered == "QUIT":
            if ser: ser.close()
            cap.release(); cv2.destroyAllWindows(); sys.exit()

        cv2.imshow(win_name, canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyWindow(win_name)

if __name__ == "__main__":
    run_calibration(0, "left")
    run_calibration(1, "right")
    if ser: ser.close()