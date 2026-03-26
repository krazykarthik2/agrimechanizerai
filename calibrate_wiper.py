import cv2
import numpy as np
import random
import tkinter as tk
from tkinter import simpledialog

# --- 1. GUI CONFIG ---
def get_theta_limits():
    root = tk.Tk()
    root.withdraw()
    t_min = simpledialog.askfloat("Input", "Wiper Start Angle (Degrees):", initialvalue=10.0)
    t_max = simpledialog.askfloat("Input", "Wiper End Angle (Degrees):", initialvalue=130.0)
    return t_min, t_max

THETA_MIN, THETA_MAX = get_theta_limits()
arc_pixels = []
swap_extremes = False

def click_event(event, x, y, flags, param):
    global arc_pixels
    if event == cv2.EVENT_LBUTTONDOWN:
        arc_pixels.append((x, y))

def get_circle_3p(p1, p2, p3):
    """Algebraic solution for a circle passing through 3 points."""
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3
    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(D) < 1e-6: return None # Points are colinear (a straight line)
    
    xc = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / D
    yc = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / D
    r = np.sqrt((x1 - xc)**2 + (y1 - yc)**2)
    return xc, yc, r

def ransac_fit_circle(pts, iterations=100, threshold=2.0):
    """Uses RANSAC to find the best circle among noisy points."""
    if len(pts) < 3: return None
    best_circle = None
    max_inliers = -1

    for _ in range(iterations):
        sample = random.sample(pts, 3)
        circle = get_circle_3p(sample[0], sample[1], sample[2])
        if circle is None: continue
        
        xc, yc, r = circle
        # Count points that lie on this circle's edge
        inliers = 0
        for p in pts:
            dist = abs(np.sqrt((p[0]-xc)**2 + (p[1]-yc)**2) - r)
            if dist < threshold:
                inliers += 1
        
        if inliers > max_inliers:
            max_inliers = inliers
            best_circle = circle
            
    return best_circle

cap = cv2.VideoCapture(0)
cv2.namedWindow("RANSAC Curve Fit")
cv2.setMouseCallback("RANSAC Curve Fit", click_event)

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w = frame.shape[:2]

    cv2.putText(frame, f"Theta: {THETA_MIN} - {THETA_MAX}", (20, 40), 1, 1.5, (0, 255, 255), 2)
    cv2.putText(frame, "T: Swap | S: Save | R: Reset", (20, 75), 1, 1.1, (255, 255, 255), 1)

    for p in arc_pixels:
        cv2.circle(frame, p, 4, (0, 255, 0), -1)

    if len(arc_pixels) >= 3:
        result = ransac_fit_circle(arc_pixels)
        if result:
            xc, yc, r = result
            
            # Find angles of clicks to define the visible segment
            angles = [np.arctan2(p[1]-yc, p[0]-xc) for p in arc_pixels]
            min_a, max_a = min(angles), max(angles)
            
            # Determine start/end points based on 'T' key
            start_a, end_a = (max_a, min_a) if swap_extremes else (min_a, max_a)

            # Draw the curve
            curve_pts = []
            for t in np.linspace(min_a, max_a, 200):
                px, py = int(xc + r * np.cos(t)), int(yc + r * np.sin(t))
                if -2000 < px < 4000 and -2000 < py < 4000:
                    curve_pts.append([px, py])
            
            if len(curve_pts) > 1:
                cv2.polylines(frame, [np.array(curve_pts, np.int32).reshape((-1, 1, 2))], 
                             False, (255, 255, 0), 3)

            # Draw labels on the extreme clicks
            p_start = arc_pixels[angles.index(start_a)]
            p_end = arc_pixels[angles.index(end_a)]
            
            cv2.circle(frame, p_start, 10, (0, 0, 255), 2) # Red = MIN
            cv2.putText(frame, f"{THETA_MIN}deg", (p_start[0], p_start[1]-20), 1, 1.2, (0, 0, 255), 2)
            
            cv2.circle(frame, p_end, 10, (255, 0, 0), 2) # Blue = MAX
            cv2.putText(frame, f"{THETA_MAX}deg", (p_end[0], p_end[1]-20), 1, 1.2, (255, 0, 0), 2)

    cv2.imshow("RANSAC Curve Fit", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('t'): swap_extremes = not swap_extremes
    elif key == ord('r'): arc_pixels = []
    elif key == ord('s') and len(arc_pixels) >= 3:
        with open("wiper_final.txt", "w") as f:
            f.write(f"{xc},{yc},{r},{start_a},{end_a},{THETA_MIN},{THETA_MAX}")
        print("Final Calibration Saved!")
        break
    elif key == ord('q'): break

cap.release()
cv2.destroyAllWindows()