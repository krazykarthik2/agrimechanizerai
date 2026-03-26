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
CONFIDENCE_THRESHOLD = 0.35

# FIXED USB IDs (From your ls /dev/v4l/by-id command)
CAM_A_ID = "/dev/v4l/by-id/usb-Ingenic_Semiconductor_Co._Ltd_HD_Web_Camera_1234567890-video-index0"
CAM_B_ID = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_Camera_SN0001-video-index0"

# Polygon coordinates
poly_points_1 = [(50, 200), (300, 200), (300, 450), (50, 450)]
poly_points_2 = [(340, 200), (600, 200), (600, 450), (340, 450)]

zone1 = Polygon(poly_points_1)
zone2 = Polygon(poly_points_2)

# --- 2. INITIALIZATION ---
try:
    ser = serial.Serial(SERIAL_PORT, baudrate=BAUD_RATE, timeout=0.1)
except Exception as e:
    print(f"Serial Error: {e}")
    ser = None

net = jetson_inference.detectNet("ssd-mobilenet-v2", threshold=CONFIDENCE_THRESHOLD)

# Helper to get video index from the /dev/v4l/by-id/ path
def get_video_index(dev_id_path):
    if not os.path.exists(dev_id_path):
        print(f"CRITICAL ERROR: Camera path not found: {dev_id_path}")
        return None
    # Resolves symlink (e.g., /dev/v4l/by-id/... -> /dev/video0)
    real_path = os.path.realpath(dev_id_path)
    # Extract the number from the end (e.g., '0' from '/dev/video0')
    try:
        return int(''.join(filter(str.isdigit, real_path.split('/')[-1])))
    except ValueError:
        return None

def open_camera_stable(path):
    idx = get_video_index(path)
    if idx is None:
        return None
    
    # Force V4L2 backend and set low-bandwidth settings to prevent timeouts
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    return cap

cap1 = open_camera_stable(CAM_A_ID)
cap2 = open_camera_stable(CAM_B_ID)

if cap1 is None or not cap1.isOpened():
    print("Failed to open Camera A (Ingenic)")
if cap2 is None or not cap2.isOpened():
    print("Failed to open Camera B (Sonix)")

def process_and_send(cap, zone, cam_id, ser_prefix):
    if cap is None or not cap.isOpened():
        return None, False

    ret, frame = cap.read()
    if not ret:
        return None, False

    rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    cuda_mem = jetson_utils.cudaFromNumpy(rgba)
    detections = net.Detect(cuda_mem)
    
    human_in_zone = False
    val = "90" # Default
    
    for det in detections:
        if net.GetClassDesc(det.ClassID).lower() == "person":
            person_box = box(det.Left, det.Top, det.Right, det.Bottom)
            if person_box.intersects(zone):
                human_in_zone = True
                cv2.rectangle(frame, (int(det.Left), int(det.Top)), 
                              (int(det.Right), int(det.Bottom)), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (int(det.Left), int(det.Top)), 
                              (int(det.Right), int(det.Bottom)), (255, 0, 0), 1)

    # SERIAL LOGIC: ;a,0; or ;a,90;
    if ser:
        val = "0" if human_in_zone else "90"
        message = f";{ser_prefix},{val};\n"
        ser.write(message.encode())

    # Drawing
    pts = np.array(list(zone.exterior.coords), np.int32)
    color = (0, 255, 0) if human_in_zone else (0, 0, 255)
    cv2.polylines(frame, [pts], True, color, 3)
    cv2.putText(frame, f"CAM {cam_id}: {val}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return frame, human_in_zone

while True:
    f1, hit1 = process_and_send(cap1, zone1, "Ingenic", "a")
    f2, hit2 = process_and_send(cap2, zone2, "Sonix", "b")

    if f1 is not None and f2 is not None:
        if f1.shape[0] != f2.shape[0]:
            f2 = cv2.resize(f2, (int(f2.shape[1] * (f1.shape[0] / f2.shape[0])), f1.shape[0]))
        canvas = np.hstack((f1, f2))
        cv2.imshow("Fixed Dual Detection (by-ID)", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
if ser: ser.close()
cv2.destroyAllWindows()
