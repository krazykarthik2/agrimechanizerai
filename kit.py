import jetson_inference
import jetson_utils
import cv2

# 1. Load YOLOv4-Tiny (Pre-trained on COCO which includes humans/persons)
# The first time you run this, it will download and take ~5 mins to optimize.

net = jetson_inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
# 2. Setup OpenCV Camera (USB = 0)
cap = cv2.VideoCapture(0)

print("YOLO Human Detection Started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Step A: Convert OpenCV BGR to CUDA RGBA ---
    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    cuda_mem = jetson_utils.cudaFromNumpy(frame_rgba)

    # --- Step B: Run Detection ---
    detections = net.Detect(cuda_mem)

    # --- Step C: Filter for Humans only ---
    # ClassID 0 in COCO/YOLO is 'person'
    human_count = 0
    for det in detections:
        if det.ClassID == 0:  # 0 is the ID for 'person'
            human_count += 1

    # --- Step D: Display Back in OpenCV ---
    output_numpy = jetson_utils.cudaToNumpy(cuda_mem)
    output_frame = cv2.cvtColor(output_numpy, cv2.COLOR_RGBA2BGR)

    # Add a counter on the screen using OpenCV
    cv2.putText(output_frame, f"Humans: {human_count}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Human Detection (YOLOv4-Tiny)", output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
