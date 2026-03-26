import jetson_inference
import jetson_utils
import cv2
import numpy as np

# 1. Load the DepthNet model
# "fcn-mobilenet" is the default and runs great on Orin Nano
net = jetson_inference.depthNet("fcn-mobilenet")

# 2. Setup OpenCV Camera
cap = cv2.VideoCapture(0)

print("DepthNet + OpenCV Started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Step A: Convert BGR to CUDA RGBA ---
    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    cuda_mem = jetson_utils.cudaFromNumpy(frame_rgba)

    # --- Step B: Allocate Depth Output ---
    # We create a new CUDA buffer to hold the colorized depth map
    depth_cuda = jetson_utils.cudaAllocMapped(width=cuda_mem.width, 
                                              height=cuda_mem.height, 
                                              format=cuda_mem.format)

    # --- Step C: Process Depth ---
    # This computes the depth and colorizes it into 'depth_cuda'
    net.Process(cuda_mem, depth_cuda, colormap="viridis-inverted")

    # --- Step D: Convert back to OpenCV ---
    depth_numpy = jetson_utils.cudaToNumpy(depth_cuda)
    depth_frame = cv2.cvtColor(depth_numpy, cv2.COLOR_RGBA2BGR)

    # Optional: Stack original frame and depth side-by-side
    combined = np.hstack((frame, depth_frame))

    cv2.imshow("Depth Estimation (OpenCV)", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
