import jetson_inference
import jetson_utils
import sys

# 1. Load the DepthNet model
net = jetson_inference.depthNet("fcn-mobilenet", sys.argv)

# 2. Initialize Two Cameras at 640x480
# We pass the resolution as command-line style arguments to videoSource
cam_options = ["--input-width=640", "--input-height=480"]
cam_left = jetson_utils.videoSource("/dev/video0", argv=cam_options)
cam_right = jetson_utils.videoSource("/dev/video1", argv=cam_options)

# 3. Create a single Output Window
display = jetson_utils.videoOutput("display://0")

# Setup buffers for blending and side-by-side view
composite = None
left_blend = None
right_blend = None

print("Dual 640x480 Depth Blending Started. Press Ctrl+C to exit.")

while display.IsStreaming():
    # Capture from both cameras
    img_left = cam_left.Capture()
    img_right = cam_right.Capture()

    if img_left is None or img_right is None:
        continue

    # Initialize buffers based on the first frame captured
    if composite is None:
        w, h = img_left.width, img_left.height # Should be 640, 480
        # Buffer for the depth maps (colorized)
        depth_left = jetson_utils.cudaAllocMapped(width=w, height=h, format=img_left.format)
        depth_right = jetson_utils.cudaAllocMapped(width=w, height=h, format=img_right.format)
        # Buffers for the blended results
        left_blend = jetson_utils.cudaAllocMapped(width=w, height=h, format=img_left.format)
        right_blend = jetson_utils.cudaAllocMapped(width=w, height=h, format=img_right.format)
        # Master buffer (side-by-side): Width is 1280 (640*2), Height is 480
        composite = jetson_utils.cudaAllocMapped(width=w*2, height=h, format=img_left.format)

    # --- Step 1: Process Depth (No filter_mode) ---
    net.Process(img_left, depth_left)
    net.Process(img_right, depth_right)

    # --- Step 2: Blending (Overlay with Transparency) ---
    # First, copy the original camera frames into the blend buffers (Base Layer)
    jetson_utils.cudaOverlay(img_left, left_blend, 0, 0)
    jetson_utils.cudaOverlay(img_right, right_blend, 0, 0)
    
    # Second, overlay the depth map on top. 
    # jetson-inference/utils handles alpha if the colormap supports it.
    jetson_utils.cudaOverlay(depth_left, left_blend, 0, 0)
    jetson_utils.cudaOverlay(depth_right, right_blend, 0, 0)

    # --- Step 3: Side-by-Side Composition ---
    # Stamp the left blended result at (0, 0)
    jetson_utils.cudaOverlay(left_blend, composite, 0, 0)
    # Stamp the right blended result at (640, 0)
    jetson_utils.cudaOverlay(right_blend, composite, img_left.width, 0)

    # Render the master window
    display.Render(composite)
    display.SetStatus(f"Dual 640x480 | {net.GetNetworkFPS():.1f} FPS")
