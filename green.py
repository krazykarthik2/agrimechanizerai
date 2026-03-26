import cv2
import numpy as np

def main():
    # Use index 0 for default camera
    cap = cv2.VideoCapture(0)
    
    # --- Configuration ---
    # Polygons defined as list of (x, y) points.
    polygons = [
        np.array([(50, 100), (200, 100), (200, 300), (50, 300)], np.int32),   # Left (P1)
        np.array([(220, 100), (370, 100), (370, 300), (220, 300)], np.int32), # Center (P2)
        np.array([(390, 100), (540, 100), (540, 300), (390, 300)], np.int32)  # Right (P3)
    ] 
    
    # Green detection thresholds (HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Sensitivity: 5% of the polygon must be green to trigger
    GREEN_THRESHOLD = 0.05 

    print("Direct Detection Started.")
    print("Press 'q' or 'ESC' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Mirror frame for intuitive interaction
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Convert to HSV for color masking
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_all_green = cv2.inRange(hsv, lower_green, upper_green)
        
        active_states = [False, False, False]
        
        # --- Real-Time Detection Logic ---
        for i, poly in enumerate(polygons):
            # Create a black mask the size of the frame
            poly_mask = np.zeros((h, w), dtype=np.uint8)
            # Fill the specific polygon area with white
            cv2.fillPoly(poly_mask, [poly], 255)
            
            # Find green pixels only within that polygon
            region_green = cv2.bitwise_and(mask_all_green, mask_all_green, mask=poly_mask)
            
            total_pixels = cv2.countNonZero(poly_mask)
            green_pixels = cv2.countNonZero(region_green)
            
            detect_ratio = 0.0
            if total_pixels > 0:
                detect_ratio = green_pixels / total_pixels
            
            # If green is found, set state to True
            if detect_ratio > GREEN_THRESHOLD:
                active_states[i] = True
                # Print "Serial" output for integration
                print(f"TRIGGER: Polygon P{i+1} ACTIVE ({detect_ratio:.1%})")

        # --- Drawing Overlay ---
        draw_img = frame.copy()
        
        # Draw 3 Status Dots in the center of the screen
        dot_y = h - 50
        center_x = w // 2
        spacing = 80
        
        for i in range(3):
            dx = (i - 1) * spacing
            cx = center_x + dx
            
            # Visual: Green if triggered, Red if empty
            color = (0, 255, 0) if active_states[i] else (0, 0, 255)
            cv2.circle(draw_img, (cx, dot_y), 20, color, -1)
            cv2.putText(draw_img, f"P{i+1}", (cx-10, dot_y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw the Polygon outlines on the camera feed
        for i, poly in enumerate(polygons):
            # Box is Green if active, Blue if inactive
            line_color = (0, 255, 0) if active_states[i] else (255, 0, 0)
            cv2.polylines(draw_img, [poly], True, line_color, 2)

        cv2.imshow('Direct Green Detection', draw_img)
        
        # Exit handling
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
