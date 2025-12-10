import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# --- 1. SETUP ---
# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Initialize drawing utility
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

# Get screen dimensions for mouse mapping
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
print(f"Screen Size: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")

# Define the region of interest (ROI) and smoothing parameters
SMOOTHING_FACTOR = 5  # Higher value means smoother, but slower, movement
FRAME_REDUCTION = 50 # Reduces the capture frame size to define a movement area

# Variables for smoothing (Exponentially Weighted Moving Average)
prev_x, prev_y = 0, 0

# --- Click Detection Threshold ---
# This value determines how close the thumb and index finger must be to trigger a click.
# You may need to tune this value based on your webcam and hand size.
CLICK_THRESHOLD = 0.05

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Mirror the image for intuitive control
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and find hands
    results = hands.process(image_rgb)

    # Get frame dimensions
    h, w, _ = image.shape
    
    # Draw a visual box for the active control area
    # This helps stabilize the mouse; movement is only mapped within this central box.
    cv2.rectangle(image, 
                  (FRAME_REDUCTION, FRAME_REDUCTION), 
                  (w - FRAME_REDUCTION, h - FRAME_REDUCTION), 
                  (255, 0, 0), 2)
    
    # --- 2. LOGIC: TRACKING AND MAPPING ---
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks on the image 
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get landmarks for Index Finger Tip (ID 8) and Thumb Tip (ID 4)
            # Normalized coordinates (0 to 1)
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            
            # Convert normalized index tip coordinates to pixel coordinates (for visualization)
            ix = int(index_tip.x * w)
            iy = int(index_tip.y * h)
            
            # --- CURSOR MOVEMENT ---
            
            # Check if the index finger is within the designated movement area (ROI)
            if FRAME_REDUCTION < ix < w - FRAME_REDUCTION and FRAME_REDUCTION < iy < h - FRAME_REDUCTION:
                
                # Normalize coordinates to the ROI (0 to 1)
                norm_x = (ix - FRAME_REDUCTION) / (w - 2 * FRAME_REDUCTION)
                norm_y = (iy - FRAME_REDUCTION) / (h - 2 * FRAME_REDUCTION)

                # Map normalized coordinates to screen resolution
                target_x = np.interp(norm_x, (0, 1), (0, SCREEN_WIDTH))
                target_y = np.interp(norm_y, (0, 1), (0, SCREEN_HEIGHT))

                # Apply Smoothing (Exponentially Weighted Moving Average)
                # This makes the cursor movement less jittery.
                smooth_x = prev_x + (target_x - prev_x) / SMOOTHING_FACTOR
                smooth_y = prev_y + (target_y - prev_y) / SMOOTHING_FACTOR

                # --- 3. CONTROL: MOUSE MOVE ---
                pyautogui.moveTo(smooth_x, smooth_y)
                
                # Update previous coordinates for the next frame
                prev_x, prev_y = smooth_x, smooth_y
                
                # Draw a highlight circle on the index tip
                cv2.circle(image, (ix, iy), 10, (0, 255, 0), cv2.FILLED)
            
            # --- CLICK DETECTION ---
            
            # Calculate the Euclidean distance between thumb and index tips
            # The distance is calculated in normalized coordinates (0 to 1).
            distance = np.sqrt((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)
            
            if distance < CLICK_THRESHOLD:
                # --- 3. CONTROL: MOUSE CLICK ---
                pyautogui.click()
                print("Click detected!")
                
                # Flash a message on the screen
                cv2.putText(image, 'CLICK!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


# Display the resulting image
    cv2.imshow('Hand Gesture Mouse Control', image)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- Clean up ---
cap.release()
cv2.destroyAllWindows()