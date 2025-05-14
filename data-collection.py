import cv2
import os
import numpy as np
from config import ACTIONS, DATA_PATH, SEQUENCE_LENGTH
from mediapipe_detection import mediapipe_detection, extract_keypoints, draw_styled_landmarks, mp_holistic

def collect_data():
    cap = cv2.VideoCapture(1)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        # NEW LOOP
        # Loop through actions
        for action in ACTIONS:
            action_dir = os.path.join(DATA_PATH, action)
            if not os.path.exists(action_dir):
                os.makedirs(action_dir)

            # Loop through sequences aka videos
            for sequence in range(SEQUENCE_LENGTH, 2*SEQUENCE_LENGTH- 1):
                sequence_dir = os.path.join(action_dir, str(sequence))
                os.makedirs(sequence_dir, exist_ok=True)  # Ensure directory exists or create it
                # Loop through video length aka sequence length
                print(f"Collecting data for action: {action}, sequence: {30}")
                for frame_num in range(SEQUENCE_LENGTH):

                    # Read feed
                    ret, frame = cap.read()

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    draw_styled_landmarks(image, results)
                    
                    # NEW Apply wait logic
                    if frame_num == 0: 
                        cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(500)
                    else: 
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                    
                    # NEW Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(sequence_dir, str(frame_num) + ".npy")
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_data()
