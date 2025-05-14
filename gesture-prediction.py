import cv2
import numpy as np
import mediapipe as mp
from mediapipe_detection import mediapipe_detection, draw_styled_landmarks, extract_keypoints
from config import ACTIONS, DATA_PATH, SEQUENCE_LENGTH, LOG_DIR, MODEL_PATH
from tensorflow.keras.models import load_model
from scipy import stats

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

def predict_gesture():
    cap = cv2.VideoCapture(1)
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5
    colors = [(245,117,16), (117,245,16), (16,117,245)]

    # Load the model
    model = load_model(MODEL_PATH)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:]

            if len(sequence) == SEQUENCE_LENGTH:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                action = ACTIONS[np.argmax(res)]
                print(action)
                predictions.append(np.argmax(res))
                
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        if len(sentence) > 0: 
                            if action != sentence[-1]:
                                sentence.append(action)
                        else:
                            sentence.append(action)

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                image = prob_viz(res, ACTIONS, image, colors)
                
                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
            cv2.imshow('OpenCV Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_gesture()
