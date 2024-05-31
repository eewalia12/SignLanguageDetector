import cv2
import mediapipe as mp
import numpy as np
from data_normalization import normalize_landmarks_array
import pandas as pd
import pickle

def main():
    with open('hand_landmark_model.pkl', 'rb') as file:
        loaded_model_data = pickle.load(file)

    model = loaded_model_data['model']

    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence = 0.3)

    while True:
        ret, frame = cap.read()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_map = hands.process(frame_rgb)
        if hand_map.multi_hand_landmarks:
            for hand_landmark in hand_map.multi_hand_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmark.landmark]).flatten()
                normalized_landmarks = normalize_landmarks_array(landmarks)
                prediction = model.predict(normalized_landmarks)

                cv2.putText(frame, f'Label: {prediction[0]}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmark, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()