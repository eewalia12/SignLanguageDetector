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

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    padding = 50  # Padding value in pixels

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_map = hands.process(frame_rgb)
        if hand_map.multi_hand_landmarks:
            for hand_landmark in hand_map.multi_hand_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmark.landmark])
                
                # Calculate the center of the landmarks
                x_center = int(np.mean(landmarks[:, 0]) * frame.shape[1])
                y_center = int(np.mean(landmarks[:, 1]) * frame.shape[0])
                
                # Calculate the bounding box size (square around the center with padding)
                box_size = max(int(np.max(landmarks[:, 0]) * frame.shape[1] - np.min(landmarks[:, 0]) * frame.shape[1]),
                               int(np.max(landmarks[:, 1]) * frame.shape[0] - np.min(landmarks[:, 1]) * frame.shape[0])) + 2 * padding
                
                half_box_size = box_size // 2
                
                # Calculate the bounding box coordinates
                x_min = x_center - half_box_size
                x_max = x_center + half_box_size
                y_min = y_center - half_box_size
                y_max = y_center + half_box_size

                # Ensure the bounding box is within frame bounds
                x_min = max(0, x_min)
                x_max = min(frame.shape[1], x_max)
                y_min = max(0, y_min)
                y_max = min(frame.shape[0], y_max)

                # Crop the frame
                cropped_frame = frame[y_min:y_max, x_min:x_max]

                # Normalize the landmarks for prediction
                normalized_landmarks = normalize_landmarks_array(landmarks.flatten())
                if isinstance(normalized_landmarks, pd.DataFrame):
                    normalized_landmarks = normalized_landmarks.to_numpy()
                normalized_landmarks = normalized_landmarks.reshape(1, -1)  # Ensure 2D shape for prediction
                prediction = model.predict(normalized_landmarks)
                confidece =  round(model.predict_proba(normalized_landmarks).max(), 3)*100

                # Display the prediction on the cropped frame
                cv2.putText(cropped_frame, f'Label: {prediction[0]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(cropped_frame, f'Confidence: {confidece}%', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                # Draw landmarks on the original frame
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmark,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Draw the bounding box on the original frame
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        cv2.imshow('Cropped Frame', cropped_frame if 'cropped_frame' in locals() else frame)
        cv2.imshow('Original Frame', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()  