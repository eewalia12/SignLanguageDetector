import mediapipe as mp
import os
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data_path = "./alphabet_dir"

for dir in os.listdir(data_path)[1:]:
    for img_path in os.listdir(os.path.join(data_path, dir))[1]:
        img_path = img_path + ".jpg"
        img = cv2.imread(os.path.join(data_path, dir, img_path))
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img_process = hands.process(img_rgb)
            if img_process.multi_hand_landmarks:
                for landmark in img_process.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        img_rgb,
                        landmark,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
            plt.figure()
            plt.imshow(img_rgb)

plt.show()
