import cv2
import mediapipe as mp

def main():
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