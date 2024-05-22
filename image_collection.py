import os
import cv2

# creates a directory called alphabet_dir in your local dir if it doesn't exist
DATA_DIR = './alphabet_dir2'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# labels for each directory/set of images
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# starts image capture with the default camera (0)
capture = cv2.VideoCapture(0)

# loops through each letter
for letter in alphabet:
    # creates a directory for each letter if it doesn't exist
    letter_dir = os.path.join(DATA_DIR, letter)
    if not os.path.exists(letter_dir):
        os.makedirs(letter_dir)

    # print which letter data is being collected for
    print(f'Collecting data for letter {letter}')

    counter = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        preview_frame = frame.copy()
        cv2.putText(preview_frame, f'Collecting {letter} - Press "Q" to capture, "Space" for next letter', 
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(25)
        if key == ord('q'):
            # capture image
            cv2.imwrite(os.path.join(letter_dir, f'{counter}.jpg'), frame)
            counter += 1
            print(f'Captured image {counter} for letter {letter}')
        elif key == 32:  # Space bar key
            # move to the next letter
            break

capture.release()
cv2.destroyAllWindows()