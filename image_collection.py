import os
import cv2

# creates a directory called alphabet_dir in your local dir if it doesn't exist
DATA_DIR = './alphabet_dir'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# labels for each directory/set of images
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# specify number of images for each letter
num_images = 20

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

    # wait for Q to be pressed
    while True:
        ret, frame = capture.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # once q is pressed, capture num_images images for that letter
    counter = 0
    while counter < num_images:
        ret, frame = capture.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(letter_dir, f'{counter}.jpg'), frame)
        counter += 1

capture.release()
cv2.destroyAllWindows()