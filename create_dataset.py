import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Initialize MediaPipe Hand module for hand landmark detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Create a MediaPipe Hands object with specified configurations
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory containing the images for the dataset
DATA_DIR = './data'

# Lists to store the feature data and corresponding labels
data = []
labels = []

# Loop through all subdirectories (one for each letter of the ASL alphabet)
for dir_ in os.listdir(DATA_DIR):
    # Loop through all images in each subdirectory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Temporary list to store hand landmark data for the current image

        x_ = []  # List to store x-coordinates of the hand landmarks
        y_ = []  # List to store y-coordinates of the hand landmarks

        # Read the image and convert it from BGR to RGB (since MediaPipe expects RGB input)
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to detect hand landmarks
        results = hands.process(img_rgb)

        # If hands are detected in the image
        if results.multi_hand_landmarks:
            # Loop through each hand detected
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract x, y coordinates for each landmark (21 landmarks for each hand)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    # Append the x, y coordinates to the respective lists
                    x_.append(x)
                    y_.append(y)

                # Normalize the coordinates by subtracting the minimum x and y values
                # This helps in standardizing the position and scale of the hand in the image
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))  # Normalize x-coordinate
                    data_aux.append(y - min(y_))  # Normalize y-coordinate

            # If the data for this image contains exactly 42 features (21 landmarks × 2 coordinates)
            # Then append the processed data and its corresponding label (the directory name, which represents the letter)
            if len(data_aux) == 42:  # 21 hand landmarks × (x, y) coordinates
                data.append(data_aux)
                labels.append(dir_)  # The directory name represents the letter label (A, B, C, etc.)

# After processing all images, save the data and labels to a pickle file
# This allows us to easily load the dataset in the future
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
