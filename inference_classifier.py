import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model from the pickle file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize the webcam
cap = cv2.VideoCapture(0)

# MediaPipe setup for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the mapping of label indices to ASL letters
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
    10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P',
    15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U',
    20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

warned = False  # Flag to avoid printing multiple warning messages

while True:
    # Initialize empty lists for storing the hand landmark data
    data_aux = []
    x_coords, y_coords = [], []

    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        continue  # Skip the frame if it's not captured correctly

    # Get the height and width of the frame for scaling coordinates
    height, width, _ = frame.shape

    # Convert the frame to RGB (MediaPipe expects RGB input)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame using MediaPipe to detect hand landmarks
    results = hands.process(frame_rgb)

    # If hands are detected in the frame
    if results.multi_hand_landmarks:
        # Loop through all detected hands
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks and connections on the frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Use the first detected hand (in case there are multiple)
        hand_landmarks = results.multi_hand_landmarks[0]

        # Collect the x and y coordinates for the hand landmarks
        for lm in hand_landmarks.landmark:
            x_coords.append(lm.x)
            y_coords.append(lm.y)

        # Normalize the coordinates of the hand landmarks to the range [0,1]
        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min(x_coords))  # Normalizing x coordinates
            data_aux.append(lm.y - min(y_coords))  # Normalizing y coordinates

        # If we have exactly 42 features (21 x-y pairs), make a prediction
        if len(data_aux) == 42:
            # Predict the ASL character using the trained model
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Calculate the bounding box coordinates around the hand
            x1 = int(min(x_coords) * width) - 10
            y1 = int(min(y_coords) * height) - 10
            x2 = int(max(x_coords) * width) + 10
            y2 = int(max(y_coords) * height) + 10

            # Draw the rectangle around the hand
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

            # Display the predicted character near the hand
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            warned = False
        else:
            # Print a warning if the feature vector length is incorrect
            if not warned:
                print(f"⚠️ Skipped frame: got {len(data_aux)} features, expected 42.")
                warned = True

    # Display the frame with the annotations
    cv2.imshow('frame', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any open windows
cap.release()
cv2.destroyAllWindows()
