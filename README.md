# Sign Language Interpreter

## Description
This project is a real-time Sign Language Interpreter that uses computer vision and machine learning to recognize American Sign Language (ASL) letters. By using your computer's webcam, the system detects hand gestures and predicts which ASL letter you are signing. The model is trained on a custom dataset and can recognize 24 letters of the ASL alphabet, excluding "J" and "Z" (due to their dynamic nature). The trained model is saved for future use, providing a seamless way to interpret ASL signs.

## Files 
- `README.md`: Provides guidance and instructions on how to use the Sign Language Interpreter, including setup, steps, and examples.
- `collect_imgs.py`: Collects and saves images of ASL gestures from the webcam to create a dataset for training the classifier.
- `create_dataset.py`: Processes collected hand gesture images and extracts features (hand landmarks) to create a dataset for model training.
- `train_classifier`: Trains a machine learning classifier (Random Forest) on the dataset to predict ASL hand gestures.
- `inference_classifier.py`: Runs real-time inference using the trained model to recognize and display predicted ASL letters from webcam input.

## Citation

### Data
The data for the Sign Language Interpreter was obtained by capturing images of hand gestures using a webcam. The process involved the following steps:

  1) Image Capture: Using the collect_imgs.py script, images of each ASL gesture were captured in real-time. For each letter of the ASL alphabet, multiple images were collected to      ensure a varied dataset.

  2) Gesture Variation: The data was gathered by performing the ASL gestures in different hand positions, angles, and lighting conditions, to account for variations in real-world       scenarios.

  3) Manual Labeling: Each image was manually labeled with the corresponding ASL letter, ensuring accurate training data for the machine learning model.

This method of data collection created a diverse set of images, which served as the foundation for training the ASL gesture recognition model.

Data folder also contains all the images we used to train our model.

## How to Use 

1. Clone the git repo using the following command <pre> git clone https://github.com/ArshDauwa/Sign-Language-Interpreter</pre> 
2. Run the `inference_classifier.py` script and begin signing

## Sign Language Alphabet

Refer to this image to see how to sign specific letters

![image](https://github.com/user-attachments/assets/0b7573cf-f1bd-434d-959b-fd10b9e8e55b)














