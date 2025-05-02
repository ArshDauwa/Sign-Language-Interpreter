import pickle

from sklearn.ensemble import RandomForestClassifier  # Importing the Random Forest Classifier
from sklearn.model_selection import train_test_split  # For splitting the data into training and testing sets
from sklearn.metrics import accuracy_score  # To calculate the accuracy of the model
import numpy as np  # For numerical operations

# Load the preprocessed dataset from a pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Convert the data and labels from the pickle file into numpy arrays for easier manipulation
data = np.asarray(data_dict['data'])  # Feature data
labels = np.asarray(data_dict['labels'])  # Corresponding labels (ASL letters)

# Split the data into training and testing sets (80% training, 20% testing)
# shuffle=True ensures the data is shuffled before splitting, stratify=labels ensures the class distribution is maintained
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize the Random Forest Classifier model
model = RandomForestClassifier()

# Train the model using the training data
model.fit(x_train, y_train)

# Use the trained model to predict labels for the test data
y_predict = model.predict(x_test)

# Calculate the accuracy of the model by comparing the predicted labels to the actual labels
score = accuracy_score(y_predict, y_test)

# Print the accuracy as a percentage
print('{}% of samples were classified correctly !'.format(score * 100))

# Save the trained model to a pickle file for future use (so we don't have to retrain it every time)
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)  # Dump the model into the file
f.close()  # Close the file
