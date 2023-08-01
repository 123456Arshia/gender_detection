import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset parameters
data_directory = 'file location'
img_height, img_width = 150, 150
batch_size = 32

# Load and preprocess images using Keras ImageDataGenerator
data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Create a data generator for training data
train_generator = data_generator.flow_from_directory(
    data_directory,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'  # Subset of the data used for training
)

# Create a data generator for validation data
validation_generator = data_generator.flow_from_directory(
    data_directory,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'  # Subset of the data used for validation
)

# Create the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Get the number of neurons in the first convolutional layer
first_conv_layer_neurons = model.layers[0].get_weights()[1].shape[0]

# Function to draw the number of neurons in the first convolutional layer
def draw_neurons(num_neurons):
    # ... (implementation not provided)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
num_train_samples = len(train_generator.filenames)
num_validation_samples = len(validation_generator.filenames)
num_epochs = 10

model.fit(
    train_generator,
    steps_per_epoch=num_train_samples // batch_size,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=num_validation_samples // batch_size
)

# Save the trained model
model.save('gender_detector_model.h5')
print("Training complete! Model saved as gender_detector_model.h5.")

from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('gender_detector_model.h5')

# Load and preprocess the new image
img = cv2.imread('file location')
img = cv2.resize(img, (img_height, img_width))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Make gender prediction
prediction = model.predict(img)

# Convert the prediction to a human-readable label (e.g., 'male' or 'female')
gender_label = 'female' if prediction[0][0] < 0.5 else 'male'
print(f"Gender prediction: {gender_label}")

