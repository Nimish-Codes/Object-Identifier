import os
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

# Set the environment variable to disable GUI libraries for OpenCV
os.environ['OPENCV_IO_ENABLE_JASPER'] = '0'

# Explicitly import cv2 after setting the environment variable
try:
    import cv2
except ImportError:
    st.write("Error importing cv2. Make sure OpenCV is installed and try restarting the application.")

# Load a pre-trained MobileNetV2 model trained on ImageNet
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Load ImageNet class labels
imagenet_labels_path = tf.keras.utils.get_file('imagenet_labels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
with open(imagenet_labels_path) as f:
    imagenet_labels = f.readlines()
imagenet_labels = [label.strip() for label in imagenet_labels]

# Function to preprocess an image
def preprocess_image(image):
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    image = tf.image.resize(image, (224, 224))  # Resize the image
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)  # Preprocess the image
    image = image[tf.newaxis, ...]  # Add batch dimension
    return image

# Function to perform object detection
def detect_objects(image):
    # Preprocess the image
    input_image = preprocess_image(image)

    # Perform object detection
    predictions = model(input_image)

    return predictions

def main():
    st.title("Object Detection with MobileNetV2")

    # Get user input for the image file
    uploaded_file = st.file_uploader("Choose an image...")

    if uploaded_file is not None:
        # Load the image
        user_image = Image.open(uploaded_file)

        # Explicitly check if cv2 was imported successfully
        try:
            image_np = cv2.cvtColor(np.array(user_image), cv2.COLOR_RGB2BGR)
        except NameError:
            st.write("Error importing cv2. Make sure OpenCV is installed and try restarting the application.")
            return

        # Convert image to uint8
        image_np_uint8 = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # Perform object detection
        predictions = detect_objects(image_np_uint8)

        # Display the image along with the top predicted class
        st.image(image_np, caption="Uploaded Image", use_column_width=True)

        # Get the top predicted class and its confidence
        predicted_class_index = np.argmax(predictions.numpy())
        confidence = predictions.numpy()[0, predicted_class_index]
        predicted_class_name = imagenet_labels[predicted_class_index]

        # Set a confidence threshold
        confidence_threshold = 0.5  # You can adjust this threshold as needed

        # Check if the confidence is above the threshold
        if confidence >= confidence_threshold:
            st.write(f"This is an image of '{predicted_class_name}', [matching {confidence:.2f} (in scale of 0 to 1)]")
        else:
            st.write("Not found in dataset")

if __name__ == "__main__":
    main()

st.write("Note: Expect better result by uploading better quality image")
