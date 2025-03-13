import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

st.title("Hand Landmark Detection with Camera Input")
st.write("Capture a photo using your camera to detect hand landmarks.")

# Use camera input to capture an image
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # Convert the image file buffer to a PIL Image
    image = Image.open(img_file_buffer)
    # Convert the PIL image to a NumPy array
    image_np = np.array(image)

    # If image has an alpha channel, convert it to RGB
    if image_np.shape[-1] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    # Display the captured image
    st.image(image_np, caption="Captured Image", use_column_width=True)

    # Initialize MediaPipe Hands module and drawing utilities
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    # Process the captured image to detect hand landmarks
    with mp_hands.Hands(static_image_mode=True,
                        max_num_hands=2,
                        min_detection_confidence=0.5) as hands:
        # MediaPipe requires an RGB image
        results = hands.process(image_np)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks and connections on the image
                mp_draw.draw_landmarks(image_np, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            st.image(image_np, caption="Processed Image with Hand Landmarks", use_column_width=True)
        else:
            st.write("No hand landmarks detected.")
