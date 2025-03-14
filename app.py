import streamlit as st
import cv2
import mediapipe as mp
import time

st.title("Live Hand Landmark Detection")

# A checkbox to start/stop the camera.
start_camera = st.checkbox("Start Camera", value=False)

if start_camera:
    # Initialize video capture from the default webcam.
    cap = cv2.VideoCapture(0)
    # Create a placeholder in the Streamlit app for the video frames.
    frame_placeholder = st.empty()

    # Initialize MediaPipe Hands and the drawing utilities.
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam. Exiting...")
                break

            # Flip the frame horizontally for a mirror-effect.
            frame = cv2.flip(frame, 1)

            # Convert the frame to RGB for MediaPipe processing.
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # Draw hand landmarks if detected.
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display the resulting frame.
            frame_placeholder.image(frame, channels="BGR")

            # A short delay to yield control and make the video smoother.
            time.sleep(0.03)
    cap.release()
