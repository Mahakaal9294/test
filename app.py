import streamlit as st
import cv2
import mediapipe as mp
import time

st.title("Real-Time Hand Landmark Detection")

# Toggle to start/stop the video stream using session state.
if 'run' not in st.session_state:
    st.session_state['run'] = False

def toggle_run():
    st.session_state['run'] = not st.session_state['run']

if st.button("Start/Stop Video"):
    toggle_run()

# Placeholder for the video frames.
frame_placeholder = st.empty()

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# Open the webcam.
cap = cv2.VideoCapture(0)

# Process video stream while the toggle is enabled.
while st.session_state['run']:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture video from the camera.")
        break

    # Flip the frame horizontally for a selfie-view.
    frame = cv2.flip(frame, 1)
    # Convert the image from BGR (OpenCV default) to RGB.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hand landmarks.
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame with landmarks.
    frame_placeholder.image(frame_rgb, channels="RGB")

    # Small delay to yield control and reduce CPU usage.
    time.sleep(0.03)

cap.release()
