import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

st.title("Live Hand Landmark Detection using Streamlit and MediaPipe")

# Define a video transformer that processes frames using MediaPipe Hands.
class HandLandmarkTransformer(VideoTransformerBase):
    def __init__(self):
        # Initialize MediaPipe Hands and the drawing utilities.
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def transform(self, frame):
        # Convert the incoming frame to a NumPy array in BGR format.
        img = frame.to_ndarray(format="bgr24")
        # Convert the BGR image to RGB for processing.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        # If hands are detected, draw landmarks and connections.
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        return img

# Use the webrtc_streamer to start capturing and processing the video.
webrtc_streamer(key="hand-detection", video_transformer_factory=HandLandmarkTransformer)
