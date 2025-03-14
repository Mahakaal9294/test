import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

st.header("Live Hand Landmark Detection")

# Transformer class for processing video frames
class HandLandmarkTransformer(VideoTransformerBase):
    def __init__(self):
        # Initialize MediaPipe Hands.
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        st.write("Initialized hand detection transformer.")

    def transform(self, frame):
        # Convert the frame to a numpy array (BGR format)
        img = frame.to_ndarray(format="bgr24")
        # Convert BGR to RGB for MediaPipe processing.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        # If hands are detected, draw landmarks on the frame.
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        return img

# Start the WebRTC streamer.
webrtc_streamer(
    key="hand-detection",
    video_transformer_factory=HandLandmarkTransformer,
    media_stream_constraints={"video": True, "audio": False},
)
