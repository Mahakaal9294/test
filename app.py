import cv2
import mediapipe as mp
import av
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import streamlit as st

st.title("Real-Time Hand Landmark Detection with WebRTC")

class HandLandmarkTransformer(VideoTransformerBase):
    def __init__(self):
        # Initialize MediaPipe Hands.
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert frame to a NumPy array in BGR format.
        img = frame.to_ndarray(format="bgr24")
        # Convert BGR to RGB.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the image and detect hand landmarks.
        results = self.hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the image.
                self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        # Return the annotated frame.
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start the WebRTC streamer with our custom transformer.
webrtc_streamer(key="hand_landmark", video_transformer_factory=HandLandmarkTransformer)
