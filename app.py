import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

st.header("Live Hand Landmark Detection Debug")

# Transformer class for processing video frames.
class HandLandmarkTransformer(VideoTransformerBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.frame_count = 0

    def transform(self, frame):
        self.frame_count += 1
        # Log every 30 frames to see if processing is happening.
        if self.frame_count % 30 == 0:
            st.write(f"Processing frame {self.frame_count}")
        
        # Convert the frame to a NumPy array in BGR format.
        img = frame.to_ndarray(format="bgr24")
        
        # Convert BGR to RGB for MediaPipe processing.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        # Draw hand landmarks if detected.
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        return img

# Start the WebRTC streamer with explicit video constraints.
webrtc_streamer(
    key="hand-detection",
    video_transformer_factory=HandLandmarkTransformer,
    media_stream_constraints={"video": True, "audio": False},
)
