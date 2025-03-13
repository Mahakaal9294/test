import cv2
import mediapipe as mp
import av
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import streamlit as st

st.title("Real-Time Hand Landmark Detection with WebRTC")

class HandLandmarkTransformer(VideoTransformerBase):
    def __init__(self):
        # Initialize MediaPipe Hands with additional tracking confidence.
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        # Use default drawing styles for better visibility.
        self.landmark_style = mp.solutions.drawing_styles.get_default_hand_landmarks_style()
        self.connection_style = mp.solutions.drawing_styles.get_default_hand_connections_style()

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert frame to a NumPy array in BGR format.
        img = frame.to_ndarray(format="bgr24")
        # Flip the image horizontally (mirror effect).
        img = cv2.flip(img, 1)
        # Convert to RGB for MediaPipe processing.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.landmark_style,
                    connection_drawing_spec=self.connection_style
                )
        # Return the annotated frame.
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Configure the WebRTC streamer with custom video HTML attributes to enlarge the video box.
webrtc_streamer(
    key="hand_landmark",
    video_transformer_factory=HandLandmarkTransformer,
    video_html_attrs={"style": {"width": "720px", "height": "auto"}}
)
