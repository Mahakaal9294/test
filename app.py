import cv2
import mediapipe as mp
import av
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Inject custom CSS to enlarge the video element.
st.markdown(
    """
    <style>
    /* Force all video elements to be larger */
    video {
        width: 100% !important;
        height: 600px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Real-Time Hand Landmark Detection with WebRTC")

class HandLandmarkTransformer(VideoTransformerBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert frame to NumPy array in BGR format.
        img = frame.to_ndarray(format="bgr24")
        # Flip the image horizontally for a mirror effect.
        img = cv2.flip(img, 1)
        # Convert BGR to RGB for MediaPipe.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    # Optionally, you can adjust drawing specs here
                    landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_hand_connections_style()
                )
        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="hand_landmark",
    video_transformer_factory=HandLandmarkTransformer,
    # You can also try setting video_html_attrs, but the CSS injection often overrides defaults.
    video_html_attrs={"style": {"width": "100%", "height": "600px"}}
)
