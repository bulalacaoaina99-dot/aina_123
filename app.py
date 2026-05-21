import subprocess
import sys

try:
    import tornado
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tornado"])

try:
    from twilio.rest import Client
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "twilio"])
    from twilio.rest import Client

import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import av
import cv2
import glob
import io
import zipfile
from datetime import datetime
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

st.set_page_config(
    page_title="📹 Live Object Detection & Tracing",
    layout="wide"
)

SAVE_DIR = "detection_logs"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# TWILIO CONFIG
# =========================

TWILIO_ACCOUNT_SID = "ACcd3c04d2fc8d40b43f6921f4d08b9403"
TWILIO_AUTH_TOKEN = "YOUR_AUTH_TOKEN"
TWILIO_PHONE_NUMBER = "+1234567890"
ALERT_PHONE_NUMBER = "+639123456789"

twilio_client = Client(
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN
)

if "gallery_mode" not in st.session_state:
    st.session_state.gallery_mode = False

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()
CLASS_NAMES = list(model.names.values())

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Deep Premium Midnight Background */
.stApp {
    background: radial-gradient(circle at top center, #1e1b4b 0%, #0f111a 70%, #07080d 100%);
    color: #f1f5f9;
}

/* Clean Professional Title */
.title {
    text-align: center;
    font-size: clamp(32px, 4.5vw, 52px);
    font-weight: 800;
    letter-spacing: -0.05em;
    background: linear-gradient(135deg, #a5b4fc, #6366f1, #e0a7ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2px;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 16px;
    letter-spacing: 0.02em;
    font-weight: 400;
    margin-bottom: 40px;
}

/* Sleek Glassmorphism Container */
.panel {
    background: rgba(15, 17, 26, 0.7);
    border: 1px solid rgba(99, 102, 241, 0.2);
    padding: 30px;
    border-radius: 24px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(20px);
}

/* Premium Button Styling */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    color: #ffffff;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 10px 24px;
    font-weight: 600;
    letter-spacing: -0.01em;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(99, 102, 241, 0.5);
    background: linear-gradient(135deg, #4f46e5, #4338ca);
    border-color: rgba(255, 255, 255, 0.2);
}

/* Download Button Styling */
.stDownloadButton > button {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25) !important;
}

.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #059669, #047857) !important;
    box-shadow: 0 8px 20px rgba(16, 185, 129, 0.4) !important;
}

/* Clean Sidebar */
section[data-testid="stSidebar"] {
    background: #090b11;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

.stSlider label,
.stSelectbox label,
.stToggle label {
    color: #cbd5e1 !important;
    font-weight: 500;
    font-size: 14px;
}

/* Rounded Clean Image Borders */
img {
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 10px 25px rgba(0,0,0,0.3);
}

.block-container {
    padding-top: 3rem;
}

#MainMenu, footer, header {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title">📹 Live Object Detection & Tracing</div>
<div class="subtitle">Real-Time AI Detection using YOLOv8 + Streamlit</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Detection Settings")

    confidence = st.slider(
        "Confidence Threshold",
        0.1,
        1.0,
        0.5,
        0.05
    )

    target_object = st.selectbox(
        "🚨 Alert Target",
        CLASS_NAMES
    )

    save_images = st.toggle(
        "📸 Save Detection",
        value=True
    )

    show_boxes = st.toggle(
        "🟦 Show Bounding Boxes",
        value=True
    )

class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.prev_objects = set()
        self.alert_sent = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        results = model.predict(
            img,
            conf=confidence,
            imgsz=480,
            verbose=False
        )

        detected_counts = {}
        current_objects = set()
        alert_detected = False

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = model.names.get(cls_id, "unknown")

                detected_counts[label] = detected_counts.get(label, 0) + 1
                current_objects.add(label)

                if label == target_object:
                    alert_detected = True

                if show_boxes:
                    # Match the Indigo design palette for bounding boxes
                    color = (241, 102, 99) # BGR value for sleek indigo hue

                    if label == target_object:
                        color = (0, 0, 255) # Red alert remains red

                    cv2.rectangle(
                        img,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2 # Sleeker line thickness
                    )

                    cv2.putText(
                        img,
                        f"{label}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

        # =========================
        # SEND TWILIO SMS ALERT
        # =========================
        if alert_detected and not self.alert_sent:
            try:
                twilio_client.messages.create(
                    body=f"🚨 ALERT: {target_object.upper()} detected by YOLOv8 system.",
                    from_=TWILIO_PHONE_NUMBER,
                    to=ALERT_PHONE_NUMBER
                )
                print("SMS Alert Sent!")
            except Exception as e:
                print("Twilio Error:", e)

            self.alert_sent = True

        if not alert_detected:
            self.alert_sent = False

        total_objects = sum(detected_counts.values())
        overlay = img.copy()

        cv2.rectangle(
            overlay,
            (10, 10),
            (350, 140),
            (15, 17, 26),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.85,
            img,
            0.15,
            0,
            img
        )

        cv2.putText(
            img,
            f"Total Objects: {total_objects}",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (241, 102, 99),
            2
        )

        y_position = 80
        for obj, count in detected_counts.items():
            cv2.putText(
                img,
                f"{obj}: {count}",
                (25, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            y_position += 30

        if alert_detected:
            alert_text = f"ALERT: {target_object.upper()} DETECTED"
            (text_width, text_height), _ = cv2.getTextSize(
                alert_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2
            )

            cv2.rectangle(
                img,
                (15, 15),
                (text_width + 35, 55),
                (0, 0, 255),
                -1
            )

            cv2.putText(
                img,
                alert_text,
                (25, 43),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            print("\a")

        if save_images and current_objects != self.prev_objects:
            filename = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
            filepath = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, img)
            self.prev_objects = current_objects

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.markdown('<div class="panel">', unsafe_allow_html=True)

webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("## 📂 Detection Gallery")

image_files = glob.glob(os.path.join(SAVE_DIR, "*.jpg"))

if image_files:
    cols = st.columns(3)
    for index, img_path in enumerate(reversed(image_files[-9:])):
        with cols[index % 3]:
            st.image(
                img_path,
                use_container_width=True
            )
            st.caption(
                os.path.basename(img_path)
            )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for file in image_files:
            zip_file.write(file, os.path.basename(file))

    st.download_button(
        label="⬇ Download Detection Logs",
        data=zip_buffer.getvalue(),
        file_name="detection_logs.zip",
        mime="application/zip"
    )
else:
    st.info("No saved detections yet.")
