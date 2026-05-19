import streamlit as st
import cv2
import tempfile
import os
from roboflow import Roboflow
import numpy as np
import time

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Pavement Distress Detection",
    page_icon="🛣️",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.main {
    background-color: #050816;
    color: white;
}

.stApp {
    background-color: #050816;
}

h1, h2, h3 {
    color: white;
}

.stButton>button {
    background: linear-gradient(90deg, #7b2ff7, #f107a3);
    color: white;
    border-radius: 12px;
    border: none;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    transform: scale(1.02);
}

.block-container {
    padding-top: 2rem;
}

.metric-card {
    background-color: #111827;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #1f2937;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.title("🛣️ Pavement Distress Detection System")

st.markdown("""
### AI-Based Road Damage Detection

Upload road inspection video to automatically detect:

- Cracks
- Potholes
- Pavement Defects
""")

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.header("⚙️ System Settings")

    confidence = st.slider(
        "Confidence Threshold",
        10,
        100,
        40
    )

    frame_skip = st.slider(
        "Processing Speed",
        1,
        10,
        5
    )

    st.markdown("---")

    st.info("""
    Recommended Video:
    
    - Format: MP4
    - Resolution: 720p
    - FPS: 20-30
    - Size: Below 300MB
    """)

# =========================================================
# ROBOFLOW CONFIG
# =========================================================
ROBOFLOW_API_KEY = "PEg5q48Ar8j8zKbAqHd7"

PROJECT_ID = "pavement-distress-detection-c5hzn"

VERSION_NUMBER = 1

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)

    project = rf.workspace().project(PROJECT_ID)

    model = project.version(VERSION_NUMBER).model

    return model

# =========================================================
# LOAD MODEL SAFELY
# =========================================================
try:
    model = load_model()
    st.success("✅ AI Model Loaded Successfully")

except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()

# =========================================================
# VIDEO UPLOAD
# =========================================================
uploaded_video = st.file_uploader(
    "📤 Upload Road Video",
    type=["mp4", "mov", "avi"]
)

# =========================================================
# PROCESS VIDEO
# =========================================================
if uploaded_video is not None:

    st.video(uploaded_video)

    analyze = st.button("🚀 Start AI Analysis")

    if analyze:

        try:

            st.info("⏳ Processing video... Please wait.")

            # =================================================
            # SAVE INPUT VIDEO
            # =================================================
            temp_input = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            temp_input.write(uploaded_video.read())

            input_path = temp_input.name

            output_path = "processed_output.mp4"

            # =================================================
            # OPEN VIDEO
            # =================================================
            cap = cv2.VideoCapture(input_path)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps <= 0:
                fps = 20

            # =================================================
            # VIDEO WRITER
            # =================================================
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            progress_bar = st.progress(0)

            status_text = st.empty()

            frame_index = 0

            max_frames = 1500

            detection_count = 0

            # =================================================
            # PROCESS LOOP
            # =================================================
            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                frame_index += 1

                if frame_index > max_frames:
                    break

                # =============================================
                # PROCESS EVERY N FRAMES
                # =============================================
                if frame_index % frame_skip == 0:

                    try:

                        # Resize for faster inference
                        small_frame = cv2.resize(
                            frame,
                            (416, 234)
                        )

                        predictions = model.predict(
                            small_frame,
                            confidence=confidence,
                            overlap=30
                        ).json()

                        for prediction in predictions["predictions"]:

                            detection_count += 1

                            x = prediction["x"]
                            y = prediction["y"]
                            w = prediction["width"]
                            h = prediction["height"]

                            class_name = prediction["class"]

                            conf = prediction["confidence"]

                            # Coordinates
                            x1 = int(x - w / 2)
                            y1 = int(y - h / 2)
                            x2 = int(x + w / 2)
                            y2 = int(y + h / 2)

                            # Scale back
                            scale_x = width / 416
                            scale_y = height / 234

                            x1 = int(x1 * scale_x)
                            y1 = int(y1 * scale_y)
                            x2 = int(x2 * scale_x)
                            y2 = int(y2 * scale_y)

                            # =================================
                            # PROFESSIONAL BOX
                            # =================================
                            cv2.rectangle(
                                frame,
                                (x1, y1),
                                (x2, y2),
                                (0, 255, 0),
                                3
                            )

                            label = f"{class_name} {conf:.2f}"

                            cv2.rectangle(
                                frame,
                                (x1, y1 - 35),
                                (x1 + 220, y1),
                                (0, 255, 0),
                                -1
                            )

                            cv2.putText(
                                frame,
                                label,
                                (x1 + 10, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 0),
                                2
                            )

                    except Exception as infer_error:
                        st.warning(f"Inference Error: {infer_error}")

                # =============================================
                # WRITE FRAME
                # =============================================
                out.write(frame)

                # =============================================
                # UPDATE PROGRESS
                # =============================================
                progress = min(
                    frame_index / total_frames,
                    1.0
                )

                progress_bar.progress(progress)

                status_text.text(
                    f"Processing Frame: {frame_index}/{total_frames}"
                )

            # =================================================
            # RELEASE
            # =================================================
            cap.release()
            out.release()

            # =================================================
            # RESULTS
            # =================================================
            st.success("✅ Video Analysis Complete")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Frames Processed",
                    frame_index
                )

            with col2:
                st.metric(
                    "Detections",
                    detection_count
                )

            with col3:
                st.metric(
                    "Processing Speed",
                    f"1/{frame_skip} Frames"
                )

            st.markdown("---")

            st.subheader("🎥 Processed Video")

            st.video(output_path)

            # =================================================
            # DOWNLOAD BUTTON
            # =================================================
            with open(output_path, "rb") as file:

                st.download_button(
                    label="⬇️ Download Processed Video",
                    data=file,
                    file_name="processed_output.mp4",
                    mime="video/mp4"
                )

            # =================================================
            # CLEANUP
            # =================================================
            os.remove(input_path)

        except Exception as e:

            st.error(f"Processing Error: {e}")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.caption("Developed for Pavement Distress Detection Research")
