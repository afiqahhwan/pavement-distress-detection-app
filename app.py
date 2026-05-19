import streamlit as st
import cv2
import supervision as sv
import tempfile
import os
import time
from roboflow import Roboflow

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
    background-color: #0E1117;
}

h1, h2, h3, h4 {
    color: white;
}

.stButton button {
    width: 100%;
    background: linear-gradient(to right, #7B2FF7, #F107A3);
    color: white;
    font-size: 20px;
    border-radius: 12px;
    height: 60px;
    border: none;
    font-weight: bold;
}

.stButton button:hover {
    background: linear-gradient(to right, #9B4DFF, #FF4DA6);
}

.upload-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #1B1F2A;
}

.metric-box {
    background-color: #1B1F2A;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================
st.title("🛣️ Pavement Distress Detection System")

st.markdown("""
### AI-Based Road Damage Detection using YOLO + Roboflow

Upload a road inspection video to automatically detect:

- Cracks
- Potholes
- Pavement Defects

Supports large video optimization and HD processing.
""")

# =========================================================
# ROBOFLOW CONFIG
# =========================================================
ROBOFLOW_API_KEY = "PEg5q48Ar8j8zKbAqHd7"

WORKSPACE_ID = "wans-workspace-na8wt"

PROJECT_ID = "pavement-distress-detection-c5hzn"

VERSION_NUMBER = 1

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)

    workspace = rf.workspace(WORKSPACE_ID)

    project = workspace.project(PROJECT_ID)

    model = project.version(VERSION_NUMBER).model

    return model

model = load_model()

# =========================================================
# SUPERVISION TOOLS
# =========================================================
box_annotator = sv.BoxAnnotator(thickness=3)

label_annotator = sv.LabelAnnotator(text_scale=0.7)

# =========================================================
# VIDEO COMPRESSION
# =========================================================
def compress_video(input_path):

    compressed_path = "compressed_video.mp4"

    cap = cv2.VideoCapture(input_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0 or fps > 60:
        fps = 20

    width = 1280
    height = 720

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        compressed_path,
        fourcc,
        fps,
        (width, height)
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        resized = cv2.resize(frame, (width, height))

        out.write(resized)

    cap.release()
    out.release()

    return compressed_path

# =========================================================
# PROCESS VIDEO
# =========================================================
def process_video(video_path):

    output_path = "processed_output.mp4"

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0 or fps > 60:
        fps = 20

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

    total_detections = 0

    start_time = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # FRAME SKIPPING FOR SPEED
        if frame_index % 2 != 0:
            frame_index += 1
            continue

        try:

            result = model.predict(
                frame,
                confidence=40,
                overlap=30
            ).json()

            detections = sv.Detections.from_inference(result)

            total_detections += len(result["predictions"])

            labels = [
                f"{item['class']} {item['confidence']:.2f}"
                for item in result["predictions"]
            ]

            annotated_frame = box_annotator.annotate(
                scene=frame.copy(),
                detections=detections
            )

            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels
            )

        except Exception as e:

            annotated_frame = frame.copy()

            cv2.putText(
                annotated_frame,
                "Inference Error",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        out.write(annotated_frame)

        frame_index += 1

        progress = min(frame_index / total_frames, 1.0)

        progress_bar.progress(progress)

        status_text.text(
            f"Processing frame {frame_index}/{total_frames}"
        )

    cap.release()

    out.release()

    end_time = time.time()

    processing_time = round(end_time - start_time, 2)

    return output_path, total_detections, processing_time

# =========================================================
# SIDEBAR SETTINGS
# =========================================================
st.sidebar.title("⚙️ System Settings")

confidence_setting = st.sidebar.slider(
    "Confidence Threshold",
    10,
    100,
    40
)

st.sidebar.markdown("---")

st.sidebar.info("""
Recommended Video Settings:

- Format: MP4
- Resolution: 720p
- FPS: 20-30
- Size: Below 500MB
""")

# =========================================================
# VIDEO UPLOAD
# =========================================================
uploaded_video = st.file_uploader(
    "📤 Upload Road Video",
    type=["mp4", "avi", "mov"]
)

# =========================================================
# MAIN UI
# =========================================================
if uploaded_video is not None:

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📹 Original Video")

        st.video(uploaded_video)

    if st.button("🚀 Start Detection"):

        with st.spinner("Analyzing road condition..."):

            temp_input = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            temp_input.write(uploaded_video.read())

            temp_input.close()

            # STEP 1
            compressed_video = compress_video(temp_input.name)

            # STEP 2
            output_video, total_detections, processing_time = process_video(
                compressed_video
            )

            with col2:

                st.subheader("✅ Processed Output")

                st.video(output_video)

            # =========================================================
            # METRICS
            # =========================================================
            st.markdown("---")

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "Total Detections",
                total_detections
            )

            metric2.metric(
                "Processing Time",
                f"{processing_time} sec"
            )

            metric3.metric(
                "Video Status",
                "Completed"
            )

            # =========================================================
            # DOWNLOAD BUTTON
            # =========================================================
            with open(output_video, "rb") as file:

                st.download_button(
                    label="📥 Download Processed Video",
                    data=file,
                    file_name="processed_output.mp4",
                    mime="video/mp4"
                )

            # =========================================================
            # CLEANUP
            # =========================================================
            os.remove(temp_input.name)

            if os.path.exists(compressed_video):
                os.remove(compressed_video)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.caption("Developed for Pavement Distress Detection Research")
