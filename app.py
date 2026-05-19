import streamlit as st
import cv2
import tempfile
import supervision as sv
from roboflow import Roboflow
import os

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Pavement Distress Detection",
    page_icon="🚧",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #0B1020;
    color: white;
}

.main {
    background-color: #0B1020;
}

h1, h2, h3 {
    color: white;
}

.stButton > button {
    background-color: #7C3AED;
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background-color: #6D28D9;
    color: white;
}

[data-testid="stMetricValue"] {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.title("🚧 Pavement Distress Detection System")

st.markdown("""
### Final Year Project (FYP)

This AI system uses:
- YOLO Object Detection
- Roboflow AI
- Computer Vision
- Deep Learning

to automatically detect:
- Cracks
- Potholes
- Pavement defects

from uploaded road inspection videos.
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
# SETTINGS
# =========================================================
FRAME_SKIP = 5
RESIZE_WIDTH = 480

# =========================================================
# ANNOTATORS
# =========================================================
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# =========================================================
# RESIZE FRAME
# =========================================================
def resize_frame(frame, width=480):

    h, w = frame.shape[:2]

    if w <= width:
        return frame

    ratio = width / w

    height = int(h * ratio)

    resized = cv2.resize(frame, (width, height))

    return resized

# =========================================================
# PROCESS VIDEO
# =========================================================
def process_video(video_path):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video settings
    output_width = 480

    output_height = int(height * (output_width / width))

    output_path = "processed_output.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (output_width, output_height)
    )

    progress_bar = st.progress(0)

    status_text = st.empty()

    frame_count = 0

    crack_count = 0

    pothole_count = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Skip frames
        if frame_count % FRAME_SKIP != 0:
            continue

        # Resize frame
        frame = resize_frame(frame, output_width)

        try:

            # =================================================
            # ROBOFLOW PREDICTION
            # =================================================
            result = model.predict(
                frame,
                confidence=40,
                overlap=30
            ).json()

            detections = sv.Detections.from_inference(result)

            labels = []

            for pred in result.get("predictions", []):

                class_name = pred["class"]

                confidence = pred["confidence"]

                labels.append(
                    f"{class_name} {confidence:.2f}"
                )

                # Count defects
                if "crack" in class_name.lower():
                    crack_count += 1

                if "pothole" in class_name.lower():
                    pothole_count += 1

            # =================================================
            # DRAW BOXES
            # =================================================
            annotated_frame = box_annotator.annotate(
                scene=frame.copy(),
                detections=detections
            )

            # =================================================
            # DRAW LABELS
            # =================================================
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels
            )

            # Write frame
            out.write(annotated_frame)

        except Exception as e:

            st.error(f"Inference Error: {e}")

        # Update progress
        progress = min(frame_count / total_frames, 1.0)

        progress_bar.progress(progress)

        status_text.text(
            f"Processing frame {frame_count}/{total_frames}"
        )

    cap.release()

    out.release()

    return output_path, frame_count, crack_count, pothole_count

# =========================================================
# UI LAYOUT
# =========================================================
col1, col2 = st.columns(2)

# =========================================================
# LEFT COLUMN
# =========================================================
with col1:

    st.subheader("📤 Upload Road Video")

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    # File size info
    if uploaded_video is not None:

        file_size = uploaded_video.size / (1024 * 1024)

        st.info(
            f"Uploaded Video Size: {file_size:.2f} MB"
        )

        if file_size > 150:

            st.warning(
                "Large videos may take longer to process."
            )

    analyze_btn = st.button("🚀 Analyze Video")

# =========================================================
# RIGHT COLUMN
# =========================================================
with col2:

    st.subheader("📥 Processed Output")

# =========================================================
# PROCESS BUTTON
# =========================================================
if uploaded_video and analyze_btn:

    st.info("Processing video... Please wait.")

    # Save uploaded video temporarily
    temp_input = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    temp_input.write(uploaded_video.read())

    temp_input.close()

    # Process video
    output_path, frames, cracks, potholes = process_video(
        temp_input.name
    )

    # Display output video
    with col2:

        st.video(output_path)

    # =====================================================
    # METRICS
    # =====================================================
    st.subheader("📊 Detection Summary")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("Frames Processed", frames)

    with m2:
        st.metric("Detected Cracks", cracks)

    with m3:
        st.metric("Detected Potholes", potholes)

    # =====================================================
    # DOWNLOAD BUTTON
    # =====================================================
    with open(output_path, "rb") as file:

        st.download_button(
            label="⬇ Download Processed Video",
            data=file,
            file_name="processed_output.mp4",
            mime="video/mp4"
        )

    st.success("Video analysis completed successfully!")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

n
""")
