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
    page_title="Pavement Distress Detection System",
    page_icon="🛣️",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #050816;
    color: white;
    font-family: 'Segoe UI';
}

.main-title {
    font-size: 52px;
    font-weight: 700;
    color: white;
}

.subtitle {
    font-size: 24px;
    color: #b8c1ec;
}

.stButton>button {
    width: 100%;
    height: 60px;
    border-radius: 12px;
    background: linear-gradient(to right,#7c3aed,#2563eb);
    color: white;
    font-size: 20px;
    font-weight: bold;
    border: none;
}

.metric-box {
    background: linear-gradient(135deg,#1f2937,#111827);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #374151;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("⚙️ System Settings")

    confidence = st.slider(
        "Confidence Threshold",
        10,
        100,
        40
    ) / 100

    st.markdown("---")

    st.markdown("""
    ### Recommended Settings

    - Format: MP4
    - Resolution: 720p
    - FPS: 24-30
    - Large video support
    - AI HD processing
    """)

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="main-title">
🛣️ Pavement Distress Detection System
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
AI-Based Road Damage Detection & Video Analysis
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

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
# VIDEO UPLOAD
# =========================================================

uploaded_video = st.file_uploader(
    "📤 Upload Road Inspection Video",
    type=["mp4", "mov", "avi", "mkv"]
)

# =========================================================
# PROCESS VIDEO
# =========================================================

if uploaded_video is not None:

    st.video(uploaded_video)

    if st.button("🚀 Start AI Analysis"):

        with st.spinner("Analyzing pavement condition..."):

            # =================================================
            # SAVE INPUT VIDEO
            # =================================================

            temp_input = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            temp_input.write(uploaded_video.read())

            input_path = temp_input.name

            # =================================================
            # VIDEO CAPTURE
            # =================================================

            cap = cv2.VideoCapture(input_path)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps <= 0:
                fps = 30

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # =================================================
            # OUTPUT VIDEO
            # =================================================

            output_path = "processed_output.mp4"

            fourcc = cv2.VideoWriter_fourcc(*'avc1')

            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            # =================================================
            # ANNOTATORS
            # =================================================

            box_annotator = sv.BoxAnnotator(
                thickness=3
            )

            label_annotator = sv.LabelAnnotator(
                text_scale=0.7
            )

            # =================================================
            # COUNTERS
            # =================================================

            total_detections = 0
            crack_count = 0
            pothole_count = 0

            frame_index = 0

            progress_bar = st.progress(0)

            # =================================================
            # PROCESS LOOP
            # =================================================

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                try:

                    # =========================================
                    # SKIP EVERY 2ND FRAME
                    # =========================================

                    if frame_index % 2 != 0:
                        frame_index += 1
                        continue

                    # =========================================
                    # RESIZE FOR FAST API INFERENCE
                    # =========================================

                    small_frame = cv2.resize(
                        frame,
                        (640, 360)
                    )

                    # =========================================
                    # RUN AI DETECTION
                    # =========================================

                    result = model.predict(
                        small_frame,
                        confidence=confidence * 100
                    ).json()

                    detections = sv.Detections.from_inference(result)

                    # =========================================
                    # SCALE DETECTIONS
                    # =========================================

                    scale_x = width / 640
                    scale_y = height / 360

                    if len(detections) > 0:

                        detections.xyxy[:, [0, 2]] *= scale_x
                        detections.xyxy[:, [1, 3]] *= scale_y

                    # =========================================
                    # LABELS
                    # =========================================

                    labels = []

                    for pred in result["predictions"]:

                        cls = pred["class"]
                        conf = pred["confidence"]

                        labels.append(
                            f"{cls} {conf:.2f}"
                        )

                        total_detections += 1

                        if "crack" in cls.lower():
                            crack_count += 1

                        if "pothole" in cls.lower():
                            pothole_count += 1

                    # =========================================
                    # DRAW DETECTIONS
                    # =========================================

                    annotated_frame = box_annotator.annotate(
                        scene=frame.copy(),
                        detections=detections
                    )

                    annotated_frame = label_annotator.annotate(
                        scene=annotated_frame,
                        detections=detections,
                        labels=labels
                    )

                    # =========================================
                    # PROFESSIONAL OVERLAY
                    # =========================================

                    overlay = annotated_frame.copy()

                    cv2.rectangle(
                        overlay,
                        (20, 20),
                        (470, 230),
                        (0, 0, 0),
                        -1
                    )

                    cv2.addWeighted(
                        overlay,
                        0.45,
                        annotated_frame,
                        0.55,
                        0,
                        annotated_frame
                    )

                    cv2.putText(
                        annotated_frame,
                        "ROAD AI ANALYSIS",
                        (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,255),
                        3
                    )

                    cv2.putText(
                        annotated_frame,
                        f"Frame: {frame_index}",
                        (40, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255,255,255),
                        2
                    )

                    cv2.putText(
                        annotated_frame,
                        f"Detections: {total_detections}",
                        (40, 135),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255,255,255),
                        2
                    )

                    cv2.putText(
                        annotated_frame,
                        f"Cracks: {crack_count}",
                        (40, 170),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,255,0),
                        2
                    )

                    cv2.putText(
                        annotated_frame,
                        f"Potholes: {pothole_count}",
                        (40, 205),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,0,255),
                        2
                    )

                    # =========================================
                    # WRITE OUTPUT VIDEO
                    # =========================================

                    out.write(annotated_frame)

                except Exception as e:

                    st.error(f"Inference Error: {e}")

                frame_index += 1

                progress_bar.progress(
                    min(frame_index / total_frames, 1.0)
                )

            # =================================================
            # RELEASE
            # =================================================

            cap.release()
            out.release()
            cv2.destroyAllWindows()

            # =================================================
            # SHOW RESULTS
            # =================================================

            st.success("AI Analysis Complete")

            with open(output_path, "rb") as video_file:
                video_bytes = video_file.read()

            st.video(video_bytes)

            # =================================================
            # METRICS
            # =================================================

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Total Detections",
                    total_detections
                )

            with col2:
                st.metric(
                    "Cracks",
                    crack_count
                )

            with col3:
                st.metric(
                    "Potholes",
                    pothole_count
                )

            # =================================================
            # DOWNLOAD BUTTON
            # =================================================

            with open(output_path, "rb") as file:

                st.download_button(
                    "⬇️ Download Processed Video",
                    file,
                    file_name="processed_output.mp4"
                )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Developed for Pavement Distress Detection Research"
)
