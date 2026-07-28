import streamlit as st
import cv2
import tempfile
import os
import gc
from roboflow import Roboflow

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Intelligent Pavement Distress Detection System",
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

.main {
    background-color: #050816;
}

h1, h2, h3 {
    color: white;
}

.stButton>button {
    background: linear-gradient(90deg, #7b2ff7, #2563eb);
    color: white;
    border-radius: 12px;
    border: none;
    height: 55px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    transform: scale(1.02);
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER & DESCRIPTION
# =========================================================
st.title("🛣️ Intelligent Pavement Distress Detection System")

st.markdown("""
### Developed for Pavement Detection Research

AI-powered road surface inspection system for automated pavement distress analysis.
""")

st.markdown("""
### 🔍 Pavement Distress Types Detected

This intelligent system automatically detects:
- ✅ Cracks
- ✅ Potholes
- ✅ Patching

The uploaded road inspection video will be analyzed frame-by-frame using AI detection technology.
""")

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("⚙️ System Settings")

    confidence = st.slider(
        "Confidence Threshold (%)",
        10,
        100,
        40
    )

    frame_skip = st.slider(
        "Processing Speed (Skip Frames)",
        1,
        10,
        5
    )

    st.markdown("---")

    st.info("""
    Recommended Video Settings:
    • Format: MP4  
    • Resolution: 720p  
    • FPS: 20–30  
    • Size: Below 300MB
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

try:
    model = load_model()
except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()

# =========================================================
# VIDEO UPLOAD
# =========================================================
uploaded_video = st.file_uploader(
    "📤 Upload Road Inspection Video",
    type=["mp4", "mov", "avi"]
)

# =========================================================
# PROCESS VIDEO
# =========================================================
if uploaded_video is not None:
    st.video(uploaded_video)

    analyze = st.button("🚀 Start Detection")

    if analyze:
        try:
            st.info("🚀 AI Detection in Progress...")

            # Save uploaded input video to a temp file
            temp_input = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )
            temp_input.write(uploaded_video.read())
            temp_input.close()

            input_path = temp_input.name
            output_path = "processed_output.mp4"

            # Open Input Video
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 20

            output_width = 960
            output_height = 540

            # Web-compatible FourCC Codec Selection
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (output_width, output_height)
            )

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_bar = st.progress(0)
            status_text = st.empty()

            frame_count = 0
            max_frames = 600

            crack_count = 0
            pothole_count = 0
            patching_count = 0

            # Persistent store for detected annotations across frame skips
            last_predictions = []

            # Process Video Loop
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count > max_frames:
                    break

                # Resize frame for canvas output
                frame = cv2.resize(frame, (output_width, output_height))

                # Update Loading Bar
                if total_frames > 0:
                    progress = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing Frame {frame_count} / {total_frames}")

                # Inference Trigger
                if frame_count % frame_skip == 0 or frame_count == 1:
                    try:
                        small_frame = cv2.resize(frame, (320, 180))

                        response = model.predict(
                            small_frame,
                            confidence=confidence,
                            overlap=30
                        ).json()

                        last_predictions = response.get("predictions", [])

                        # Count overall detected distresses
                        for pred in last_predictions:
                            lbl = pred["class"].lower()
                            if "crack" in lbl:
                                crack_count += 1
                            elif "pothole" in lbl:
                                pothole_count += 1
                            elif "patch" in lbl:
                                patching_count += 1

                    except Exception as infer_error:
                        st.warning(f"Inference Error on frame {frame_count}: {infer_error}")

                # Draw bounding boxes on target frame
                for prediction in last_predictions:
                    x = prediction["x"]
                    y = prediction["y"]
                    w = prediction["width"]
                    h = prediction["height"]
                    label = prediction["class"]
                    conf = prediction["confidence"]

                    # Convert center coords to corners
                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    x2 = int(x + w / 2)
                    y2 = int(y + h / 2)

                    # Scale from small_frame dimensions (320x180) to output dimensions (960x540)
                    scale_x = output_width / 320.0
                    scale_y = output_height / 180.0

                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)

                    # Bounding Box Draw
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    # Label Tag Draw
                    text = f"{label} {conf:.2f}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(
                        frame,
                        (x1, y1 - 25),
                        (x1 + text_size[0] + 10, y1),
                        (0, 255, 0),
                        -1
                    )
                    cv2.putText(
                        frame,
                        text,
                        (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2
                    )

                # Write frame to video stream
                out.write(frame)

                del frame
                gc.collect()

            # Release Video Handlers
            cap.release()
            out.release()

            # Display Summary Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Cracks Detected", crack_count)
            with col2:
                st.metric("Potholes Detected", pothole_count)
            with col3:
                st.metric("Patchings Detected", patching_count)

            st.markdown("---")

            # Output Video Playback & Download
            st.subheader("🎥 Processed Video")
            st.video(output_path)

            with open(output_path, "rb") as file:
                st.download_button(
                    label="⬇️ Download Processed Video",
                    data=file,
                    file_name="processed_output.mp4",
                    mime="video/mp4"
                )

            # Cleanup Temp Input File
            if os.path.exists(input_path):
                os.remove(input_path)

        except Exception as e:
            st.error(f"Processing Error: {e}")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 15px;'>
Developed for Pavement Detection Research<br>
Intelligent AI System for Detecting: Cracks • Potholes • Patching
</div>
""", unsafe_allow_html=True)
