import streamlit as st
import cv2
import tempfile
import os
from roboflow import Roboflow

# ======================================
# PAGE
# ======================================
st.set_page_config(
    page_title="Pavement Detection",
    layout="wide"
)

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
# ======================================
# API
# ======================================
ROBOFLOW_API_KEY = "PEg5q48Ar8j8zKbAqHd7"

PROJECT_ID = "pavement-distress-detection-c5hzn"

VERSION_NUMBER = 1

# ======================================
# LOAD MODEL
# ======================================
@st.cache_resource
def load_model():

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)

    project = rf.workspace().project(PROJECT_ID)

    model = project.version(VERSION_NUMBER).model

    return model

model = load_model()

# ======================================
# UPLOAD VIDEO
# ======================================
uploaded_video = st.file_uploader(
    "Upload Video",
    type=["mp4", "mov", "avi"]
)

# ======================================
# PROCESS
# ======================================
if uploaded_video is not None:

    st.video(uploaded_video)

    if st.button("Start Detection"):

        temp_input = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_input.write(uploaded_video.read())

        input_path = temp_input.name

        output_path = "output.mp4"

        cap = cv2.VideoCapture(input_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 20

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        frame_count = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # Process every 5th frame
            if frame_count % 5 == 0:

                small_frame = cv2.resize(
                    frame,
                    (416, 234)
                )

                predictions = model.predict(
                    small_frame,
                    confidence=40,
                    overlap=30
                ).json()

                for prediction in predictions["predictions"]:

                    x = prediction["x"]
                    y = prediction["y"]
                    w = prediction["width"]
                    h = prediction["height"]

                    label = prediction["class"]

                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    x2 = int(x + w / 2)
                    y2 = int(y + h / 2)

                    scale_x = width / 416
                    scale_y = height / 234

                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

            out.write(frame)

        cap.release()
        out.release()

        st.success("Processing Complete")

        st.video(output_path)

        with open(output_path, "rb") as file:

            st.download_button(
                "Download Video",
                file,
                file_name="processed_output.mp4"
            )

        os.remove(input_path)

st.markdown("---")

st.markdown("""
<div style='text-align: center; color: gray; padding: 15px;'>

Developed for Pavement Detection Research

Intelligent AI System for Detecting:
Cracks • Potholes • Patching

</div>
""", unsafe_allow_html=True)
