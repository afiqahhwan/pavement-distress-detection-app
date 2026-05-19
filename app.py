import streamlit as st
import cv2
import tempfile
from roboflow import Roboflow
import supervision as sv

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Pavement Distress Detection",
    layout="wide"
)

st.title("🚗 Pavement Distress Detection System")

st.markdown("""
Upload a road inspection video to detect:
- Cracks
- Potholes
- Pavement damage
""")

# ==========================================
# LOAD ROBOFLOW MODEL
# ==========================================

ROBOFLOW_API_KEY = "PEg5q48Ar8j8zKbAqHd7"

PROJECT_ID = "pavement-distress-detection-c5hzn"

VERSION_NUMBER = 1

rf = Roboflow(api_key=ROBOFLOW_API_KEY)

project = rf.workspace().project(PROJECT_ID)

model = project.version(VERSION_NUMBER).model

# ==========================================
# ANNOTATORS
# ==========================================

box_annotator = sv.BoxAnnotator()

label_annotator = sv.LabelAnnotator()

# ==========================================
# VIDEO UPLOAD
# ==========================================

uploaded_video = st.file_uploader(
    "Upload Video",
    type=["mp4", "mov", "avi"]
)

# ==========================================
# PROCESS BUTTON
# ==========================================

if uploaded_video is not None:

    st.video(uploaded_video)

    if st.button("Analyze Video"):

        with st.spinner("Processing video..."):

            # Save uploaded file temporarily
            temp_input = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            temp_input.write(uploaded_video.read())

            input_path = temp_input.name

            output_path = "processed_output.mp4"

            # ==================================
            # CALLBACK
            # ==================================

            def callback(frame, index):

                # Skip frames
                if index % 5 != 0:
                    return frame

                # Resize frame
                frame = cv2.resize(frame, (416, 416))

                # Run inference
                result = model.predict(frame).json()

                detections = sv.Detections.from_inference(result)

                labels = [
                    f"{pred['class']} {pred['confidence']:.2f}"
                    for pred in result.get("predictions", [])
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

                return annotated_frame

            # ==================================
            # PROCESS VIDEO
            # ==================================

            sv.process_video(
                source_path=input_path,
                target_path=output_path,
                callback=callback
            )

            st.success("Processing complete!")

            # ==================================
            # SHOW OUTPUT
            # ==================================

            st.video(output_path)

            with open(output_path, "rb") as file:

                st.download_button(
                    label="Download Processed Video",
                    data=file,
                    file_name="processed_output.mp4",
                    mime="video/mp4"
                )
