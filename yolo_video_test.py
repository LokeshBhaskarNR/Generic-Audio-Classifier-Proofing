import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO

model_path = 'yolov8s_local.pt'
model = YOLO(model_path)

st.title('Object Detection in Videos with YOLO')

uploaded_file = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = 'output_video.mp4'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        annotated_frame = results[0].plot()

        out.write(annotated_frame)

        stframe.image(annotated_frame, channels="BGR")

    cap.release()
    out.release()
    os.unlink(tfile.name)

    with open(output_path, "rb") as file:
        st.download_button(label="Download Processed Video", data=file, file_name="output_video.mp4", mime="video/mp4")
