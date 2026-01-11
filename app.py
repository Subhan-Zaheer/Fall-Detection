import tempfile
import streamlit as st
from src.inference import fall_detector

st.title("Fall Detection App")
st.write("Welcome to the Fall Detection Application!")

# Radio button (default is Video File)
inference_type = st.radio(
    "Select Inference Type",
    ("Video File", "Image Sequence Folder"),
    index=0
)

# -------------------------
# CONDITIONAL INPUTS
# -------------------------

uploaded_file = None
inference_path = None

if inference_type == "Video File":
    uploaded_file = st.file_uploader(
        "Upload Video File",
        type=["mp4", "avi", "mov", "mkv", "webm", "flv", "wmv", "mpeg", "mpg"]
    )

else:
    inference_path = st.text_input(
        "Enter Image Sequence Folder Path",
        value=""
    )

# -------------------------
# RUN INFERENCE
# -------------------------

if st.button("Run Inference"):
    if inference_type == "Video File":
        if uploaded_file is None:
            st.error("Please upload a video file.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_file.read())
                video_path = tmp.name

            result = fall_detector.predict(video_path)
            st.success(f"Inference Result: {result}")
            st.balloons()

    else:
        if not inference_path:
            st.error("Please provide a valid folder path.")
        else:
            result = fall_detector.predict(inference_path)
            st.success(f"Inference Result: {result}")
            st.balloons()
