import streamlit as st

from src.inference import fall_detector

st.title("Fall Detection App")

st.write("Welcome to the Fall Detection Application!")

# Input widgets
subject_id = st.text_input("Subject ID", value="")

inference_path = st.text_input("Inference Path", value="")

if st.button("Run Inference"):
    if not inference_path:
        st.error("Please provide a valid inference path.")
    else:
        result = fall_detector.predict(inference_path)
        st.success(f"Inference Result: {result}")
        st.balloons()