import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os

st.set_page_config(
    page_title="🦷 DentXAI: Dental Disease Detection",
    layout="centered"
)

@st.cache_resource
def load_model(model_path: str):
    """Load YOLO model from given path."""
    return YOLO(model_path)

MODEL_PATH = "runs/detect/intraoral/weights/best.pt"
model = load_model(MODEL_PATH)


st.title("🦷 DentXAI: Dental Disease Detection using YOLOv8n")
conf = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    upload_path = os.path.join("uploads", uploaded_file.name)
    with open(upload_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("Running inference...")

    results = model.predict(source=upload_path, conf=conf)

    result_image = results[0].plot()  # BGR image

    output_img = Image.fromarray(result_image[..., ::-1])

    st.image(output_img, caption="Detected Image", use_column_width=True)

    output_path = os.path.join("outputs", f"result_{uploaded_file.name}")
    output_img.save(output_path)

    with open(output_path, "rb") as f:
        st.download_button(
            label="Download Output",
            data=f,
            file_name=f"result_{uploaded_file.name}",
            mime="image/jpeg"
        )
