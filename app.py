import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os

# Load your trained YOLO model
@st.cache_resource
def load_model():
    model_path = r"C:\Users\Abdullah\intraoral\runs\detect\intraoral\weights\best.pt"
    return YOLO(model_path)

model = load_model()

st.title("🦷 DentXAI: Dental Disease Detection using YOLOv8n")

# Confidence slider
conf = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save uploaded file
    save_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Run inference
    results = model.predict(source=save_path, conf=conf)

    # YOLO result with bounding boxes
    result_image = results[0].plot()  # numpy array with boxes drawn

    # Convert to PIL Image
    output_img = Image.fromarray(result_image[..., ::-1])  # BGR -> RGB

    # Show result
    st.image(output_img, caption="Detected Image", use_column_width=True)

    # Download option
    output_path = os.path.join("outputs", "result_" + uploaded_file.name)
    os.makedirs("outputs", exist_ok=True)
    output_img.save(output_path)
    with open(output_path, "rb") as f:
        st.download_button("Download Output", f, file_name="result.jpg")
