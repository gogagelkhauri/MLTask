from pathlib import Path
from PIL import Image
import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2

st.set_page_config(
    page_title="YOLOv8 Segmentation Suite",
    page_icon="👁️",
    layout="wide"
)

# Custom CSS to hide the sidebar completely and style image borders
st.markdown(
    """
    <style>
    /* Hide the sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    /* Hide the hamburger menu and footer if desired */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .stImage img {
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Object Segmentation Pipeline")
st.write("Upload an image, select your model, and run segmentation.")

# --- MODEL CONFIGURATION (Defaults used since sidebar is hidden) ---
runs_dir = Path("runs/segment")
available_models = {"General Pre-trained (80 COCO Classes)": "yolov8n-seg.pt"}

if runs_dir.exists():
    for run_path in runs_dir.glob("**/weights/best.pt"):
        run_name = run_path.parent.parent.name
        available_models[f"Custom: {run_name}"] = str(run_path)

# Automatically select your custom unified model if available, otherwise fall back to default
default_model_key = list(available_models.keys())[-1]  # Selects custom model if found
model_path = available_models[default_model_key]

# Fixed default confidence threshold (adjust here if needed)
conf_threshold = 0.25


@st.cache_resource
def load_model(path):
    return YOLO(path)


model = load_model(model_path)
st.success(f"Active Model: {default_model_key}")

# --- MAIN INTERFACE ---
uploaded_file = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, width=400)

    if st.button("Run Object Detection Pipeline"):
        with st.spinner("Processing inference..."):
            results = model(image, conf=conf_threshold)

            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            img_h, img_w, _ = img_bgr.shape

            if results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                boxes = results[0].boxes.data.cpu().numpy()  # Format: [x1, y1, x2, y2, conf, cls]

                overlay = img_bgr.copy()
                color = (0, 255, 0)

                font_scale = max(1.2, img_w / 450.0)
                thickness = max(2, int(img_w / 250.0))

                for i, mask in enumerate(masks):
                    mask_resized = cv2.resize(mask, (img_w, img_h))
                    bool_mask = mask_resized > 0.5
                    overlay[bool_mask] = color

                    box = boxes[i]
                    x1, y1 = int(box[0]), int(box[1])
                    conf_score = float(box[4])
                    cls_id = int(box[5])
                    class_name = model.names.get(cls_id, "object")

                    # Include confidence score in the label
                    label_text = f"{class_name} ({conf_score:.2f})"

                    # Get text bounds for background box
                    (w, h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    padding = int(h * 0.5)

                    # Draw dark background box behind text for clear visibility
                    cv2.rectangle(
                        overlay,
                        (x1, max(y1 - h - (padding * 2), 0)),
                        (x1 + w + (padding * 2), y1),
                        (0, 0, 0),
                        -1
                    )

                    # Draw class name and confidence score text
                    cv2.putText(
                        overlay,
                        label_text,
                        (x1 + padding, max(y1 - padding, h)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA
                    )

                alpha = 0.5
                final_bgr = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)
                output_image = Image.fromarray(cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB))

                with col2:
                    st.subheader("Detected Objects")
                    st.image(output_image, width=400)

                st.success("Segmentation completed successfully!")
            else:
                with col2:
                    st.subheader("Detected Objects")
                    st.warning(
                        "No objects detected with the current threshold.")