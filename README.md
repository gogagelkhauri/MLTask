ML Task

An automated, single-workflow YOLOv8 instance segmentation system. This repository provides end-to-end dataset merging, model training, accuracy evaluation, and interactive Streamlit web inference.

🛠️ Used Libraries

Ultralytics (YOLOv8): Instance segmentation, model training, and validation metrics
Streamlit: Web dashboard for interactive image uploads and inference
OpenCV & NumPy: Matrix manipulation, mask processing, and rendering image overlays
Pillow (PIL): Image loading and format transformations

📂 Project Structure

```text
MLTask/
  config/
    data_unified.yaml       - Auto-generated dataset and class mapping configuration
  datasets/
    raw_camel/              - Source raw dataset folder
    raw_elephant/           - Source raw dataset folder
    unified_dataset/        - Auto-merged unified dataset (train & val split)
  runs/
    segment/
      unified_run/          - Persistent output directory (contains best.pt)
  train_pipeline.py         - Automated dataset builder and trainer
  evaluate.py               - Validation & accuracy metrics evaluator
  main.py                   - Streamlit web interface for model inference
  ```

⚡ Quickstart Guide

Follow these commands in sequence after cloning the repository.

1. Environment Setup

Install project dependencies using uv:

uv sync


2. Dataset Preparation

Place your raw image and annotation folders inside the datasets/ directory adhering to the raw_<class_name> naming convention (e.g., datasets/raw_camel/, datasets/raw_elephant/).

3. Run Automated Training

Merge all raw datasets, construct config/data_unified.yaml, and train the unified YOLOv8 segmentation model using a single command:

uv run python train_pipeline.py --epochs 40


Optional command line flags:

--epochs: Set number of training epochs (default: 40)

--batch: Set batch size (default: 16)

--lr0: Set initial learning rate (default: 0.01)

--imgsz: Image resolution size in pixels (default: 640)

# Example with custom hyperparameters:
uv run python train_pipeline.py --epochs 50 --batch 8 --lr0 0.005


4. Evaluate Model Metrics

To measure model performance (Box mAP50-95 and Mask mAP50-95) against the validation split:

uv run python evaluate.py


5. Launch Interactive Interface

Start the Streamlit web app to upload test images and perform real-time visual segmentation:

uv run streamlit run main.py


⚙️ How It Works

Dataset Unification (train_pipeline.py): Automatically scans all raw_* folders in datasets/, assigns sequential class index IDs, renames files to prevent naming collisions, updates label text files, and generates config/data_unified.yaml.

Unified Training: Automatically outputs training runs into a single persistent folder: runs/segment/unified_run/weights/best.pt.

Streamlit Inference (main.py): Directly loads runs/segment/unified_run/weights/best.pt to process user-uploaded images and apply segmentation overlays using OpenCV.