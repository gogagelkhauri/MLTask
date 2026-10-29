from ultralytics import YOLO


def evaluate_model():
    print("=== Model Evaluation Started ===")
    # Load your unified trained weights
    model = YOLO("runs/segment/unified_run/weights/best.pt")

    # Run validation using the unified dataset config
    metrics = model.val(data="config/data_unified.yaml")

    print("\n[+] Evaluation Complete!")
    print(f"    Box mAP50-95: {metrics.box.map}")
    print(f"    Mask mAP50-95: {metrics.seg.map}")


if __name__ == "__main__":
    evaluate_model()