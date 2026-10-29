import argparse
import os
import shutil
from pathlib import Path
from ultralytics import YOLO


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--epochs", default=40, type=int, help="Number of training epochs"
  )
  args = parser.parse_args()

  print(f"\n==============================================")
  print(f"[*] Building Unified Dataset from all raw folders...")
  print(f"==============================================")

  datasets_root = Path("datasets")
  unified_dir = datasets_root / "unified_dataset"

  # 1. Setup unified YOLO directory structure
  for split in ["train", "val"]:
    os.makedirs(unified_dir / "images" / split, exist_ok=True)
    os.makedirs(unified_dir / "labels" / split, exist_ok=True)

  # 2. Automatically find all raw_* folders
  raw_folders = sorted([f for f in datasets_root.glob("raw_*") if f.is_dir()])
  if not raw_folders:
    print("[!] ERROR: No raw_* folders found in datasets/!")
    return

  class_names = []
  print(f"[+] Found {len(raw_folders)} raw class folders.")

  # 3. Process and merge each raw folder into the unified dataset
  for class_idx, raw_dir in enumerate(raw_folders):
    class_name = raw_dir.name.replace("raw_", "")
    class_names.append(class_name)
    print(f"    -> Processing class [{class_idx}]: {class_name}")

    src_images_dir = (
        raw_dir / "images" if (raw_dir / "images").exists() else raw_dir
    )
    src_labels_dir = (
        raw_dir / "labels" if (raw_dir / "labels").exists() else raw_dir
    )

    image_files = []
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
      image_files.extend(list(src_images_dir.glob(f"*{ext}")))

    if not image_files:
      print(f"       [!] Warning: No images found in {raw_dir}. Skipping.")
      continue

    # Copy files with unique prefixing to avoid name collisions
    for img_file in image_files:
      unique_name = f"{class_name}_{img_file.name}"
      target_img = unified_dir / "images" / "train" / unique_name
      shutil.copy(img_file, target_img)

      # Handle labels
      label_file = src_labels_dir / f"{img_file.stem}.txt"
      target_label = (
          unified_dir / "labels" / "train" / f"{class_name}_{img_file.stem}.txt"
      )

      if label_file.exists():
        lines = label_file.read_text().strip().splitlines()
        updated_lines = []
        for line in lines:
          parts = line.strip().split()
          if parts:
            parts[0] = str(
                class_idx
            )  # Update class ID to match unified index
            updated_lines.append(" ".join(parts))
        target_label.write_text("\n".join(updated_lines) + "\n")
      else:
        # Auto-create placeholder box with correct unified class ID
        placeholder_content = f"{class_idx} 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n"
        target_label.write_text(placeholder_content)

  # Quick fix: Copy samples into 'val' so validation doesn't fail
  train_imgs = list((unified_dir / "images" / "train").glob("*"))
  val_sample = train_imgs[: min(4, len(train_imgs))]
  for img_file in val_sample:
    shutil.copy(img_file, unified_dir / "images" / "val" / img_file.name)
    train_label = (
        unified_dir / "labels" / "train" / f"{img_file.stem}.txt"
    )
    if train_label.exists():
      shutil.copy(
          train_label,
          unified_dir / "labels" / "val" / f"{img_file.stem}.txt",
      )

  print(f"[+] Unified dataset successfully created at: {unified_dir}")

  # 4. Generate unified config YAML
  yaml_path = Path("config/data_unified.yaml")
  yaml_path.parent.mkdir(exist_ok=True)

  names_dict_str = "\n".join(
      [f"  {idx}: {name}" for idx, name in enumerate(class_names)]
  )
  yaml_content = f"""path: {unified_dir.absolute()}
train: images/train
val: images/val

names:
{names_dict_str}
"""
  yaml_path.write_text(yaml_content)
  print(f"[+] Unified config generated at: {yaml_path}")

  # 5. Train model once into the single unified run folder
  print(f"\n=== Starting Unified Training (Epochs: {args.epochs}) ===")
  model = YOLO("yolov8n-seg.pt")

  model.train(
      data=str(yaml_path),
      epochs=args.epochs,
      imgsz=640,
      name="unified_run",
      exist_ok=True,  # Always overwrites this single folder cleanly
  )
  print(f"[+] Training complete! Saved in runs/segment/unified_run")


if __name__ == "__main__":
  main()