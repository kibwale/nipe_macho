"""
Access module for the fynesse framework.

This module handles data access functionality including:
- Data loading from various sources (web, local files, databases)
- Legal compliance (intellectual property, privacy rights)
- Ethical considerations for data usage
- Error handling for access issues

Legal and ethical considerations are paramount in data access.
Ensure compliance with e.g. .GDPR, intellectual property laws, and ethical guidelines.

Best Practice on Implementation
===============================

1. BASIC ERROR HANDLING:
   - Use try/except blocks to catch common errors
   - Provide helpful error messages for debugging
   - Log important events for troubleshooting

2. WHERE TO ADD ERROR HANDLING:
   - File not found errors when loading data
   - Network errors when downloading from web
   - Permission errors when accessing files
   - Data format errors when parsing files

3. SIMPLE LOGGING:
   - Use print() statements for basic logging
   - Log when operations start and complete
   - Log errors with context information
   - Log data summary information

4. EXAMPLE PATTERNS:
   
   Basic error handling:
   try:
       df = pd.read_csv('data.csv')
   except FileNotFoundError:
       print("Error: Could not find data.csv file")
       return None
   
   With logging:
   print("Loading data from data.csv...")
   try:
       df = pd.read_csv('data.csv')
       print(f"Successfully loaded {len(df)} rows of data")
       return df
   except FileNotFoundError:
       print("Error: Could not find data.csv file")
       return None
"""

from typing import Any, Union
import pandas as pd
import logging
import shutil
import cv2
import random
import matplotlib.pyplot as plt
from pathlib import Path

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

import xml.etree.ElementTree as ET
import os
from pathlib import Path


def find_xml_dirs(root):
    """Finding XML directories."""
    xml_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == "xmls":
            xml_dirs.append(dirpath)
    return xml_dirs


def convert_annotation(xml_path, txt_path, classes):
    """Converting individual annotations."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    img_w = int(size.find('width').text)
    img_h = int(size.find('height').text)

    lines = []
    for obj in root.findall('object'):
        cls_name = obj.find('name').text
        if cls_name not in classes:
            continue  # drop non-target boxes
        cls_id = classes.index(cls_name)

        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    with open(txt_path, 'w') as f:
        f.write("\n".join(lines))


def convert_dataset(root, classes):
    """Processing the entire dataset."""
    xml_dirs = find_xml_dirs(root)
    print(f"Found {len(xml_dirs)} annotation folders")

    total_boxes = 0
    total_images_with_boxes = 0

    for xdir in xml_dirs:
        annotations_dir = os.path.dirname(xdir)
        country_dir = os.path.dirname(annotations_dir)
        images_dir = os.path.join(country_dir, "images")
        labels_dir = os.path.join(country_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)

        xml_files = list(Path(xdir).glob("*.xml"))
        for xml_file in xml_files:
            txt_path = os.path.join(labels_dir, xml_file.stem + ".txt")
            try:
                convert_annotation(xml_file, txt_path, classes)
                if os.path.getsize(txt_path) > 0:
                    total_images_with_boxes += 1
                    with open(txt_path) as f:
                        total_boxes += sum(1 for _ in f)
            except Exception as e:
                print(f"Error on {xml_file}: {e}")

        if os.path.isdir(images_dir):
            for img_file in Path(images_dir).glob("*.jpg"):
                txt_path = os.path.join(labels_dir, img_file.stem + ".txt")
                if not os.path.exists(txt_path):
                    open(txt_path, 'w').close()

        print(f"Converted {len(xml_files)} files in {country_dir}")

    print(f"\nTotal boxes: {total_boxes}")
    print(f"Images containing at least one target box: {total_images_with_boxes}")
    return classes



from pathlib import Path

def write_data_yaml(root, classes,train_subdir="train/images",  val_subdir="test/images", filename="data.yaml"):
    """Write a YOLO data.yaml configuration file."""
    root = Path(root)
    yaml_path = root / filename

    with open(yaml_path, "w") as f:
        f.write(f"train: {root / train_subdir}\n")
        f.write(f"val: {root / val_subdir}\n")
        f.write(f"nc: {len(classes)}\n")
        f.write(f"names: {classes}\n")

    print(f"data.yaml written to {yaml_path}")
   
def data() -> Union[pd.DataFrame, None]:
    """
    Read the data from the web or local file, returning structured format such as a data frame.

    IMPLEMENTATION GUIDE
    ====================

    1. REPLACE THIS FUNCTION WITH YOUR ACTUAL DATA LOADING CODE:
       - Load data from your specific sources
       - Handle common errors (file not found, network issues)
       - Validate that data loaded correctly
       - Return the data in a useful format

    2. ADD ERROR HANDLING:
       - Use try/except blocks for file operations
       - Check if data is empty or corrupted
       - Provide helpful error messages

    3. ADD BASIC LOGGING:
       - Log when you start loading data
       - Log success with data summary
       - Log errors with context

    4. EXAMPLE IMPLEMENTATION:
       try:
           print("Loading data from data.csv...")
           df = pd.read_csv('data.csv')
           print(f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
           return df
       except FileNotFoundError:
           print("Error: data.csv file not found")
           return None
       except Exception as e:
           print(f"Error loading data: {e}")
           return None

    Returns:
        DataFrame or other structured data format
    """
    logger.info("Starting data access operation")

    try:
        # IMPLEMENTATION: Replace this with your actual data loading code
        # Example: Load data from a CSV file
        logger.info("Loading data from data.csv")
        df = pd.read_csv("data.csv")

        # Basic validation
        if df.empty:
            logger.warning("Loaded data is empty")
            return None

        logger.info(
            f"Successfully loaded data: {len(df)} rows, {len(df.columns)} columns"
        )
        return df

    except FileNotFoundError:
        logger.error("Data file not found: data.csv")
        print("Error: Could not find data.csv file. Please check the file path.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        print(f"Error loading data: {e}")
        return None


def polygon_to_bbox(parts):
    """Converting a polygon-format label line into an axis-aligned bounding box."""
    cls = parts[0]
    coords = list(map(float, parts[1:]))
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    return f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"


def fix_polygon_labels(final_root, split, prefix="Kenya_"):
    """Converting any polygon-format lines in a split's label files to
    box format, in place. Only touches files matching the given prefix."""
    lbl_dir = Path(f"{final_root}/{split}/labels")
    fixed_count = 0
    total_converted = 0

    for f in lbl_dir.glob(f"{prefix}*.txt"):
        if f.stat().st_size == 0:
            continue
        lines = f.read_text().strip().split("\n")
        new_lines = []
        changed = False
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            if len(parts) == 5:
                new_lines.append(line)
            elif len(parts) >= 7 and len(parts) % 2 == 1:
                new_lines.append(polygon_to_bbox(parts))
                changed = True
                total_converted += 1
        if changed:
            f.write_text("\n".join(new_lines))
            fixed_count += 1

    print(f"{split}: fixed {fixed_count} files, converted {total_converted} polygon lines to boxes")
    return fixed_count, total_converted



def load_annotations(ROOT):
    annotations = []

    for split in ["train", "val", "test"]:
        labels = ROOT / split / "labels"
        images = ROOT / split / "images"

        for label in labels.glob("*.txt"):
            image = next(
                (images / f"{label.stem}{ext}"
                 for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
                 if (images / f"{label.stem}{ext}").exists()),
                None
            )

            if image:
                for line in open(label):
                    _, xc, yc, w, h = map(float, line.split()[:5])
                    annotations.append(
                        (image, xc, yc, w, h, w * h * 100)
                    )

    return annotations


def select_examples(annotations, targets, tolerance=0.25):
    examples = []

    for target in targets:
        candidates = [
            a for a in annotations
            if abs(a[-1] - target) <= tolerance
        ]
        examples.append(random.choice(candidates) if candidates else None)

    return examples


def plot_examples(examples, targets):
    fig, axes = plt.subplots(1, len(targets),
                             figsize=(5 * len(targets), 5))

    if len(targets) == 1:
        axes = [axes]

    for ax, target, example in zip(axes, targets, examples):

        if example is None:
            ax.set_title(f"{target}%\nNo example found")
            ax.axis("off")
            continue

        image, xc, yc, w, h, area = example

        img = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        bw, bh = w * W, h * H
        x, y = xc * W - bw / 2, yc * H - bh / 2

        ax.imshow(img)
        ax.add_patch(
            plt.Rectangle((x, y), bw, bh,
                          fill=False, linewidth=2)
        )
        ax.set_title(f"{target}% (actual: {area:.2f}%)")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
def fix_polygon_labels_all_splits(final_root, splits=("train", "val"), prefix="Kenya_"):
    """Running fix_polygon_labels across multiple splits."""
    results = {}
    for split in splits:
        results[split] = fix_polygon_labels(final_root, split, prefix=prefix)
    return results
