from typing import Any, Union
import pandas as pd
import logging
import random
from .config import *
from . import access
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
# Set up logging
logger = logging.getLogger(__name__)

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded.
How are missing values encoded, how are outliers encoded? What do columns represent,
makes rure they are correctly labeled. How is the data indexed. Crete visualisation
routines to assess the data (e.g. in bokeh). Ensure that date formats are correct
and correctly timezoned."""

def check_country(final_root, splits=("train", "val"), country_keyword="kenya", sample_n=5):
    """Checking how many label files match a country keyword in each split,
    and previewing a sample of filenames — useful for confirming a merge
    actually landed the expected files before trusting downstream analysis."""
    results = {}

    for split in splits:
        lbl_dir = Path(f"{final_root}/{split}/labels")
        if not lbl_dir.exists():
            print(f"{split}: folder not found at {lbl_dir}")
            continue

        all_files = list(lbl_dir.glob("*.txt"))
        matched_files = [f for f in all_files if country_keyword.lower() in f.name.lower()]

        print(f"{split}: {len(all_files)} total label files, "
              f"{len(matched_files)} contain '{country_keyword}' (any case)")
        print("Sample filenames:", [f.name for f in all_files[:sample_n]])

        results[split] = {"total": len(all_files), "matched": len(matched_files)}

    return results
def data() -> Union[pd.DataFrame, Any]:
    """
    Load the data from access and ensure missing values are correctly encoded as well as
    indices correct, column names informative, date and times correctly formatted.
    Return a structured data structure such as a data frame.

    IMPLEMENTATION GUIDE FOR STUDENTS:
    ==================================

    1. REPLACE THIS FUNCTION WITH YOUR DATA ASSESSMENT CODE:
       - Load data using the access module
       - Check for missing values and handle them appropriately
       - Validate data types and formats
       - Clean and prepare data for analysis

    2. ADD ERROR HANDLING:
       - Handle cases where access.data() returns None
       - Check for data quality issues
       - Validate data structure and content

    3. ADD BASIC LOGGING:
       - Log data quality issues found
       - Log cleaning operations performed
       - Log final data summary

    4. EXAMPLE IMPLEMENTATION:
       df = access.data()
       if df is None:
           print("Error: No data available from access module")
           return None

       print(f"Assessing data quality for {len(df)} rows...")
       # Your data assessment code here
       return df
    """
    logger.info("Starting data assessment")

    # Load data from access module
    df = access.data()

    # Check if data was loaded successfully
    if df is None:
        logger.error("No data available from access module")
        print("Error: Could not load data from access module")
        return None

    logger.info(f"Assessing data quality for {len(df)} rows, {len(df.columns)} columns")

    try:
        # STUDENT IMPLEMENTATION: Add your data assessment code here

        # Example: Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Found missing values: {missing_counts.to_dict()}")
            print(f"Missing values found: {missing_counts.sum()} total")

        # Example: Check data types
        logger.info(f"Data types: {df.dtypes.to_dict()}")

        # Example: Basic data cleaning (students should customize this)
        # Remove completely empty rows
        df_cleaned = df.dropna(how="all")
        if len(df_cleaned) < len(df):
            logger.info(f"Removed {len(df) - len(df_cleaned)} completely empty rows")

        logger.info(f"Data assessment completed. Final shape: {df_cleaned.shape}")
        return df_cleaned

    except Exception as e:
        logger.error(f"Error during data assessment: {e}")
        print(f"Error assessing data: {e}")
        return None

def is_valid_label_line(parts):
    """Checking whether a label line is a valid box (5 fields) or
    valid polygon (class + even number of x,y pairs, so odd total count)."""
    if len(parts) == 5:
        return True
    if len(parts) >= 7 and len(parts) % 2 == 1:
        return True
    return False

def plot_metric_row(df, metrics, figsize=(18, 5)):
    """
    Plotting one row of metric subplots.
    `metrics` is a list of either:
      (train_col, val_col, title)  -> plots both lines
      (col, title)                  -> plots a single line
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    for ax, spec in zip(axes, metrics):
        if len(spec) == 3:
            train_col, val_col, title = spec
            if train_col in df.columns and val_col in df.columns:
                ax.plot(df["epoch"], df[train_col], label="train")
                ax.plot(df["epoch"], df[val_col], label="val")
                ax.set_title(title)
                ax.set_xlabel("Epoch")
                ax.legend()
                ax.grid(alpha=0.3)
            else:
                ax.set_visible(False)
        else:
            col, title = spec
            if col in df.columns:
                ax.plot(df["epoch"], df[col], color="darkorange", marker='o', markersize=2)
                ax.set_title(title)
                ax.set_xlabel("Epoch")
                ax.grid(alpha=0.3)
            else:
                ax.set_visible(False)

    plt.tight_layout()
    plt.show()
    
def is_box_or_polygon(final_root, splits=("train", "val"), expected_class_ids=(0,), verbose_n=20):
    """Checking every label file for structurally valid lines, correct
    class ids, and in-range coordinates. Handles both box-format (5 fields)
    and polygon-format (class + x,y pairs) lines correctly."""
    bad_lines = []

    for split in splits:
        lbl_dir = Path(f"{final_root}/{split}/labels")
        if not lbl_dir.exists():
            continue
        for f in lbl_dir.glob("*.txt"):
            with open(f) as fh:
                for i, line in enumerate(fh):
                    parts = line.split()
                    if not parts:
                        continue

                    if not is_valid_label_line(parts):
                        bad_lines.append((f, i, f"{len(parts)} fields"))
                        continue

                    cls = int(parts[0])
                    if cls not in expected_class_ids:
                        bad_lines.append((f, i, f"unexpected class id {cls}"))

                    coords = list(map(float, parts[1:]))
                    if any(v < 0 or v > 1 for v in coords):
                        bad_lines.append((f, i, "coordinate out of [0,1] range"))

                    # box-only sanity check: width/height must be positive
                    if len(parts) == 5:
                        w, h = coords[2], coords[3]
                        if w <= 0 or h <= 0:
                            bad_lines.append((f, i, "zero/negative width or height"))

    print(f"Found {len(bad_lines)} genuinely problematic label lines")
    for f, i, reason in bad_lines[:verbose_n]:
        print(f, i, reason)

    return bad_lines


def get_country(filename, countries=("Japan", "Czech", "India", "Kenya")):
    """Matching a filename prefix to its country label."""
    for c in countries:
        if filename.startswith(c + "_"):
            return c
    return "Unknown"


def collect_box_sizes(drive_root, splits=("train", "val"), countries=("Japan", "Czech", "India", "Kenya")):
    """Collecting (width_pct, height_pct, area_pct) per box, plus matching country labels."""
    sizes = []
    countries_list = []

    for split in splits:
        lbl_dir = Path(f"{drive_root}/final/{split}/labels")
        if not lbl_dir.exists():
            continue
        for f in lbl_dir.glob("*.txt"):
            country = get_country(f.name, countries)
            with open(f) as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    _, xc, yc, w, h = map(float, parts)
                    sizes.append((w * 100, h * 100, w * h * 100))
                    countries_list.append(country)

    return sizes, countries_list


def print_size_summary(sizes):
    """Printing overall width/height/area mean, min, max."""
    widths = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]
    areas = [s[2] for s in sizes]

    print(f"Total pothole boxes (final dataset): {len(sizes)}")
    print(f"Width  (% of image): mean={sum(widths)/len(widths):.1f}%, min={min(widths):.1f}%, max={max(widths):.1f}%")
    print(f"Height (% of image): mean={sum(heights)/len(heights):.1f}%, min={min(heights):.1f}%, max={max(heights):.1f}%")
    print(f"Area   (% of image): mean={sum(areas)/len(areas):.1f}%, min={min(areas):.1f}%, max={max(areas):.1f}%")

    return widths, heights, areas


def print_size_buckets(areas, small_thresh=1, large_thresh=9):
    """Printing small/medium/large box counts by area %."""
    small = sum(1 for a in areas if a < small_thresh)
    medium = sum(1 for a in areas if small_thresh <= a < large_thresh)
    large = sum(1 for a in areas if a >= large_thresh)

    print(f"\nSmall boxes (<{small_thresh}% area):  {small} ({small/len(areas)*100:.1f}%)")
    print(f"Medium boxes ({small_thresh}-{large_thresh}%):     {medium} ({medium/len(areas)*100:.1f}%)")
    print(f"Large boxes (>{large_thresh}%):       {large} ({large/len(areas)*100:.1f}%)")

    return {"small": small, "medium": medium, "large": large}


def group_by_country(countries_list, areas):
    """Grouping box areas by country into a dict of lists."""
    by_country = defaultdict(list)
    for c, a in zip(countries_list, areas):
        by_country[c].append(a)
    return by_country


def print_country_breakdown(by_country):
    """Printing per-country box counts, mean area, and % small boxes."""
    print("\n--- Per-country box counts and mean area ---")
    for c, vals in by_country.items():
        print(f"{c}: {len(vals)} boxes, mean area={sum(vals)/len(vals):.2f}%, "
              f"small={sum(1 for v in vals if v < 1)/len(vals)*100:.1f}%")


def plot_area_histogram(areas):
    """Plotting the overall box area distribution as a histogram."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(areas, bins=50, color='steelblue', edgecolor='black')
    ax.set_xlabel("Box area (% of image)")
    ax.set_ylabel("Count")
    ax.set_title("Pothole box area distribution (all countries)")
    plt.tight_layout()
    plt.show()


def plot_width_vs_height(widths, heights):
    """Plotting a width vs height scatter of all boxes."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(widths, heights, alpha=0.3, s=8)
    ax.set_xlabel("Width (% of image)")
    ax.set_ylabel("Height (% of image)")
    ax.set_title("Width vs Height of pothole boxes")
    plt.tight_layout()
    plt.show()


def plot_area_by_country(by_country):
    """Plotting overlaid area density histograms, one per country."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for c, vals in by_country.items():
        ax.hist(vals, bins=30, alpha=0.5, label=c, density=True)
    ax.set_xlabel("Box area (% of image)")
    ax.set_ylabel("Density")
    ax.set_title("Box area distribution by country")
    ax.legend()
    plt.tight_layout()
    plt.show()
def subsample_negatives(root, country, keep_ratio=1.5, seed=42):
    """
    Reduce the number of background-only images by randomly removing
    excess negative samples while preserving all positive samples.

    Parameters
    ----------
    root : str or Path
        Root directory of the RDD2020 dataset.
    country : str
        Country subset (Japan, Czech, India).
    keep_ratio : float
        Number of negative samples to retain per positive sample.
    seed : int
        Random seed for reproducibility.
    """

    random.seed(seed)

    img_dir = Path(root) / "train" / country / "images"
    lbl_dir = Path(root) / "train" / country / "labels"

    pos_files = [f for f in lbl_dir.glob("*.txt") if f.stat().st_size > 0]
    neg_files = [f for f in lbl_dir.glob("*.txt") if f.stat().st_size == 0]

    n_keep = min(int(len(pos_files) * keep_ratio), len(neg_files))
    neg_to_drop = random.sample(neg_files, len(neg_files) - n_keep)

    for lbl in neg_to_drop:
        lbl.unlink()

        img = img_dir / f"{lbl.stem}.jpg"
        if img.exists():
            img.unlink()

    print(
        f"{country}: "
        f"{len(pos_files)} positive, "
        f"{n_keep} negatives kept, "
        f"{len(neg_to_drop)} negatives removed"
    )

import hashlib


def file_hash(path):
    """Computing an MD5 hash of a file's contents."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def check_train_val_overlap(final_root, sample_n=10):
    """Checking for duplicate images between train and val by content hash —
    catches data leakage from re-running merge/split scripts without
    clearing the destination first."""
    train_dir = Path(f"{final_root}/train/images")
    val_dir = Path(f"{final_root}/val/images")

    train_hashes = {file_hash(f): f for f in train_dir.glob("*")}
    val_hashes = {file_hash(f): f for f in val_dir.glob("*")}

    overlap = set(train_hashes) & set(val_hashes)
    print(f"Duplicate images between train and val: {len(overlap)}")
    for h in list(overlap)[:sample_n]:
        print(train_hashes[h], val_hashes[h])

    return overlap

def copy_split(
    src_root,
    src_split,
    dst_root,
    dst_split,
    prefix="",
    val_ratio=None,
    seed=42,
):
    """
    Copying a YOLO dataset split into a unified destination dataset.

    If val_ratio is None:
        Copies the entire src_split as-is into dst_root/dst_split
        (use this for datasets that are already pre-split, e.g. Roboflow's
        train/valid/test).

    If val_ratio is given:
        Randomly splits src_split into train/val before copying, writing
        results into dst_root/train and dst_root/val directly
        (use this for datasets that need splitting, e.g. RDD2020's
        per-country folders). dst_split is ignored in this mode.
    """
    src_img_dir = Path(src_root) / src_split / "images"
    src_lbl_dir = Path(src_root) / src_split / "labels"

    img_files = list(src_img_dir.glob("*.jpg"))
    label_prefix = f"{prefix}_" if prefix else ""

    if val_ratio is None:
        # Direct copy, no splitting
        dst_img_dir = Path(dst_root) / dst_split / "images"
        dst_lbl_dir = Path(dst_root) / dst_split / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for f in img_files:
            lbl = src_lbl_dir / f"{f.stem}.txt"
            if not lbl.exists():
                continue
            shutil.copy(f, dst_img_dir / f"{label_prefix}{f.name}")
            shutil.copy(lbl, dst_lbl_dir / f"{label_prefix}{f.stem}.txt")
            count += 1

        print(f"{prefix or src_split} -> {dst_split}: {count} images")

    else:
        # Split into train/val, then copy each into its own destination
        random.seed(seed)
        random.shuffle(img_files)

        n_val = int(len(img_files) * val_ratio)
        val_files = img_files[:n_val]
        train_files = img_files[n_val:]

        dst_train_img = Path(dst_root) / "train" / "images"
        dst_train_lbl = Path(dst_root) / "train" / "labels"
        dst_val_img = Path(dst_root) / "val" / "images"
        dst_val_lbl = Path(dst_root) / "val" / "labels"
        for d in [dst_train_img, dst_train_lbl, dst_val_img, dst_val_lbl]:
            d.mkdir(parents=True, exist_ok=True)

        for f in train_files:
            lbl = src_lbl_dir / f"{f.stem}.txt"
            if not lbl.exists():
                continue
            shutil.copy(f, dst_train_img / f"{label_prefix}{f.name}")
            shutil.copy(lbl, dst_train_lbl / f"{label_prefix}{f.stem}.txt")

        for f in val_files:
            lbl = src_lbl_dir / f"{f.stem}.txt"
            if not lbl.exists():
                continue
            shutil.copy(f, dst_val_img / f"{label_prefix}{f.name}")
            shutil.copy(lbl, dst_val_lbl / f"{label_prefix}{f.stem}.txt")

        print(f"{prefix or src_split}: {len(train_files)} train, {len(val_files)} val")
        
def query(data: Union[pd.DataFrame, Any]) -> str:
    """Request user input for some aspect of the data."""
    raise NotImplementedError


def view(data: Union[pd.DataFrame, Any]) -> None:
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError


def labelled(data: Union[pd.DataFrame, Any]) -> Union[pd.DataFrame, Any]:
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError
