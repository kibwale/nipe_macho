from typing import Any, Union
import pandas as pd
import logging
import random
from .config import *
from . import access
import shutil
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



from pathlib import Path


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
