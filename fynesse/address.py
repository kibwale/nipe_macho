"""
Address module for the fynesse framework.

This module handles question addressing functionality including:
- Statistical analysis
- Predictive modeling
- Data visualization for decision-making
- Dashboard creation
"""

from typing import Any, Union
import pandas as pd
import logging
import os
import glob
import random
from IPython.display import display, Image
import time
from pathlib import Path
from ultralytics import YOLO

# Set up logging
logger = logging.getLogger(__name__)

# Here are some of the imports we might expect
# import sklearn.model_selection  as ms
# import sklearn.linear_model as lm
# import sklearn.svm as svm
# import sklearn.naive_bayes as naive_bayes
# import sklearn.tree as tree

# import GPy
# import torch
# import tensorflow as tf

# Or if it's a statistical analysis
# import scipy.stats




def measure_inference_time(weights_path, test_images_path, n_images=50, imgsz=800):
    """Measuring average inference time (ms/image) for one model,
    using Ultralytics' own speed breakdown for a cleaner, more accurate number
    than wall-clock timing (which includes Python loop overhead)."""
    model = YOLO(weights_path)
    img_files = list(Path(test_images_path).glob("*.jpg"))[:n_images]

    model.predict(img_files[0], imgsz=imgsz, verbose=False)  # warm-up, excluded

    inference_times = []
    for img in img_files:
        results = model.predict(img, imgsz=imgsz, verbose=False)
        inference_times.append(results[0].speed["inference"])

    avg_ms = sum(inference_times) / len(inference_times)
    return avg_ms


def compare_inference_across_models(weights_paths, test_images_path, imgsz=800, n_images=50):
    """Running measure_inference_time across multiple models at one imgsz."""
    summary = []
    for name, wpath in weights_paths.items():
        avg_ms = measure_inference_time(wpath, test_images_path, n_images=n_images, imgsz=imgsz)
        summary.append({"model": name, "avg_inference_ms": avg_ms})
        print(f"{name}: {avg_ms:.2f} ms/image")

    return pd.DataFrame(summary)
def analyze_data(data: Union[pd.DataFrame, Any]) -> dict[str, Any]:
    """
    Address a particular question that arises from the data.

    IMPLEMENTATION GUIDE FOR STUDENTS:
    ==================================

    1. REPLACE THIS FUNCTION WITH YOUR ANALYSIS CODE:
       - Perform statistical analysis on the data
       - Create visualizations to explore patterns
       - Build models to answer specific questions
       - Generate insights and recommendations

    2. ADD ERROR HANDLING:
       - Check if input data is valid and sufficient
       - Handle analysis failures gracefully
       - Validate analysis results

    3. ADD BASIC LOGGING:
       - Log analysis steps and progress
       - Log key findings and insights
       - Log any issues encountered

    4. EXAMPLE IMPLEMENTATION:
       if data is None or len(data) == 0:
           print("Error: No data available for analysis")
           return {}

       print("Starting data analysis...")
       # Your analysis code here
       results = {"sample_size": len(data), "analysis_complete": True}
       return results
    """
    logger.info("Starting data analysis")

    # Validate input data
    if data is None:
        logger.error("No data provided for analysis")
        print("Error: No data available for analysis")
        return {"error": "No data provided"}

    if len(data) == 0:
        logger.error("Empty dataset provided for analysis")
        print("Error: Empty dataset provided for analysis")
        return {"error": "Empty dataset"}

    logger.info(f"Analyzing data with {len(data)} rows, {len(data.columns)} columns")

    try:
        # STUDENT IMPLEMENTATION: Add your analysis code here

        # Example: Basic data summary
        results = {
            "sample_size": len(data),
            "columns": list(data.columns),
            "data_types": data.dtypes.to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "analysis_complete": True,
        }

        # Example: Basic statistics (students should customize this)
        numeric_columns = data.select_dtypes(include=["number"]).columns
        if len(numeric_columns) > 0:
            results["numeric_summary"] = data[numeric_columns].describe().to_dict()

        logger.info("Data analysis completed successfully")
        print(f"Analysis completed. Sample size: {len(data)}")

        return results

    except Exception as e:
        logger.error(f"Error during data analysis: {e}")
        print(f"Error analyzing data: {e}")
        return {"error": str(e)}



def load_trained_model(results_dir):
    """Loading the best weights from a completed training run."""
    best_weights = os.path.join(results_dir, 'weights', 'best.pt')
    if not os.path.exists(best_weights):
        raise FileNotFoundError(f"No best.pt found at {best_weights}")
    print(f"Loaded model from: {best_weights}")
    return YOLO(best_weights)


def run_inference(model, test_images_path, conf=0.25, save=True):
    """Running prediction on a folder of test images."""
    print(f"Running predictions on: {test_images_path}")
    results = model.predict(source=test_images_path, save=save, conf=conf)
    return results


def find_latest_predict_dir(pattern='runs/detect/predict*'):
    """Finding the most recently created prediction output folder."""
    predict_dirs = glob.glob(pattern)
    if not predict_dirs:
        print("No prediction directory found. Check if model.predict() succeeded.")
        return None
    return max(predict_dirs, key=os.path.getmtime)


def display_random_predictions(predict_dir, n=3, width=800):
    """Displaying a random sample of predicted images from a prediction folder."""
    if predict_dir is None:
        return

    predicted_images = glob.glob(f'{predict_dir}/*.jpg')
    if not predicted_images:
        print("No predicted images found inside the prediction folder.")
        return

    print(f"\n --- Displaying Random Predictions from {predict_dir} ---")
    sample_images = random.sample(predicted_images, min(n, len(predicted_images)))
    for img_path in sample_images:
        display(Image(filename=img_path, width=width))


def predict_and_show(results_dir, test_images_path, conf=0.25, n_display=3):
    """Running the full predict-and-visualize pipeline end to end."""
    model = load_trained_model(results_dir)
    run_inference(model, test_images_path, conf=conf)
    latest_predict = find_latest_predict_dir()
    display_random_predictions(latest_predict, n=n_display)
    return model
