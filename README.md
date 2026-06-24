# Nipe Macho : Automated Road Infrastructure Assessment

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v8%20%7C%20v11%20%7C%20v26-purple)](https://github.com/ultralytics/ultralytics)

## Overview

Road potholes pose significant challenges to transportation infrastructure, contributing to vehicle damage, increased maintenance costs, traffic disruptions, and road safety hazards. Traditional manual road inspection methods rely heavily on physical surveys, which are often time-consuming, labor-intensive, and inefficient for large road networks. 

This project automates road defect detection using computer vision to be used on edge devices. Specifically, it explores, trains, and benchmarks multiple generations of **YOLO (You Only Look Once)** architectures to identify the most efficient approach for real-time inference on Kenyan roads. The system evaluates surface defects, extracts embedded GPS metadata, and logs hazard locations on an interactive monitoring dashboard.

## Methodology & Pipeline

This repository is structured around a three-phase deep learning and telematics pipeline:

1. **Automated Annotation (Knowledge Distillation):** 
   Instead of manual labeling, the pipeline utilizes `autodistill` and the **Grounded SAM 2** (Segment Anything) foundation model. By using zero-shot text prompts, it automatically generates bounding boxes and YOLO-formatted labels for unannotated road imagery (Chepalungu dataset).

2. **Deep Transfer Learning & Evaluation:** 
   The core analysis involves fine-tuning pre-trained weights and systematically benchmarking three YOLO architectures for edge deployment:
   * **YOLOv8n:** The lightweight, resource-constrained baseline.
   * **YOLOv11:** Evaluated for its Cross-Stage Partial Spatial Attention modules.
   * **YOLOv26:** The state-of-the-art, NMS-Free architecture utilizing Small Target Aware Labeling (STAL) to detect distant anomalies with low latency.

3. **Telematics & Deployment:**
   A modern React-based web dashboard allowing users to upload images, run inference via an API, extract browser-based Geolocation data, and map the coordinates of detected defects on an interactive geospatial interface.

## Repository Structure

```text
├── fynesse/ # not implimented 
│   ├── access.py        # Data procurement and loading logic
│   ├── assess.py        # Data quality checks and EDA
│   ├── address.py       # YOLO model training, inference, and evaluation
│   ├── config.py        # Configuration management
│   └── defaults.yml     # Default configuration values (paths, YOLO hyperparameters)
├── notebooks/
│   ├── data_annotation.ipynb            # Do annotations for rest of dataset images (not implimented)
│   ├── potholes.ipynb                   # Baseline experimental notebook
│   ├── potholes_v8n.ipynb               # YOLOv8n training & evaluation
│   ├── potholes_v11.ipynb               # YOLOv11 training & evaluation
│   └── potholes_v26.ipynb               # YOLOv26 training & evaluation
├── .gitignore                           # Git ignore rules
└── README.md                            # Project documentation
