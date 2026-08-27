# OOP API & Pruning Experiments Guide

This document provides the specific instructions and documentation required to run the Object-Oriented Programming (OOP) API for YOLOv11-Nano. These classes have been structured to match the established evaluation template (e.g., `ResNet18_CIFAR10`).

## 1. Environment Setup

Before running the experiments, ensure that all dependencies are installed. The custom YOLO loss functions (CIoU + DFL) and COCO metrics evaluation components require specific libraries.

```bash
pip install -r requirements.txt

(Key dependencies include tensorflow, keras-cv, opencv-python, and tqdm)

2. Repository & Directory Structure
To ensure the relative paths within the OOP classes work correctly, your local execution environment should match the following structure.

Note: The optimal baseline .h5 weights are provided in this repository. However, due to storage limits, the image datasets must be downloaded and placed manually in a datasets/ directory.

Plaintext
YoloModels/ (Root)
├── my_models/                             # YOLO11 architecture builder
├── oop/                                   # The API Wrapper Classes
│   ├── YOLO11_OxfordPets_Model.py
│   └── YOLO11_GlobalWheat_Model.py
├── trained_models/                        # Pre-trained baselines for pruning
│   └── yolo11n/
│       ├── oxford/oxford_yolo11n_pretrain.weights.h5
│       └── wheathead/yolo11_nano_Global_Wheat_Preatrained.weights.h5
├── weights/
│   └── yolo11n_coco_pretrained.weights.h5 # Initial COCO weights
├── datasets/                              #  MUST BE CREATED MANUALLY (See Sec. 3)
└── OOP_EXPERIMENTS.md                     # This documentation file
3. Local Dataset Configuration
Please download the datasets and place them in the datasets/ directory at the root of the repository exactly as shown below:

A. Oxford Pets Dataset
Plaintext
datasets/
└── oxford_pets/
    ├── images/               # Contains all .jpg images
    └── annotations/
        └── xmls/             # Contains all corresponding .xml bounding box files
B. Global Wheat Dataset
Plaintext
datasets/
└── global_wheat/
    ├── train/                # Contains all training .jpg images
    └── train.csv             # The original CSV annotations file
4. API Usage (Pruning Integration)
The API is built to bypass the primary train() method, as the baseline models are already fully trained and located in trained_models/.



Example Implementation
Python
# Choose the dataset you want to experiment on:
from oop.YOLO11_OxfordPets_Model import YOLO11_OxfordPets
# or: from oop.YOLO11_GlobalWheat_Model import YOLO11_GlobalWheat

# 1. Initialize the API
api = YOLO11_OxfordPets()

# 2. Load the baseline model and apply your pruning methodology
baseline_path = "trained_models/yolo11n/oxford/oxford_yolo11n_pretrain.weights.h5"
# ... [Insert your pruning logic here] ...
my_pruned_model = ... # The resulting pruned tf.keras.Model

# 3. Recover the pruned model
recovered_model, history = api.fine_tune(
    path_to_save="logs/recovered_model.weights.h5",
    new_model=my_pruned_model,
    new_fine_tune_epochs=5
)

# 4. Extract the monitoring metric (e.g., mAP_50)
final_metric = api.extract_monitoring_metric_value()
print(f"Final Retained Metric: {final_metric}")