# Deepfake Detection

Comprehensive pipeline for reliable deepfake image identification, developed for the “Face the Future” ML Challenge

## Project Overview

This project implements a robust deepfake detection model using transfer learning with ResNet-18, focusing on classifying images as real or fake. The goals were reproducibility, strong generalization, and deployment-readiness

## Features

- Transfer learning with ResNet-18 for efficient feature extraction.
- Supports flexible, reproducible training and evaluation workflows.
- Handles data imbalance using a weighted sampler.
- Early stopping and checkpointing for stable model selection.
- Visual performance analysis: ROC AUC, Confusion Matrix, Precision–Recall curves.
- Modular code for preprocessing, training, prediction, and metrics.
- Documentation of challenges and strategies (e.g., imbalance, overfitting, calibration).

## Directory Structure


    ```
    ML Code/
	├── data/
	│   ├── train/
	│   │   ├── real/              # Real training images
	│   │   ├── fake/              # Fake training images
	│   ├── test/                  # Unlabeled test images
	│   ├── real_cifake_preds.json # Real image labels
	│   └── fake_cifake_preds.json # Fake image labels
	│
	├── src/
	│   ├── dataset.py             # Dataset loading and preprocessing
	│   ├── train.py               # Model training and early stopping
	│   ├── predict.py             # Test inference and submission generation
	│   ├── metrics.py             # Validation metrics computation
	│   └── visualize_metrics.py   # ROC, PR, and confusion matrix plots
	│
	├── output/
	│   ├── model.pth              # Final trained model
	│   ├── best_model.pth         # Best validation checkpoint
	│   ├── submission.json        # Final test predictions
	│   ├── confusion_matrix.png   # Confusion Matrix
	│   ├── roc.png                # ROC Curve
	│   └── pr_curve.png           # Precision–Recall Curve
	│
	├── report.tex                 # LaTeX report
	├── requirements.txt           # Python dependencies
	└── README.md                  # Project documentation
    ```


## Installation

1. **Clone the repository**

    ```
    git clone https://github.com/secretcoder45/ML-Code.git
    cd ML-Code
    ```

2. **Create virtual environment & install dependencies**

    ```
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
[attached_file:1]    

## Usage

- **Train the model**

    ```
    python src/train.py
    ```

- **Predict on test set**

    ```
    python src/predict.py
    ```

- **Analyze and visualize metrics**

    ```
    python src/metrics.py
    python src/visualizemetrics.py
    ```

    Output will be stored in `/output`.
[attached_file:1]    

## Model Details

- **Primary Architecture**: ResNet-18 (pretrained on ImageNet)
- **Augmentations**: Horizontal flip, color jitter, small rotations
- **Optimizer**: Adam; **Loss**: BCEWithLogitsLoss
- **Batch Size**: 32; **Image Size**: 224×224
[attached_file:1]    

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-score, ROC AUC
- Confusion Matrix, Precision–Recall Curve

Validation performance (held-out set):

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 97.75%  |
| Precision | 99.03%  |
| Recall    | 96.68%  |
| F1 Score  | 97.84%  |

## Dependencies

- Python
- torch==2.3.0
- torchvision==0.17.0
- numpy
- pandas
- matplotlib
- scikit-learn
- pillow
- tqdm
- albumentations
[attached_file:1]

## Future Work

- Explore larger backbones (EfficientNet, Vision Transformers)
- Add explainability with Grad-CAM or Integrated Gradients
- Extend to video-level deepfake detection

## License

MIT License

## Attribution

- Model: ResNet-18 from torchvision
- Challenge: Synergy25 “Face the Future” ML Challenge
- Special thanks to dataset providers and competition organizers.

## Contact

Author: Palash Garg  
Institute: IIT Guwahati  
📧 palashgarg45@gmail.com  
📎 LinkedIn: https://www.linkedin.com/in/palash-garg-003014345/  
