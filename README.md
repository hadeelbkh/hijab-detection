# Hijab Detection Project

📄 **Paper:** M. Obaid, H. Bkhaitan, D. Maali, S. Hammad, T. Thaher, *An Identity-Aware Privacy-Preserving Deep Learning Framework for Culturally Sensitive Image Sharing*, IJACSA 17(4), 909–919, 2026. ([link](https://thesai.org/Publications/ViewPaper?Volume=17&Issue=4&Code=IJACSA&SerialNo=88))

🤗 **Model:** [hadeelbkh/sitr-hijab-densenet121](https://huggingface.co/hadeelbkh/sitr-hijab-densenet121) · 

**Dataset (gated):** [hadeelbkh/sitr-hijab-dataset](https://huggingface.co/datasets/hadeelbkh/sitr-hijab-dataset) · 

**Demo:** [hadeelbkh/sitr-hijab-detection](https://huggingface.co/spaces/hadeelbkh/sitr-hijab-detection)


## Overview
This project implements a hijab detection system using deep learning. It uses a DenseNet121 model, fine-tuned for binary classification to detect whether an individual is wearing a hijab in an image. The labels are defined as:

- `0`: Hijab present
- `1`: No hijab present

The model outputs a single logit; `sigmoid(logit)` is the probability of class `1` (**no hijab**).

The project includes scripts for training, evaluation, inference on single images, and batch processing. It was developed and tested using Google Colab, with data stored locally and on Google Drive.

## Features

- **Model**: DenseNet121, fine-tuned for binary classification.
- **Training**: Supports training on a custom dataset with configurable hyperparameters.
- **Evaluation**: Computes metrics like accuracy, precision, recall, and F1-score.
- **Inference**: Processes single images or batches of images for prediction.
- **Environment**: Compatible with Google Colab and local setups (Windows/Linux).

## Dataset
The dataset used in this project is sourced from Roboflow: Hijab Dataset by Team S. However, the original dataset had several issues:

- Some images had no objects detected, making them unsuitable for training or evaluation.
- Certain images were misclassified, such as hijabi individuals labeled as "No Hijab."

These issues were addressed by manually reviewing and correcting the dataset. The corrected dataset is stored in the `sitr_data/` folder of this project, which contains the full dataset without the aforementioned problems.

## Dataset Structure and Labeling

The `sitr_data/` folder is organized into three subdirectories:
- `sitr_data/train/`
- `sitr_data/val/`
- `sitr_data/test/`


Each subdirectory contains images and corresponding XML annotation files.
Labels are specified in the XML files, where the `<name>` tag indicates:
- `0`: Hijabi (wearing a hijab)
- `1`: Non-hijabi (not wearing a hijab)



## Prerequisites

- **Python**: Version 3.8 or higher.
- **Dependencies**:
    - `torch`
    - `torchvision`
    - `pandas`
    - `scikit-learn`
    - `pillow`

- **Hardware**: GPU recommended for faster training (e.g., Google Colab with GPU runtime).
- **Dataset**: The corrected dataset in `sitr_data/` as described above.

## Setup Instructions
### Local Setup (Windows/Linux)

1. **Clone the Repository:**
```
git clone https://github.com/hadeelbkh/hijab-detection.git
cd hijab-detection
```

2. **Install Dependencies:**
```
pip install -r requirements.txt
```

If `requirements.txt` is missing, install manually:
```
pip install torch torchvision pandas scikit-learn pillow
```

3. **Prepare Dataset:**

    - Ensure the `sitr_data/` directory is in the project root with the structure mentioned above.
    - Images should be in `.jpg` format, each accompanied by an XML annotation file.



### Google Colab Setup

1. **Upload to Google Drive:**

    - Upload the `hijab-detection` directory to your Google Drive (e.g., `/content/drive/MyDrive/hijab-detection/`).

2. **Mount Google Drive:In a Colab notebook:**
```
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/hijab-detection/
```

3. **Install Dependencies:**
```
!pip install torch torchvision pandas scikit-learn pillow
```

4. **Prepare Dataset:**

    - Ensure the `sitr_data/` directory is in `/content/drive/MyDrive/hijab-detection/sitr_data/`.


## Usage
### Training the Model

1. **Imoprt run_traning from utils_colab:**
```
from src.utils_colab import run_training
```

2. **Call run_training on the densenet121 model with X runs:**
```
run_training(num_runs=4, model_name="densenet121")
```

- This trains the DenseNet121 model and saves the best checkpoint as `data/model/densenet121/densenet121_run_{run_id}/best_model.pt`.
- Training logs and metrics are saved in `model/densenet121/densenet121_run_{run_id}`.

3. **Hyperparameters:**

    - Edit `src/config.py` to adjust:
        - `BATCH_SIZE`: 4
        - `EPOCHS`: 20
        - `LEARNING_RATE`: 1e-5
        - `WEIGHT_DECAY` = 1e-6
        - `DROPOUT`: 0.3

### Evaluating the Model

- The Evaluation Script will be called while `src/train.py` is running, after each run finishes training
- This evaluates the model on the test set and saves metrics to `model/densenet121/densenet121_run_{run_id}/test_metrics_{run_id}.csv`.

### Inference on a Single Image

1. **Prepare an Image:**

    - Place the image (e.g., `hijabigirl.jpg`) in `images/`.

2. **Run Inference:**
```
!python /content/drive/MyDrive/hijab-detection/inference.py
```

Edit `inference.py` to set the image path:
```
main(image_path="images/hijabigirl.jpg")
```

### Batch Inference

- Edit `inference.py` to set the images directory to evaluate a batch of images:
```
main(image_dir="images/")
```

## File Structure
```
hijab-detection/
├── images/                  # Directory for inference images
├── model/
│   └── densenet121/         # Model checkpoints and metrics
├── sitr_data/               # Corrected dataset directory
│   ├── train/
│   ├── val/
│   └── test/
├── src/
│   ├── config.py           # Configuration and hyperparameters
│   ├── dataset.py          # Data loading utilities
│   ├── model.py            # Model architecture
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   ├── utils.py            # Utility functions
│   └── utils_colab.py      # Run training and plotting in Colab
├── inference.py            # Inference script for single images
├── README.md               # Project documentation
└── requirements.txt
```

## Results

Final model (tuned DenseNet121, released on Hugging Face) on the 880-image test set.
Precision, recall and F1 are for the positive class `1 = no hijab`:

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 92.16% | 91.63% | 81.72% | 86.39% | 0.967 |

See the [model card](https://huggingface.co/hadeelbkh/sitr-hijab-densenet121) for per-class results, the confusion matrix and the trade-offs at different thresholds.


- **Single Image Example:**

    - **Image:** `hijabigirl.jpg`
    - **Prediction:** Hijab
    - **Confidence:** 0.9954
    - **Probability (of no hijab):** 0.0046


## Contributing

- Feel free to fork the repository and submit pull requests.
- For issues, create a new issue on the GitHub repository.

## License
The code is licensed under the MIT License. The dataset is derived from the Roboflow [Hijab Dataset by Team S](https://universe.roboflow.com/team-s/hijab-dataset-nw57d/dataset/4) and is released under CC BY 4.0.

## Citation
```bibtex
@article{obaid2026sitr,
  title   = {An Identity-Aware Privacy-Preserving Deep Learning Framework for Culturally Sensitive Image Sharing},
  author  = {Obaid, Mahmoud and Bkhaitan, Hadeel and Maali, Duha and Hammad, Saja and Thaher, Thaer},
  journal = {International Journal of Advanced Computer Science and Applications},
  volume  = {17},
  number  = {4},
  pages   = {909--919},
  year    = {2026},
  doi     = {10.14569/IJACSA.2026.0170488}
}
```

## Acknowledgments

- Built with PyTorch and torchvision.
- Dataset sourced from Roboflow.
- Tested on Google Colab as of May 30, 2025.

