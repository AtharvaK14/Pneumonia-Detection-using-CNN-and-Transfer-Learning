# Pneumonia Detection using CNN and Transfer Learning 🧠🩻

## 📌 Project Overview

This project applies Convolutional Neural Networks (CNNs) with **Transfer Learning** to detect **pneumonia** from chest X-ray images. By leveraging pre-trained deep learning models, the system classifies X-ray images into **Pneumonia** or **Normal**, with the goal of improving early and efficient diagnosis.

The model aims to reduce the diagnostic burden on clinicians, particularly in situations such as the COVID-19 pandemic, by offering an assistive tool for image-based screening.

---

## 🚀 Key Features

- 📸 **Image Classification**: Binary classification of chest X-rays into Pneumonia and Normal.
- 🔁 **Transfer Learning**: Uses pre-trained CNN models (e.g., VGG16, ResNet50) for high performance on small datasets.
- ⚖️ **Handles Data Imbalance**: Includes data preprocessing techniques and augmentation to balance dataset classes.
- 📈 **Visualization Tools**: Accuracy/loss graphs, confusion matrix, and performance reports.

---

## 📂 Dataset

The dataset used consists of chest X-ray images sourced from open datasets available on [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

- Two folders: `train/`, `val/`, and `test/`
- Two classes per folder: `NORMAL/` and `PNEUMONIA/`

> **Note:** Dataset is not included in this repo due to size. Download it from Kaggle and place it inside a `/data/` directory.

---

## 🧠 Model Architecture

- **Input Layer**: Image resizing and normalization
- **Convolution Blocks**: Using pretrained model layers (e.g., VGG16 or ResNet50)
- **Custom Classifier Head**:
  - Flatten
  - Dense layer with ReLU activation
  - Dropout for regularization
  - Output layer with Sigmoid activation (binary classification)

---

## 🛠️ Technologies & Libraries

- Python 3.x
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

---

## ⚙️ How to Run This Project

### 📦 Step 1: Clone the Repo

```bash
git clone https://github.com/yourusername/pneumonia-detection-cnn.git
cd pneumonia-detection-cnn
````

### 🧱 Step 2: Set Up Environment

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

> If `requirements.txt` is not available, install manually:

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn
```

### 🗃️ Step 3: Prepare Dataset

1. Download dataset from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
2. Place it like so:

```
project-root/
├── data/
│   ├── train/
│   ├── val/
│   └── test/
```

### 🚀 Step 4: Train the Model

```bash
python train_model.py
```

> The script handles preprocessing, model creation, training, evaluation, and saving results.

---

## 🧪 Evaluation Metrics

* Accuracy
* Precision, Recall, F1-score
* Confusion Matrix
* Training vs. Validation Loss/Accuracy graphs

---

## 📊 Results

* The best model achieved \~94% accuracy on the test dataset.
* Transfer learning significantly reduced training time and improved generalization.
* Proper preprocessing and augmentation improved performance on imbalanced classes.

---

## 📚 References

* [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/guide/keras/transfer_learning)
* [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
* [CNN Tutorial - Mr. D. Bourke](https://github.com/mrdbourke/tensorflow-deep-learning)

---

*For educational and research use only.*
