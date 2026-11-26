# 🧠 Brain Tumor Detection Using Deep Learning

A deep learning-based project for classifying brain tumors from MRI images using Convolutional Neural Networks (CNNs) and transfer learning.

## 📌 Project Overview

Brain Tumor Detection is a medical image classification project that uses deep learning techniques to identify different types of brain tumors from MRI scans.

The project focuses on training and evaluating deep learning models to classify MRI images into four categories:

- Glioma
- Meningioma
- No Tumor
- Pituitary

The objective is to explore the performance of different deep learning architectures for automated brain MRI classification.

## 🎯 Objectives

- Develop a deep learning-based brain tumor classification system.
- Classify MRI images into four categories.
- Train and evaluate different deep learning models.
- Compare model performance using evaluation metrics.
- Develop a user-friendly interface for MRI image classification.

## 🧠 Deep Learning Models

The project explores the following architectures:

1. **Dilated CNN** – Uses dilated convolutions to capture image features at different scales.
2. **ResNet-50** – Uses residual connections to support deep neural network training.
3. **EfficientNetB4** – Uses compound scaling to balance model depth, width and resolution.

## 📂 Dataset Description

The dataset contains brain MRI images organized into four classes:

| Class | Training Images | Testing Images |
|---|---:|---:|
| Glioma | 1,400 | 400 |
| Meningioma | 1,400 | 400 |
| No Tumor | 1,400 | 400 |
| Pituitary | 1,400 | 400 |
| **Total** | **5,600** | **1,600** |

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit
- Google Colab

## 📊 EfficientNetB4 Benchmark Results

The recorded EfficientNetB4 benchmark on 1,600 test images:

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 91.312% |
| Macro Precision | 91.765% |
| Macro Recall | 91.312% |
| Macro F1-Score | 91.031% |
| Weighted F1-Score | 91.031% |
| Macro ROC-AUC | 98.376% |
| Mean Inference Time | 618.282 ms/image |
| Throughput | 1.617 images/sec |
| Parameters | 17,903,843 |
| Model Size | 71.638 MB |

*Results are from the recorded EfficientNetB4 benchmark. Performance may vary with hardware and execution environment.*

## 📁 Project Structure

```text
Brain_Tumour_Detection/
│
├── README.md
├── Brain_Tumor_Detection.ipynb
├── brain_tumor_detection.py
├── app.py
├── requirements.txt
│
├── models/
│   └── efficientnetb4_best.keras
│
└── results/
    ├── confusion_matrix.png
    └── training_graph.png
```

*The model and result files should be added to the repository when available.*

## ⚙️ Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/Vsb21/Brain_Tumour_Detection.git
```

### 2. Open the project folder

```bash
cd Brain_Tumour_Detection
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

Upload an MRI image to view the model's predicted class and confidence.

## 🚀 Future Scope

- Improve classification accuracy.
- Explore additional deep learning architectures.
- Optimize model inference speed.
- Develop a more comprehensive web-based interface.
- Evaluate the system on additional MRI datasets.

## ⚠️ Disclaimer

This project is developed for academic and educational purposes only. Its predictions are not a medical diagnosis and should not replace professional medical advice or radiological examination.

## 👨‍💻 Author

**Vaibhav Bhagat**

Electronics and Telecommunication Engineering

GitHub: [Vsb21](https://github.com/Vsb21)

---

⭐ If you find this project useful, consider giving the repository a star.
