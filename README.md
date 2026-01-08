# Facial Drowsiness Recognition Using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

A comprehensive deep learning system for driver drowsiness detection using facial analysis. This project implements and compares three state-of-the-art CNN architectures: VGG16, DenseNet-121, and LSTM-CNN hybrid for real-time drowsiness detection.

## 🎓 Project Overview

This Bachelor's Project (2023-2024) from the University of Southampton investigates driver drowsiness detection using deep learning techniques. The system analyzes facial features to identify signs of drowsiness, potentially preventing fatigue-related accidents.

### Key Features

- **Multiple CNN Architectures**: Comparative analysis of VGG16, DenseNet-121, and LSTM-CNN hybrid models
- **Temporal Modelling**: LSTM-CNN hybrid captures temporal dynamics of drowsiness progression
- **Transfer Learning**: Leverages ImageNet pre-trained weights for improved performance
- **Robustness Analysis**: Evaluation under real-world challenges (occlusion, illumination, blur)
- **High Accuracy**: LSTM-CNN achieves 96% accuracy with excellent F1-scores

## 📊 Results Summary

### Model Performance Comparison

| Model | Accuracy | Precision (Class 0) | Precision (Class 1) | Recall (Class 0) | Recall (Class 1) | F1-Score (Class 0) | F1-Score (Class 1) |
|-------|----------|---------------------|---------------------|------------------|------------------|--------------------|--------------------|| VGG16 | ~50% | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| DenseNet-121 | ~50% | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| **LSTM-CNN** | **96%** | **1.00** | **0.95** | **0.88** | **1.00** | **0.93** | **0.97** |

### ROC-AUC Scores

- **VGG16**: 0.51 (Random performance)
- **DenseNet-121**: 0.49 (Below random)
- **LSTM-CNN**: **0.97** (Excellent discrimination)

### Robustness Analysis (Modified Dataset)

- **VGG16**: ~80% accuracy (Most robust to real-world challenges)
- **DenseNet-121**: ~60% accuracy
- **LSTM-CNN**: ~60% accuracy

## 🛠️ Architecture Details

### 1. VGG16 Model
- **Base**: VGG16 pre-trained on ImageNet
- **Modifications**: 
  - GlobalAveragePooling2D layer
  - Dense layer (512 neurons, ReLU activation)
  - Dropout (0.5)
  - Output layer (1 neuron, sigmoid activation)
- **Training**: Learning rate 1e-5, Binary crossentropy loss

### 2. DenseNet-121 Model
- **Base**: DenseNet-121 pre-trained on ImageNet
- **Modifications**:
  - Flatten layer
  - BatchNormalization
  - Dense layer (256 neurons, He uniform initialization)
  - BatchNormalization
  - ReLU activation
  - Dropout (0.5)
  - Output layer (1 neuron, sigmoid activation)
- **Dense Connectivity**: Feature reuse for efficiency

### 3. LSTM-CNN Hybrid Model
- **Architecture**:
  - Time-distributed Conv2D layers (32, 64, 128 filters)
  - MaxPooling2D layers
  - Flatten layer
  - LSTM layer (64 units)
  - Dense layers (128, 64 neurons)
  - Dropout (0.5)
  - Output layer (1 neuron, sigmoid activation)
- **Temporal Modeling**: Captures drowsiness progression over time
- **Input**: Sequences of 10 frames

## 💾 Dataset

**Source**: Drowsiness Detection Dataset from Kaggle

**Classes**:
- Active Subjects (alert/awake)
- Fatigue Subjects (drowsy/fatigued)

**Data Split**: 70/30 train-test ratio

**Preprocessing**:
- Image resizing to 224x224
- Normalization (pixel values 0-1)
- Data augmentation (rotation, zoom, horizontal flip)

**Modified Dataset**: Created to test robustness
- Applied blur
- Applied occlusion
- Applied illumination changes

## ⚙️ Installation

### Prerequisites

```bash
Python 3.8+
TensorFlow 2.x
Keras
OpenCV
NumPy
Matplotlib
Scikit-learn
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/mukilan2101/facial-drowsiness-recognition-deep-learning.git
cd facial-drowsiness-recognition-deep-learning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle and organize as:
```
Dataset/
├── Active Subjects/
└── Fatigue Subjects/
```

## 🚀 Usage

### Training Models

The Jupyter notebook `Facial-Drowsiness-Recognition-Model-Performance-Comparison.ipynb` contains the complete implementation:

1. Data preprocessing and augmentation
2. Model architecture definitions
3. Training procedures
4. Evaluation and comparison
5. Robustness analysis

### Running the Notebook

```bash
jupyter notebook Facial-Drowsiness-Recognition-Model-Performance-Comparison.ipynb
```

## 📈 Key Findings

### 1. Temporal Modelling Superiority
- The LSTM-CNN hybrid significantly outperformed single-frame models
- Temporal information crucial for capturing drowsiness progression
- 96% accuracy demonstrates excellent discrimination capability

### 2. Single-Frame Model Limitations
- VGG16 and DenseNet-121 struggled with drowsiness detection on standard dataset
- Performance near random (~50% accuracy)
- Suggests need for temporal context in drowsiness detection

### 3. Robustness Trade-offs
- VGG16 showed best robustness to real-world challenges (80% accuracy)
- LSTM-CNN and DenseNet-121 more sensitive to image distortions
- Simple architectures may generalize better to noisy conditions

### 4. Hyperparameter Tuning Impact
- Learning rate scheduling improved convergence
- Dropout regularization essential for preventing overfitting
- Transfer learning accelerated training significantly

## 📁 Project Structure

```
facial-drowsiness-recognition-deep-learning/
├── Facial-Drowsiness-Recognition-Model-Performance-Comparison.ipynb
├── Report.pdf
├── requirements.txt
├── README.md
```

## 📝 Research Questions Addressed

1. **How does classification accuracy compare between different deep learning models?**
   - LSTM-CNN (96%) >> VGG16/DenseNet-121 (~50%)
   - Temporal modeling provides significant advantage

2. **Which model architecture is most robust to real-world challenges?**
   - VGG16 showed highest robustness (80% on modified dataset)
   - Trade-off between accuracy and robustness observed

3. **How does temporal modelling compare to single-frame classification?**
   - LSTM-CNN captures drowsiness progression over time
   - Superior performance demonstrates importance of temporal context
   - Single-frame models insufficient for complex drowsiness patterns

## 🔮 Future Work

- **Real-time Implementation**: Deploy models on edge devices
- **Multi-modal Fusion**: Combine facial analysis with physiological signals
- **Expanded Dataset**: Include diverse demographics and lighting conditions
- **Attention Mechanisms**: Investigate transformer-based architectures
- **Explainability**: Implement Grad-CAM for interpretability
- **Hybrid Approach**: Combine robustness of VGG16 with accuracy of LSTM-CNN

## 📖 Documentation

For detailed methodology, results, and analysis, refer to `Report.pdf`.

## 📊 Visualizations

The project includes comprehensive visualizations:
- Model architecture diagrams
- Training/validation accuracy curves
- ROC curves with AUC scores
- Confusion matrices
- Robustness comparison charts
- Sample drowsy/alert facial images

## ⚠️ Limitations

- Dataset may not cover all variations in lighting, angles, and demographics
- System relies on visible facial features (affected by occlusions)
- LSTM-CNN shows reduced robustness to image distortions
- Real-time performance not yet evaluated

## 🎯 Applications

- **Automotive Safety**: In-vehicle drowsiness monitoring systems
- **Transportation**: Bus and truck driver fatigue detection
- **Aviation**: Pilot alertness monitoring
- **Healthcare**: Patient monitoring in medical settings
- **Industrial Safety**: Heavy machinery operator monitoring

## 🚀 Deployment Considerations

- Model compression needed for edge deployment
- Real-time inference optimization required
- Privacy-preserving implementations essential
- Multi-camera setup for comprehensive coverage
- Integration with vehicle control systems

## 👥 Author

**Mukilan Raj**
- University of Southampton
- Bachelor's Project (2023-2024)
- Contact: [mukilan2101](https://github.com/mukilan2101)

## 🙏 Acknowledgments

- University of Southampton for project supervision
- Kaggle for providing the Drowsiness Detection Dataset
- TensorFlow and Keras teams for deep learning frameworks
- Research community for foundational work on drowsiness detection

## 📚 References

For complete references and citations, please refer to the full project report (`Report.pdf`).

## 💬 Citation

If you use this work in your research, please cite:

```bibtex
@misc{mukilan2024drowsiness,
  author = {Mukilan Raj},
  title = {Facial Drowsiness Recognition Using Deep Learning},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/mukilan2101/facial-drowsiness-recognition-deep-learning}}
}
```

---

**Note**: This project is for educational and research purposes. For production deployment, additional safety measures, testing, and regulatory compliance are required.
