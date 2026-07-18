# LFANet

**LFANet: A Lightweight Explainable AI Model for Liver Fibrosis Classification with Dilated Attention and Two-Stage Interpretation**

This repository contains the implementation of **LFANet**, a lightweight convolutional neural network for multi-stage liver fibrosis classification from ultrasound images. The framework integrates multi-scale dilated convolutions, depthwise separable convolutions, residual learning, self-attention, Grad-CAM visualization, and an optional post hoc large language model (LLM) for natural language explanation generation.

---

## Repository Structure

```
.
├── modelling/
│   ├── LFANet.ipynb
│   ├── LFANet_NoAttention.ipynb
│   ├── LFANet_NoResidual.ipynb
│   ├── LFANet_NoDepthwise.ipynb
│   ├── LFANet_NoDilated.ipynb
│   ├── LFANet_D111.ipynb
│   ├── LFANet_D123.ipynb
│   ├── LFANet_D125.ipynb
│   ├── LFANet_D135.ipynb
│   ├── ResNet50_Baseline.ipynb
│   ├── DenseNet121_Baseline.ipynb
│   ├── EfficientNetB0_Baseline.ipynb
│   └── MobileNetV2_Baseline.ipynb
├── saliency/
├── static/
├── templates/
├── app.py
├── livermodel.h5
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.10
- TensorFlow 2.15.1
- Keras
- NumPy
- OpenCV
- Scikit-learn
- Matplotlib
- Streamlit

Install the required packages using:

```bash
pip install -r requirements.txt
```

---

## Dataset

Experiments were conducted using the publicly available liver fibrosis ultrasound dataset released on Kaggle and originally reported by Joo et al.

---

## Experiments

The repository includes:

- Proposed LFANet model
- Architectural ablation studies
  - Without Attention
  - Without Residual Connection
  - Without Depthwise Separable Convolution
  - Without Dilated Convolution
- Dilation-rate analysis
  - (1,1,1)
  - (1,2,3)
  - (1,2,5)
  - (1,3,5)
- Baseline CNN models
  - ResNet50
  - DenseNet121
  - EfficientNetB0
  - MobileNetV2

All experiments were conducted using identical preprocessing, data augmentation, hyperparameter settings, and evaluation protocols.

---

## Explainability

Model predictions are interpreted using:

- Grad-CAM visualization
- Optional post hoc LLM-generated textual explanations

The LLM receives only the predicted fibrosis stage and confidence score. It does not access ultrasound images, Grad-CAM heatmaps, learned feature representations, or model parameters.

---

## Citation

If you use this repository, please cite:

<!-- > Ahmad, K. H., Kehinde, T. O., Salami, F. O., Petinrin, O. O., Nadarajah, S., & Akpan, J. *LFANet: A Lightweight Explainable AI Model for Liver Fibrosis Classification with Dilated Attention and Two-Stage Interpretation.* Scientific Reports (under review). -->

---

## License

This repository is provided for research and educational purposes.