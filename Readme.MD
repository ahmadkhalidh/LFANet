# Liver Fibrosis Detection System with Explainable AI

This repository contains a deep learning based system for liver fibrosis stage classification using medical images.  
The system integrates a convolutional neural network for prediction and Grad-CAM for visual explainability.  
An optional large language model component can be used to generate textual explanations of saliency maps for research analysis.

This project is intended strictly for **research and educational purposes**.

---

## Features

- Multiclass liver fibrosis classification (F0–F4)
- Custom CNN architecture trained on normalized image data
- Grad-CAM visualization for model interpretability
- Optional LLM-based explanation of saliency patterns
- Web-based interface using Flask or Streamlit
- Exportable prediction reports in PDF format

---

## Model Overview

- Input size: 128 × 128 × 3
- Architecture:
  - Convolutional layers with Batch Normalization
  - Depthwise separable convolutions
  - Residual and multi-scale feature aggregation
  - Global Average Pooling
  - Fully connected classification head
- Output: Softmax probabilities for five fibrosis stages

---

## Class Labels

| Label | Description |
|-----|------------|
| F0 | No fibrosis |
| F1 | Mild fibrosis |
| F2 | Moderate fibrosis |
| F3 | Severe fibrosis |
| F4 | Cirrhosis |

---

## Explainability

The system uses **Grad-CAM** to highlight image regions that contribute most to the model’s prediction.

The explainability pipeline:
1. Forward pass for prediction
2. Gradient computation on the last convolutional layer
3. Heatmap generation and overlay
4. Optional textual explanation via LLM

No diagnostic or clinical interpretation is produced.

---

## Project Structure

LiverFibrosisApp/
│
├── app.py # Flask or Streamlit application
├── livermodel.keras # Trained Keras model
├── saliency/
│ └── gradcam.py # Grad-CAM implementation
├── templates/
│ └── index.html # Web interface
├── static/
│ ├── css/
│ └── js/
├── uploads/ # Temporary uploaded images
├── outputs/
│ ├── cam/ # Grad-CAM images
│ └── reports/ # PDF reports
└── README.md

