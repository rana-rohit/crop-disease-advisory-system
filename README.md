# Smart Crop Disease Detection and Advisory System

## Overview

The Smart Crop Disease Detection and Advisory System is an AI-powered web application designed to assist farmers, agricultural professionals, and researchers in identifying plant diseases from leaf images. The system leverages deep learning techniques to automatically classify crop diseases and provide relevant advisory information, including symptoms, treatment recommendations, and preventive measures.

The project combines computer vision, deep learning, and web technologies to create an accessible tool that can support early disease detection and improve crop management practices.

---

# Objectives

The primary objectives of this project are:

* Detect crop diseases from leaf images using deep learning.
* Provide accurate disease classification across multiple crop species.
* Offer actionable agricultural advisory information.
* Reduce dependency on manual disease diagnosis.
* Create a user-friendly web-based platform accessible through modern browsers.
* Demonstrate the application of Artificial Intelligence in agriculture.

---

# Problem Statement

Plant diseases are a major cause of crop loss worldwide. Traditional disease identification often requires expert knowledge, laboratory testing, or field inspections, which may not be readily available to all farmers.

This project addresses the problem by developing an automated disease detection system capable of:

* Identifying diseases from crop leaf images.
* Providing instant predictions.
* Suggesting treatment and prevention measures.
* Assisting users in making informed agricultural decisions.

---

# Features

## Disease Detection

* Upload crop leaf images.
* Automatic disease prediction using a trained deep learning model.
* Support for 38 disease classes.

## Top Predictions

* Displays the top three most probable predictions.
* Shows confidence scores for each prediction.

## Disease Advisory System

Provides detailed information including:

* Symptoms
* Treatment recommendations
* Preventive measures

## Unknown Plant Detection

* Detects unsupported or unrelated images.
* Prevents unreliable disease predictions for non-crop images.

## Image Preview

* Displays the uploaded leaf image before analysis.
* Allows users to verify the selected image.

## Interactive Web Interface

* Modern and responsive user interface.
* Real-time prediction workflow.
* Easy-to-use navigation and result presentation.

---

# Dataset

The model is trained using the PlantVillage Dataset, a widely used benchmark dataset for plant disease classification.

### Dataset Characteristics

* Total Classes: 38
* Multiple Crop Species
* Healthy and Diseased Categories
* Thousands of labeled leaf images

### Supported Crops

* Apple
* Blueberry
* Cherry
* Corn
* Grape
* Orange
* Peach
* Pepper
* Potato
* Raspberry
* Soybean
* Squash
* Strawberry
* Tomato

---

# Deep Learning Model

## Model Architecture

The system uses MobileNetV2 as the primary deep learning architecture.

### Why MobileNetV2?

* Lightweight architecture
* Efficient inference
* Suitable for deployment
* Strong image classification performance
* Lower computational requirements

### Training Pipeline

1. Dataset Collection
2. Image Preprocessing
3. Data Augmentation
4. Model Training
5. Validation
6. Performance Evaluation
7. Model Export

---

# System Architecture

## Frontend

The frontend is responsible for:

* Image upload
* Image preview
* Displaying prediction results
* Showing advisory information

Technologies Used:

* HTML5
* CSS3
* JavaScript

---

## Backend

The backend handles:

* API requests
* Image processing
* Model inference
* Advisory retrieval

Technologies Used:

* Python
* Flask

---

## Machine Learning Layer

Responsible for:

* Image preprocessing
* Model loading
* Prediction generation
* Confidence calculation

Technologies Used:

* TensorFlow
* Keras
* NumPy

---

# Project Structure

```text
smart-crop-disease-detection/

├── backend/
│   ├── model/
│   │   ├── artifacts/
│   │   │   ├── crop_disease_mobilenetv2.keras
│   │   │   ├── class_names.json
│   │   │   └── disease_info.json
│   │   └── model_loader.py
│   │
│   ├── routes/
│   │   └── prediction_routes.py
│   │
│   ├── services/
│   │   ├── prediction_service.py
│   │   ├── image_service.py
│   │   └── disease_info_service.py
│   │
│   └── app.py
│
├── frontend/
│   ├── pages/
│   ├── css/
│   ├── js/
│   └── assets/
│
├── ml/
│   ├── notebooks/
│   ├── src/
│   └── outputs/
│
└── README.md
```

---

# Workflow

### Step 1

User uploads a crop leaf image.

### Step 2

The image is sent to the Flask backend.

### Step 3

The backend preprocesses the image.

### Step 4

The MobileNetV2 model performs inference.

### Step 5

The system identifies the most probable disease.

### Step 6

Confidence scores are generated.

### Step 7

Disease advisory information is retrieved.

### Step 8

Results are displayed to the user.

---

# Evaluation

The model was evaluated using standard classification metrics.

### Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### Visualization

The project includes:

* Training Accuracy Curves
* Validation Accuracy Curves
* Training Loss Curves
* Validation Loss Curves
* Class Distribution Charts
* Confusion Matrix Heatmap

---

# Installation

## Clone Repository

```bash
git clone <repository-url>
cd smart-crop-disease-detection
```

## Create Virtual Environment

```bash
python -m venv .venv
```

## Activate Environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Backend

```bash
python -m backend.app
```

Backend runs on:

```text
http://127.0.0.1:5000
```

---

# Running the Frontend

Navigate to the project root and run:

```bash
python -m http.server 5500
```

Open:

```text
http://localhost:5500/frontend/pages/index.html
```

---

# API Endpoint

## Predict Disease

### Request

```http
POST /predict
```

### Form Data

```text
image: uploaded_leaf_image
```

### Sample Response

```json
{
  "prediction": {
    "disease": "Apple___Apple_scab",
    "confidence": 98.7
  },
  "top_predictions": [
    {
      "disease": "Apple___Apple_scab",
      "confidence": 98.7
    }
  ],
  "advisory": {
    "symptoms": "...",
    "treatment": "...",
    "prevention": "..."
  }
}
```

---

# Future Enhancements

* Grad-CAM based Explainable AI visualizations
* Real-time camera support
* Mobile application development
* Cloud deployment
* Multi-language support
* Crop recommendation system
* Severity estimation
* Weather-aware disease advisory
* Farmer dashboard and analytics

---

# Conclusion

The Smart Crop Disease Detection and Advisory System demonstrates the practical application of deep learning in agriculture. By combining MobileNetV2-based image classification with an advisory knowledge base and a modern web interface, the system provides an efficient and accessible solution for crop disease identification and management. The project showcases the integration of Artificial Intelligence, Computer Vision, and Web Development to address real-world agricultural challenges.
