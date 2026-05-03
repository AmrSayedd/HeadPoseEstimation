# Head Pose Estimation: Roll, Pitch, and Yaw Prediction

This project implements a machine learning pipeline for detecting facial landmarks and estimating head pose (Roll, Pitch, and Yaw) from images and video streams using **MediaPipe Face Mesh** and **Support Vector Regression (SVR)**.

---

## Notebook Descriptions

### 1. Initial Viewing and Filtering
**File:** `Initial_viewing_and_filtering_code.ipynb`

This notebook serves as the primary data exploration and cleaning stage. It loads the AFLW dataset, which contains images and corresponding pose parameters in `.mat` files.

**Fancy Data Cleaning**  
This stage is essentially a sophisticated filtering process. Because MediaPipe occasionally fails to detect landmarks on certain images—often due to extreme angles or lighting—this notebook identifies and separates "corrupt" or unusable images. This ensures that only high-quality, detectable data is used for training.

---

### 2. Machine Learning Training Pipeline
**File:** `Task Using ML.ipynb`

This notebook contains the core training logic for the head pose models.

**Preprocessing**  
Extracts specific 3D coordinates from the MediaPipe Face Mesh, including:
- Eye centers  
- Nose tip  
- Chin  
- Temporal regions  

**Model Selection**  
Evaluates multiple regression algorithms:
- Random Forest  
- K-Nearest Neighbors  
- Support Vector Regression (SVR)  

---

### 3. Inference
**File:** `Inference.ipynb`

This notebook is designed for practical application.

**Real-time Processing**  
- Loads pre-trained SVR models and scalers from Google Drive  
- Processes video frame-by-frame  
- Extracts landmarks and calculates features  
- Overlays predicted Roll, Pitch, and Yaw values directly onto the video  

---

## Feature Engineering Breakdown

The system does not feed raw landmark coordinates directly into the models. Instead, it uses feature engineering based on spatial differences between key facial landmarks.

### Roll Features (Z-axis rotation)
- `chin_minus_forehead_x`: Horizontal distance between chin tip and forehead center  
- `chin_minus_landmark366_x / landmark137_x`: Distance between chin and temporal landmarks  

### Pitch Features (X-axis rotation)
- `nose_minus_chin_y`: Vertical distance between nose and chin  
- `chin_minus_forehead_z`: Depth difference between top and bottom of the face  
- `nose_minus_landmark137_y / landmark366_y`: Vertical distances between nose and sides of the face  

### Yaw Features (Y-axis rotation)
- `nose_minus_mouth_left_x / mouth_right_x`: Horizontal distance from nose to mouth corners  
- `nose_minus_landmark137_x / landmark366_x`: Distance from nose to sides of the face  

---

## Requirements

To run these notebooks, install:
mediapipe
opencv-python
numpy
pandas
scikit-learn
joblib
scipy
