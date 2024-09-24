# German Traffic Sign Recognition System

## Table of Contents
- [Overview](#overview)
- [Files](#files)
- [Traffic Sign Classes](#traffic-sign-classes)
- [Model Architecture](#model-architecture)
- [Data Analysis Insights](#data-analysis-insights)
  - [Class Imbalance](#class-imbalance)
  - [Distribution Analysis](#distribution-analysis)
  - [Performance Correlation](#performance-correlation)
- [Usage](#usage)
  - [Jupyter Notebook](#jupyter-notebook)
  - [Streamlit App](#streamlit-app)
  - [Webcam Recognition](#webcam-recognition)
- [Requirements](#requirements)
- [Future Work](#future-work)
- [Conclusion](#conclusion)

## Overview

This project implements a deep learning model for traffic sign recognition using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. It includes a Jupyter notebook for model development and evaluation, a Streamlit web application for interactive predictions, and a real-time webcam-based recognition script.

## Files

1. `01-traffic-sign-recognition.ipynb`: Jupyter notebook containing data analysis, model development, and evaluation.
2. `02-streamlit-app-glossary.py`: Streamlit web application for traffic sign classification.
3. `03-traffic_sign_webcam.py`: Script for real-time traffic sign recognition using a webcam.

## Traffic Sign Classes

The model classifies 43 different types of traffic signs. Here's a sample of the class reference table:

| Class ID | Sign Name |
|----------|-----------|
| 0 | Speed limit (20km/h) |
| 1 | Speed limit (30km/h) |
| 2 | Speed limit (50km/h) |
| 3 | Speed limit (60km/h) |
| 4 | Speed limit (70km/h) |
| 5 | Speed limit (80km/h) |
| 6 | End of speed limit (80km/h) |
| 7 | Speed limit (100km/h) |
| 8 | Speed limit (120km/h) |
| 9 | No passing |
| 10 | No passing for vehicles over 3.5 metric tons |
| 11 | Right-of-way at the next intersection |
| 12 | Priority road |
| 13 | Yield |
| 14 | Stop |
| 15 | No vehicles |
| 16 | Vehicles over 3.5 metric tons prohibited |
| 17 | No entry |
| 18 | General caution |
| 19 | Dangerous curve to the left |
| 20 | Dangerous curve to the right |
| 21 | Double curve |
| 22 | Bumpy road |
| 23 | Slippery road |
| 24 | Road narrows on the right |
| 25 | Road work |
| 26 | Traffic signals |
| 27 | Pedestrians |
| 28 | Children crossing |
| 29 | Bicycles crossing |
| 30 | Beware of ice/snow |
| 31 | Wild animals crossing |
| 32 | End of all speed and passing limits |
| 33 | Turn right ahead |
| 34 | Turn left ahead |
| 35 | Ahead only |
| 36 | Go straight or right |
| 37 | Go straight or left |
| 38 | Keep right |
| 39 | Keep left |
| 40 | Roundabout mandatory |
| 41 | End of no passing |
| 42 | End of no passing by vehicles over 3.5 metric tons |

## Model Architecture

We used a modified LeNet architecture, achieving a 98.7% accuracy on the test set. Key modifications include:

- Increased number of filters in convolutional layers
- Additional convolutional layers
- Dropout for regularization
- Increased number of units in dense layers

These changes allowed the model to capture more complex features and reduce overfitting.

## Data Analysis Insights

### Class Imbalance

- Classes with >2000 samples: Exceptional performance
- Classes with 500-2000 samples: Good performance
- Classes with <300 samples: Lower prediction confidence

### Distribution Analysis

- Skewed distribution towards certain sign types
- Over-representation of speed limit and regulatory signs
- Under-representation of warning and uncommon signs

### Performance Correlation

- Strong correlation between sample size and prediction confidence
- Classes with <300 samples showed lower average confidence scores

## Usage

### Jupyter Notebook

Run `01-traffic-sign-recognition.ipynb` in a Jupyter environment for data analysis, model development, and evaluation.

### Streamlit App

Launch the web application:


streamlit run 02-streamlit-app-glossary.py

### Webcam Recognition

Start real-time recognition:

python 03-traffic_sign_webcam.py

## Requirements

* Python 3.7+
* TensorFlow 2.x
* OpenCV
* Streamlit
* Numpy
* Pandas
* Matplotlib
* Scikit-learn

## Future Work

* Address class imbalance through advanced data augmentation techniques
* Experiment with more complex architectures (ResNet, EfficientNet)
* Implement object detection for sign localization in complex images

## Conclusion

This project demonstrates the effectiveness of deep learning in traffic sign recognition, achieving high accuracy despite dataset challenges. The web and webcam applications showcase practical, user-friendly implementations of the model.





