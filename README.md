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

This project implements a deep learning model for traffic sign recognition using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. It showcases the application of convolutional neural networks (CNNs) in computer vision tasks, particularly in the domain of intelligent transportation systems. The project demonstrates the end-to-end process of developing, evaluating, and deploying a machine learning model for real-world applications.

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
| ... | ... |
| 42 | End of no passing by vehicles over 3.5 metric tons |

## Model Architecture

We implemented a modified LeNet architecture, achieving a 98.7% accuracy on the test set. Key modifications and their rationale include:

1. Increased number of filters in convolutional layers (32, 64, 128):
   - Allows the model to capture a wider range of features at different scales
   - Enhances the model's ability to distinguish between similar sign types

2. Additional convolutional layers (total of 3):
   - Increases the depth of the network, allowing for more complex feature hierarchies
   - Improves the model's capacity to learn intricate patterns in traffic signs

3. Dropout layers (rate: 0.25) after each convolutional block:
   - Reduces overfitting by preventing co-adaptation of features
   - Improves model generalization to unseen data

4. Increased number of units in dense layers (512, 256):
   - Provides more capacity for the model to learn high-level representations
   - Allows for better separation of classes in the feature space

These architectural choices were informed by the complexity of the GTSRB dataset and the need to balance model capacity with computational efficiency.

## Data Analysis Insights

### Class Imbalance

Our analysis revealed significant class imbalance in the dataset:

- Classes with >2000 samples (e.g., speed limits, priority signs): 
  - Exhibited exceptional performance with accuracy >99%
  - Suggests that these common signs are well-represented and easily distinguishable

- Classes with 500-2000 samples (e.g., warning signs, regulatory signs):
  - Showed good performance with accuracy between 95-98%
  - Indicates that the model can effectively learn from moderately represented classes

- Classes with <300 samples (e.g., rare or complex signs):
  - Demonstrated lower prediction confidence and accuracy (85-90%)
  - Highlights the challenge of learning from limited examples and the need for targeted data augmentation

### Distribution Analysis

We observed a skewed distribution in the dataset:

- Over-representation:
  - Speed limit signs (~ 40% of the dataset)
  - Regulatory signs (e.g., no entry, priority road)
  - Reflects the frequency of these signs in real-world scenarios but may lead to bias

- Under-representation:
  - Warning signs for specific hazards (e.g., slippery road, wild animals)
  - Uncommon signs (e.g., end of prohibitions)
  - Poses a challenge for the model to generalize to these less frequent but critical signs

This distribution analysis informs potential strategies for dataset balancing and targeted data collection to improve model robustness.

### Performance Correlation

We identified a strong correlation between sample size and prediction confidence:

- Pearson correlation coefficient: 0.78 (p < 0.001)
- Classes with <300 samples showed an average confidence score of 0.85, compared to 0.97 for well-represented classes
- Suggests that increasing samples for underrepresented classes could significantly improve model performance

## Usage

### Jupyter Notebook

Run `01-traffic-sign-recognition.ipynb` in a Jupyter environment for data analysis, model development, and evaluation.

### Streamlit App

Launch the web application:

```
streamlit run 02-streamlit-app-glossary.py
```

### Webcam Recognition

Start real-time recognition:

```
python 03-traffic_sign_webcam.py
```

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

1. Address class imbalance:
   - Implement advanced data augmentation techniques (e.g., GANs for synthetic sample generation)
   - Explore adaptive sampling methods during training

2. Architectural improvements:
   - Experiment with state-of-the-art architectures (e.g., EfficientNetV2, Vision Transformers)
   - Investigate transfer learning from large-scale datasets (e.g., ImageNet) for improved feature extraction

3. Robustness enhancements:
   - Implement adversarial training to improve model resilience to perturbed inputs
   - Develop multi-condition models to handle varying weather and lighting conditions

4. Deployment optimizations:
   - Explore model quantization and pruning for efficient edge deployment
   - Implement TensorRT or ONNX Runtime for accelerated inference

5. Extended functionality:
   - Integrate object detection for sign localization in complex scenes
   - Develop a multi-lingual classification system for international applicability

## Conclusion

This project demonstrates the efficacy of deep learning in traffic sign recognition, achieving 98.7% accuracy on the GTSRB dataset. The analysis reveals the impact of class imbalance and data distribution on model performance, highlighting areas for future improvement. The provided web and webcam applications showcase the practical deployment of the model, bridging the gap between research and real-world application in intelligent transportation systems.

