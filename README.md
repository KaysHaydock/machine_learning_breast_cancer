# Breast Cancer Prediction using Machine Learning

## Overview

This project focuses on predicting the diagnosis (benign or malignant) of breast cancer based on cellular features using machine learning techniques. We use the **Breast Cancer Wisconsin (Diagnostic) Dataset** to train and evaluate various machine learning models, including **Logistic Regression** and **Support Vector Machines (SVM)**. Additionally, clustering techniques like **Agglomerative Clustering** and **K-Means** are used to explore unsupervised learning and assess natural groupings within the data. The goal of the project is to provide accurate predictions and insights to assist in breast cancer diagnosis.

## Author
Kayleigh Haydock: [Github](https://github.com/KaysHaydock)

## Research Question

The main question we aim to address is:
> **Can we predict the diagnosis (benign or malignant) of breast cancer based on cellular features such as radius, texture, perimeter, area, and others?**

## Dataset

The dataset used in this project is the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which is widely used for machine learning tasks. The dataset contains 569 instances of breast cancer samples, with 30 features each, representing various cell characteristics such as:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Symmetry
- Fractal Dimension

Each sample is labeled as either:
- **Malignant (1)**: Cancerous
- **Benign (0)**: Non-cancerous

### Dataset Source

The dataset can be accessed from [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data).

## Machine Learning Techniques Used

In this project, two different machine learning techniques are employed for classification:

### 1. Logistic Regression
- A simple, yet effective model used for binary classification problems. **Logistic Regression** is used here to predict whether a tumor is benign or malignant based on the provided features.

### 2. Support Vector Machine (SVM)
- SVM is used with both **linear** and **RBF (Radial Basis Function)** kernels to classify the data into benign and malignant categories. Hyperparameters like **C** and **gamma** are tuned using **GridSearchCV** and **RandomizedSearchCV** for better model performance.

### 3. Clustering
- I also used the Unsupervised Learning method of **Clustering** for this dataset to see if any features naturally grouped together, making use of K-Means and Agglomerative Clustering. Clustering helps uncover hidden patterns in data that could aid in understanding the characteristics of each class.


## Installation and Setup

To set up the project locally, you can follow these steps:

### Prerequisites

- Python 3.x
- Required Python Libraries: `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `numpy`