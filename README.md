# Forest Fire Prediction - 2024

## Project Description

This project aims to predict forest fires using supervised machine learning algorithms such as **Random Forest**, **SVM**, **KNN**, and **MLP**. The dataset contains environmental factors like **NDVI (Normalized Difference Vegetation Index)**, **LST (Land Surface Temperature)**, and **BURNED_AREA** to train the models. Hyperparameter optimization using **GridSearchCV** and **cross-validation** is performed to identify the best-performing model.

---

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Machine Learning Models](#machine-learning-models)
- [Model Evaluation](#model-evaluation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)

---

## Introduction

This project focuses on predicting forest fires based on the following attributes:

- **NDVI**: A measure of vegetation health and density.
- **LST**: The land surface temperature.
- **BURNED_AREA**: The target variable representing the area burned during a fire.

The system uses supervised machine learning algorithms to train models that predict the **BURNED_AREA** based on the given features. Transfer learning and hyperparameter tuning are used to improve performance.

---

## Dataset

The dataset used in this project is **WildFires_DataSet_a_interpoler.csv**, which contains data on vegetation, temperature, and burned areas. It includes the following columns:

- **NDVI (Normalized Difference Vegetation Index)**: Indicates the vegetation density and health of the land.
- **LST (Land Surface Temperature)**: The temperature of the land surface.
- **BURNED_AREA**: The target variable representing the area burned during a fire.

---

## Data Preprocessing

The preprocessing steps are as follows:

1. **Handling Missing Values**: Missing values in the dataset are handled using **interpolation techniques**, including:
    - **Linear Interpolation**
    - **Spline Interpolation**
    - **Cubic Interpolation**

2. **Feature Scaling**: To ensure all features are on the same scale, **standard scaling** is applied to **NDVI** and **LST**.

3. **Train-Test Split**: The dataset is divided into training and testing sets (typically 80-20% or 70-30%).

---

## Machine Learning Models

The project uses several machine learning algorithms to predict **BURNED_AREA**:

1. **Random Forest**
2. **Support Vector Machine (SVM)**
3. **K-Nearest Neighbors (KNN)**
4. **Multilayer Perceptron (MLP)**

### Hyperparameter Optimization:
- **GridSearchCV** is used to tune the hyperparameters of the models.
- **Cross-validation** is performed to prevent overfitting and to estimate model performance.

---

## Model Evaluation

Models are evaluated based on several metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

Additionally, performance is visualized using **Matplotlib** and **Seaborn** for easy comparison of different models.

---

## Usage

To run the project and train the model, you can use the provided Jupyter notebooks. These notebooks contain the code to train, evaluate, and visualize the machine learning models.

1. **Exploratory Data Analysis (EDA) Notebook**:
   This notebook is used for initial data exploration and preprocessing. It also visualizes the data to understand trends and patterns.
   
   Open the **EDA notebook** by running:
   ```bash
   jupyter notebook Mini_projet_EXAM.ipynb
