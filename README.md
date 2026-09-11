# Ames Housing Price Prediction

A machine learning project for predicting residential house prices using the Ames Housing dataset from Kaggle.

The project focuses on building and comparing regression models, applying feature engineering and preprocessing, and combining XGBoost and CatBoost predictions through an ensemble approach.

## Project Overview

The goal of this project is to predict `SalePrice` for residential properties based on features describing the house, its location, quality, size, age, garage, basement, and other characteristics.

The target variable is highly skewed, so `SalePrice` is transformed using `log1p` during training.

# Dataset

This project uses the Ames Housing dataset from Kaggle's House Prices: Advanced Regression Techniques competition.

The original dataset is not included in this repository.

To run the notebook, download the dataset from Kaggle and place the following files in the project directory:

- `train.csv`
- `test.csv`

The dataset is available through the official Kaggle competition page:

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

## Approach

### 1. Data Exploration

- Examined the structure and distributions of the dataset
- Analyzed missing values
- Inspected the target distribution
- Compared the original and log-transformed target distributions

### 2. Data Preprocessing

- Separated numerical and categorical features
- Handled missing numerical values using training-set medians
- Handled missing categorical values using a dedicated `Missing` category
- Applied one-hot encoding to categorical features for the XGBoost pipeline

### 3. Feature Engineering

Additional features were created to capture more useful information about each property, including:

- Total square footage
- Total bathrooms
- Total porch area
- Total finished square footage
- House age
- Remodeling age
- Garage age
- Quality × living area interaction
- Quality × total square footage interaction

### 4. Outlier Handling

Two extreme observations with unusually large `GrLivArea` and relatively low sale prices were identified and removed before training.

## Models

Several approaches were explored during development, including:

- Random Forest
- XGBoost
- CatBoost
- Neural Network
- XGBoost + CatBoost ensemble

The final approach combines XGBoost and CatBoost predictions using out-of-fold predictions to determine the ensemble weight.

### Final Ensemble

The final ensemble combines:

- XGBoost
- CatBoost

with the weight selected using 5-fold out-of-fold validation.

## Validation Results

| Model | OOF RMSE |
|---|---:|
| XGBoost | 0.11640 |
| CatBoost | 0.11457 |
| Ensemble | 0.11314 |

The models were evaluated using 5-fold cross-validation on the log-transformed target.

## Kaggle Result

Final Kaggle submission:

**Score: 0.12431**

The final submission was generated as:

`submission_final.csv`

