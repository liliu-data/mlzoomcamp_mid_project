# Welcome to MLZoomCamp Mid Project: Stroke Prediction!

This project aims to build an end-to-end product that predicts stroke occurrences using machine learning techniques. The dataset contains various health and demographic factors, and the goal is to build a predictive model that can identify individuals at risk of stroke.

# Data

The dataset used is [`healthcare-dataset-stroke-data.csv`](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data), containing 5,110 entries with the following features:

## Features

-   **Demographic**: `id`, `gender`, `age`
    
-   **Medical History**: `hypertension`, `heart_disease`
    
-   **Lifestyle**: `ever_married`, `work_type`, `residence_type`, `smoking_status`
    
-   **Health Metrics**: `avg_glucose_level`, `bmi`
    
-   **Target**: `stroke` (binary classification)

# ETA

The data is well-prepared except few missing values in `bmi`.
The data preparation process is shown below: 

-   Standardized column names (lowercase, underscores)
    
-   Handled missing values in BMI (filled with mean)
 
 Overview distribution of all features:
 ![ditribution](https://github.com/liliu-data/mlzoomcamp_mid_project/blob/main/EDA/Stroke_Survivor_Distribution_Overview.jpg)

