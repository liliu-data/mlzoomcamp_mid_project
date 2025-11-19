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

Stroke survivors distribution:
![enter image description here](https://github.com/liliu-data/mlzoomcamp_mid_project/blob/main/EDA/Stroke_Survivor_Distribution_Overview.jpg)

Correlation with stroke heatmap:
![enter image description here](https://github.com/liliu-data/mlzoomcamp_mid_project/blob/main/EDA/correlation_heatmap.png)

As we can see, the feature importance to the stroke is quite low in the numerical features. Mutual information analysis also shows low association among the features. 

    ever_married      0.007264
    work_type         0.006085
    heart_disease     0.005525
    hypertension      0.005202
    smoking_status    0.002798
    residence_type    0.000094
    gender            0.000092
    dtype: float64

# Model Selection - Class Imbalance

Since the data is extremely imbalanced and our goal is to predict a binary target, I chose to use **XGBoost**  to train our model. 

Before feeding the data to the model, we need to solve the extreme imbalance in our dataset. According to ChatGPT and the discussions on Kaggle, several resampling methods were recommended: XGBoost's built-in `use_scale_pos_weight `, **Simple resampling** by duplicating the minority, **SMOTE** (**Synthetic Minority Over-sampling Technique**), and **SMOTE-Tomek**. 

To see how I decided which method and the tuning process, please see [the notebook (mid_project)](https://github.com/liliu-data/mlzoomcamp_mid_project/blob/main/mid_project.ipynb)

# Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/liliu-data/mlzoomcamp_mid_project.git
   cd mlzoomcamp_mid_project

2.  **Setup the environment**
```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate  # Linux/Mac
# OR .venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -r requirements.txt 
```
3. **Train the model**
```bash
python train.py
```

4. **Run the API**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Using Docker**
```
# Build the image
docker build -t stroke-prediction-api .

# Run the container
docker run -p 8000:8000 stroke-prediction-api
```
