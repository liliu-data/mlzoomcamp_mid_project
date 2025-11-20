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

# EDA

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


# Installation & Running the Web App

This project uses [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for fast dependency management and `FastAPI` for the web framework. Follow these steps to set up and run the stroke prediction application.

### Prerequisites

-   Python 3.11+ (as specified in your project)
    
-   [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
    

### 1. Install with uv (Recommended)

```bash

# Clone the repository
git clone https://github.com/liliu-data/mlzoomcamp_mid_project.git
cd mlzoomcamp_mid_project
```
#### Install uv
```bash
pip install uv
```
#### Initiate virtual environment
```bash
uv init
``` 

#### Install dependencies with uv (creates a virtual environment automatically)
```bash
uv add scikit-learn fastapi uvicorn
```

### 2. Run the Web Application

Start the FastAPI server with:

```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 8000 --reload
```

You should see output similar to:

```text
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [X] using stateless reload
INFO:     Started server process [X]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 3. Access the Application

Open your web browser and navigate to:

```text
http://localhost:8000/doc
```

#### FastAPI Automatic Documentation

FastAPI provides automatic interactive documentation:

-   **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
    
-   **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
    

The docs pages allow you to test the stroke prediction API directly from your browser!

![webservice](https://github.com/liliu-data/mlzoomcamp_mid_project/blob/main/screenshots/webservice.png)

----------

## Using the Stroke Prediction API

### API Endpoint

**POST**  `/predict`

### Request Format

Send a JSON object with the following patient information:

```json
{
    "gender": "male",
    "age": 33.00,
    "hypertension": 1,
    "heart_disease": 0,
    "ever_married": "yes",
    "work_type": "private",
    "residence_type": "rural",
    "avg_glucose_level": 58.12,
    "bmi": 32.5,
    "smoking_status": "never_smoked"
}
```

### Response Format

The API returns a JSON response with stroke prediction:

```json
{
  "stroke_probability": 0.07373718172311783,
  "stroke": false
}
```


### Testing the API

#### Method 1: Using the Interactive Documentation

1.  Visit [http://localhost:8000/docs](http://localhost:8000/docs)
    
2.  Find the `/predict` endpoint
    
3.  Click "Try it out"
    
4.  Enter patient data and click "Execute"

[See how it would look like on your brower (video)](https://drive.google.com/file/d/1UiN0Eb22zt1v1giO0n15xG60JwALkcRV/view?usp=sharing)

#### Method 2: Using the provided test script

```bash
# Make sure the server is running first, then:
python test.py
```

#### Method 3: Using curl

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "male",
    "age": 33.00,
    "hypertension": 1,
    "heart_disease": 0,
    "ever_married": "yes",
    "work_type": "private",
    "residence_type": "rural",
    "avg_glucose_level": 58.12,
    "bmi": 32.5,
    "smoking_status": "never_smoked"
  }'
```

#### Method 4: Using Python requests

```python
import requests

url = 'http://localhost:8000/predict'

patient = {
    'gender': 'male',
    'age': 33.00,
    'hypertension': 1,
    'heart_disease': 0,
    'ever_married': 'yes',
    'work_type': 'private',
    'residence_type': 'rural',
    'avg_glucose_level': 58.12,
    'bmi': 32.5,
    'smoking_status': 'never_smoked',
}

response = requests.post(url, json=patient)
predictions = response.json()

if predictions['stroke']:
    print(f"The patient is likely to have a stroke with a probability of {predictions['stroke_probability']}")
else:
    print(f"The patient is unlikely to have a stroke with a probability of {predictions['stroke_probability']}")

```
----------
## Project Structure

-   `predict.py` - Main FastAPI application with the prediction endpoint
    
-   `test.py` - Test script to verify the API is working
    
-   `uv.lock` - Dependency lock file for reproducible installs
    
-   `model/` - Directory containing your trained machine learning model
    
-   `preprocessor/` - Directory containing data preprocessing components
    

----------

## Development

For development, you also have access to:

```bash
# Install development dependencies (like requests)
uv sync --dev
```
----------

## Stopping the Application

To stop the server, press `Ctrl+C` in your terminal.

----------

## Deployment Ready

The application is ready for production deployment with:

```bash

uvicorn predict:app --host 0.0.0.0 --port $PORT
