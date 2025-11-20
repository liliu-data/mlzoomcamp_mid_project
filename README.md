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

## How to Run the Application

This project provides two ways to run the stroke prediction service: **locally with uv** or **using Docker** for containerized deployment.

### Prerequisites

-   **Option 1 (Local)**: Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
    
-   **Option 2 (Docker)**: [Docker](https://www.docker.com/products/docker-desktop/) installed on your system
    

### Method 1: Running with Docker (Recommended)

Docker provides a consistent, isolated environment that works the same way on any machine.

#### 1. Clone the Repository

```bash
git clone https://github.com/liliu-data/mlzoomcamp_mid_project.git
cd mlzoomcamp_mid_project
```

#### 2. Build the Docker Image

```bash
docker build -t stroke-prediction-app .
```

#### 3. Run the Docker Container

```bash
docker run -p 8000:8000 stroke-prediction-app
```

You should see output similar to:

```text
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [X] using stateless reload
INFO:     Started server process [X]
INFO:     Application startup complete.
```

#### 4. Access the Application

Open your web browser and navigate to:

```text
http://localhost:8000/docs
```

### Method 2: Running Locally with uv

#### 1. Clone and Setup

```bash
git clone https://github.com/liliu-data/mlzoomcamp_mid_project.git
cd mlzoomcamp_mid_project
```

#### 2. Install uv and Dependencies

```bash
pip install uv
uv sync
```

#### 3. Run the Web Application

```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 8000 --reload
```

#### 4. Access the Application
Open your web browser and navigate to:
```text
http://localhost:8000/docs
```

----------

## FastAPI Automatic Documentation

FastAPI provides automatic interactive documentation:

-   **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
    
-   **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
    

The docs pages allow you to test the stroke prediction API directly from your browser!

[https://github.com/liliu-data/mlzoomcamp_mid_project/blob/main/screenshots/webservice.png](https://github.com/liliu-data/mlzoomcamp_mid_project/blob/main/screenshots/webservice.png)

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
    

[See how it would look like on your browser (video)](https://drive.google.com/file/d/1UiN0Eb22zt1v1giO0n15xG60JwALkcRV/view?usp=sharing)

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

-  `mid_project.ipynb` - The notebook with the training process
- `healthcare-dataset-stroke-data.csv` - The dataset used for this project
- `train.py` - Train script that saves the model as `model.bin`
- `predict.py` - Main FastAPI application with the prediction endpoint
    
-   `test.py` - Test script to verify the API is working
    
-   `Dockerfile` - Docker configuration for containerized deployment
    
-   `uv.lock` - Dependency lock file for reproducible installs

-   `EDA/` - Exploratory Data Analysis visualizations
    

----------

## Development

For development, you also have access to:

```bash

# Install development dependencies (like requests)
uv sync --dev
```

----------

## Stopping the Application

-   **Docker**: Press `Ctrl+C` in the terminal or run `docker stop <container_id>`
    
-   **Local**: Press `Ctrl+C` in your terminal
    

----------

## Deployment Ready

The application is ready for production deployment with:

**Using Docker (Recommended for production):**

```bash
docker run -p 8000:8000 stroke-prediction-app
```
**Using uv:**

```bash
uvicorn predict:app --host 0.0.0.0 --port $PORT
```
----------
# Known limitations / next steps 

### Data Quality & Model Performance

-   **Class Imbalance**: The dataset suffers from extreme class imbalance (only ~5% stroke cases), which poses challenges for model training and evaluation metrics interpretation.
    
-   **Limited Features**: The dataset lacks important clinical indicators such as family history, genetic factors, medication usage, and detailed lifestyle information that could improve prediction accuracy.
    
-   **Model Performance**: While XGBoost handles imbalanced data relatively well, the model's predictive power is limited by the available features, with most showing low mutual information with the target variable.
    
-   **Data Quality**: Missing BMI values were imputed with the mean, which may not reflect the true distribution and could introduce bias.

### Technical Limitations

-   **Threshold Sensitivity**: The binary classification is highly sensitive to the probability threshold chosen, and the current threshold may need adjustment for different use cases.
    
-   **Geographic Limitations**: The data appears to be from a specific demographic/geographic population, which may limit generalizability to other populations.
    
-   **Temporal Factors**: The dataset lacks temporal information, making it impossible to account for changes in health metrics over time.

### Deployment Considerations

-   **Computational Requirements**: The current implementation runs on a single server and may not scale efficiently for high-volume prediction requests.
    
-   **Real-time Performance**: For clinical use, the model would require more rigorous validation and potentially faster inference times.

## Next Steps & Future Improvements

### Model Enhancement
    
-   **Advanced Imbalance Handling**: Experiment with more sophisticated techniques like ensemble methods combined with different sampling strategies.
    
-   **Model Interpretability**: Implement SHAP values or LIME to provide explanations for individual predictions, which is crucial for healthcare applications.
    
-   **Hyperparameter Tuning**: Conduct more extensive hyperparameter optimization using techniques like Bayesian optimization.
    

### Data & Validation

-   **External Validation**: Test the model on completely independent datasets from different healthcare systems to assess generalizability.

### Deployment & Scalability

-   **API Enhancements**: Add authentication, rate limiting, and request logging for production use.
    
-   **Container Optimization**: Create a multi-stage Docker build to reduce image size and improve security.
    
-   **Scalable Architecture**: Deploy using Kubernetes or cloud services (AWS ECS, Google Cloud Run) for better scalability and reliability.
    
-   **Monitoring**: Implement performance monitoring, model drift detection, and automated retraining pipelines.
    

### User Experience

-   **Web Interface**: Develop a user-friendly web frontend for non-technical users to input patient data and view results.
    
-   **Batch Processing**: Create endpoints for bulk predictions to handle multiple patient records simultaneously.
    
-   **Confidence Intervals**: Provide prediction confidence intervals to help users assess result reliability.
    

### Clinical Integration

-   **Risk Stratification**: Extend the model to predict different risk levels (low, medium, high) rather than just binary classification.
    
-   **Prevention Recommendations**: Integrate with clinical guidelines to provide personalized prevention recommendations based on prediction results.
    
-   **Regulatory Compliance**: Ensure the application meets healthcare data privacy standards (HIPAA, GDPR) if deployed in clinical settings.

**Any feedback or comments are welcome <3**
**Email:** liting.liu.work@gmail.com
**LinkedIn:** https://www.linkedin.com/in/liting-liu/

