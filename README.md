# Welcome to MLZoomCamp Mid Project: Stroke Prediction (Revised)!

This project aims to build an end-to-end product that predicts stroke occurrences using machine learning techniques. **This revised version includes comprehensive model comparison (regression vs tree-based models) and cloud deployment capabilities.**

## 🆕 What's New in This Revision

- **Model Comparison**: Compares multiple ML models including Logistic Regression (regression) and tree-based models (Random Forest, XGBoost, LightGBM)
- **Cloud Deployment**: Ready-to-deploy configurations for Google Cloud Platform (GCP) and AWS
- **Enhanced API**: Supports predictions from all models with comparison capabilities
- **Model Evaluation**: Comprehensive evaluation metrics and visualization tools

# Data

The dataset used is [`healthcare-dataset-stroke-data.csv`](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data), containing 5,110 entries with the following features:

## Features

-   **Demographic**: `id`, `gender`, `age`
-   **Medical History**: `hypertension`, `heart_disease`
-   **Lifestyle**: `ever_married`, `work_type`, `residence_type`, `smoking_status`
-   **Health Metrics**: `avg_glucose_level`, `bmi`
-   **Target**: `stroke` (binary classification)

# Data Preparation

The data is well-prepared except few missing values in `bmi`.
The data preparation process includes:

-   Standardized column names (lowercase, underscores)
-   Handled missing values in BMI (filled with mean)

Overview distribution of all features:
![distribution](https://github.com/liliu-data/mlzoomcamp_mid_project/blob/main/EDA/Stroke_Survivor_Distribution_Overview.jpg)

Correlation with stroke heatmap:
![correlation](https://github.com/liliu-data/mlzoomcamp_mid_project/blob/main/EDA/correlation_heatmap.png)

As we can see, the feature importance to the stroke is quite low in the numerical features. Mutual information analysis also shows low association among the features.

# Model Comparison: Regression vs Tree-based Models

This project compares multiple machine learning models to find the best performing one:

## Models Compared

### Regression Models
- **Logistic Regression**: Linear model with balanced class weights

### Tree-based Models
- **Random Forest**: Ensemble of decision trees with balanced class weights
- **XGBoost**: Gradient boosting framework optimized for performance
- **LightGBM**: Fast gradient boosting framework with leaf-wise tree growth

## Model Training

Run the training script to compare all models:

```bash
python train.py
```

This will:
1. Load and preprocess the data
2. Train all four models (Logistic Regression, Random Forest, XGBoost, LightGBM)
3. Evaluate each model on test data
4. Compare performance metrics (AUC, F1, Precision, Recall)
5. Save the best performing model to `model.bin`
6. Save all models and comparison results

### Output Files

After training, you'll get:
- `model.bin` - Best performing model (for API use)
- `models_all_YYYYMMDD_HHMMSS.bin` - All trained models
- `model_comparison_YYYYMMDD_HHMMSS.json` - Comparison results in JSON
- `model_comparison_YYYYMMDD_HHMMSS.csv` - Comparison results in CSV

## Model Visualization

Visualize model comparison results:

```bash
python compare_models.py
```

This generates `model_comparison_visualization.png` with:
- AUC Score comparison
- F1 Score comparison
- Precision vs Recall comparison
- Comprehensive metrics comparison

## Class Imbalance Handling

The dataset is extremely imbalanced (~5% stroke cases). We use **SMOTE-Tomek** (Synthetic Minority Over-sampling Technique combined with Tomek links) to handle this imbalance during training.

## How to Run the Application

### Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)

### Method 1: Running Locally

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Train Models (if not already done)

```bash
python train.py
```

#### 3. Run the Web Application

```bash
uvicorn predict:app --host 0.0.0.0 --port 8000 --reload
```

#### 4. Access the Application

Open your web browser and navigate to:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Method 2: Running with Docker

#### 1. Build the Docker Image

```bash
docker build -t stroke-prediction-app .
```

#### 2. Run the Docker Container

```bash
docker run -p 8000:8000 stroke-prediction-app
```

#### 3. Access the Application

Open your web browser and navigate to:
- http://localhost:8000/docs

## API Endpoints

### 1. Root Endpoint

**GET** `/`

Returns API information and available endpoints.

### 2. Health Check

**GET** `/health`

Returns API health status.

### 3. Predict (Best Model)

**POST** `/predict`

Predict stroke probability using the best trained model.

**Request:**
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

**Response:**
```json
{
  "stroke_probability": 0.06694265033464096,
  "stroke": false,
  "model_used": "Logistic Regression"
}
```

### 4. Predict (All Models)

**POST** `/predict/all`

Get predictions from all available models for comparison.

**Response:**
```json
{
  "stroke_probability": 0.06694265033464096,
  "stroke": false,
  "model_used": "Logistic Regression",
  "all_models": {
    "Logistic Regression": {
      "stroke_probability": 0.06694265033464096,
      "stroke": false
    },
    "Random Forest": {
      "stroke_probability": 0.1120424756099942,
      "stroke": false
    },
    "XGBoost": {
      "stroke_probability": 0.07373718172311783,
      "stroke": false
    },
    "LightGBM": {
      "stroke_probability": 0.035953917347134406,
      "stroke": false
    }
  }
}
```

### Testing the API

#### Using the Interactive Documentation

1. Visit http://localhost:8000/docs
2. Find the endpoint you want to test
3. Click "Try it out"
4. Enter patient data and click "Execute"

#### Using the Test Script

```bash
python test.py
```

#### Using curl

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

# Cloud Deployment

This project includes ready-to-use configurations for deploying to cloud platforms.

# AWS Lambda Deployment

This project can be deployed to AWS Lambda for a simple, serverless, and cost-effective solution.

## Prerequisites

- AWS CLI installed and configured
- Docker installed
- AWS account with appropriate permissions

## Quick Deploy

### Option 1: Using Deployment Script (Recommended)

```bash
# Set your AWS region (optional, defaults to us-east-1)
export AWS_REGION=us-east-1

# Make script executable
chmod +x deploy_lambda.sh

# Run deployment
./deploy_lambda.sh
```

### Option 2: Manual Deployment

#### Step 1: Create ECR Repository

```bash
# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=us-east-1

# Create ECR repository
aws ecr create-repository \
    --repository-name stroke-prediction \
    --region $AWS_REGION
```

#### Step 2: Build and Push Docker Image

```bash
# Login to ECR
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin \
    $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build image for Lambda
docker build -f Dockerfile.lambda -t stroke-prediction .

# Tag and push
docker tag stroke-prediction:latest \
    $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/stroke-prediction:latest

docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/stroke-prediction:latest
```

#### Step 3: Create IAM Role for Lambda

```bash
# Create trust policy
cat > lambda-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create IAM role
aws iam create-role \
    --role-name stroke-prediction-lambda-role \
    --assume-role-policy-document file://lambda-trust-policy.json

# Attach basic execution policy
aws iam attach-role-policy \
    --role-name stroke-prediction-lambda-role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# Wait for role to be ready
sleep 10
```

#### Step 4: Create Lambda Function

```bash
# Create Lambda function
aws lambda create-function \
    --function-name stroke-prediction \
    --package-type Image \
    --code ImageUri=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/stroke-prediction:latest \
    --role arn:aws:iam::$AWS_ACCOUNT_ID:role/stroke-prediction-lambda-role \
    --timeout 30 \
    --memory-size 512 \
    --region $AWS_REGION
```

#### Step 5: Create Function URL (Public API)

```bash
# Create function URL
aws lambda create-function-url-config \
    --function-name stroke-prediction \
    --auth-type NONE \
    --region $AWS_REGION

# Add permissions for public access
aws lambda add-permission \
    --function-name stroke-prediction \
    --statement-id FunctionURLAllowPublicAccess \
    --action lambda:InvokeFunctionUrl \
    --principal "*" \
    --function-url-auth-type NONE \
    --region $AWS_REGION
```

#### Step 6: Get Function URL

```bash
# Get the function URL
aws lambda get-function-url-config \
    --function-name stroke-prediction \
    --region $AWS_REGION \
    --query 'FunctionUrl' \
    --output text
```

Your API will be available at: `https://[unique-id].lambda-url.[region].on.aws/`

## Testing Your Lambda Deployment

### Using curl

```bash
# Replace with your actual function URL
FUNCTION_URL="https://your-unique-id.lambda-url.us-east-1.on.aws"

curl -X POST "$FUNCTION_URL/predict" \
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

### Using Python

```python
import requests
import json

FUNCTION_URL = "https://your-unique-id.lambda-url.us-east-1.on.aws"

patient_data = {
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

response = requests.post(f"{FUNCTION_URL}/predict", json=patient_data)
print(json.dumps(response.json(), indent=2))
```




## Project Structure

```
mlzoomcamp_mid_project/
├── train.py                          # Model training and comparison script
├── predict.py                        # FastAPI application
├── compare_models.py                 # Model comparison visualization
├── test.py                           # API test script
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Docker configuration
├── healthcare-dataset-stroke-data.csv  # Dataset
├── model.bin                         # Best trained model (generated)
├── EDA/                              # Exploratory Data Analysis
└── README.md                         # This file
```

## Development

### Install Development Dependencies

```bash
pip install -r requirements.txt
```

### Run Tests

```bash
python test.py
```

### Model Training Workflow

1. **Train Models**: `python train.py`
2. **Visualize Results**: `python compare_models.py`
3. **Start API**: `uvicorn predict:app --host 0.0.0.0 --port 8000`
4. **Test API**: `python test.py`

## Performance Metrics

Models are evaluated using:
- **AUC (Area Under ROC Curve)**: Overall model performance
- **F1 Score**: Balance between precision and recall
- **Precision**: Accuracy of positive predictions
- **Recall**: Ability to find all positive cases

The best model is selected based on the highest AUC score.

## Known Limitations

### Data Quality & Model Performance

- **Class Imbalance**: The dataset suffers from extreme class imbalance (only ~5% stroke cases)
- **Limited Features**: The dataset lacks important clinical indicators
- **Data Quality**: Missing BMI values were imputed with the mean

### Technical Limitations

- **Threshold Sensitivity**: Binary classification is sensitive to probability threshold
- **Geographic Limitations**: Data may be from a specific demographic/geographic population
- **Temporal Factors**: Dataset lacks temporal information

## Future Improvements

### Model Enhancement
- **Hyperparameter Tuning**: More extensive optimization using Bayesian methods
- **Model Interpretability**: Implement SHAP values or LIME
- **Ensemble Methods**: Combine predictions from multiple models

### Deployment & Scalability
- **API Enhancements**: Add authentication, rate limiting, request logging
- **Monitoring**: Implement performance monitoring and model drift detection
- **Auto-scaling**: Configure automatic scaling based on load

### User Experience
- **Web Interface**: User-friendly frontend for non-technical users
- **Batch Processing**: Endpoints for bulk predictions
- **Confidence Intervals**: Provide prediction confidence intervals

## Contributing

Any feedback or comments are welcome!

**Email:** liting.liu.work@gmail.com  
**LinkedIn:** https://www.linkedin.com/in/liting-liu/

## License

This project is for educational purposes as part of the MLZoomCamp course.
