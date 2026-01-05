# Welcome to MLZoomCamp Mid Project: Stroke Prediction (Revised)!

This project aims to build an end-to-end product that predicts stroke occurrences using machine learning techniques. **This revised version includes comprehensive model comparison (regression vs tree-based models) and cloud deployment capabilities.**

## 🆕 What's New in This Revision

- **Model Comparison**: Compares multiple ML models including Logistic Regression (regression) and tree-based models (Random Forest, XGBoost, LightGBM)
- **Cloud Deployment**: Ready-to-deploy configurations for Google Cloud Platform (GCP) and AWS
- **Enhanced API**: Supports predictions from all models with comparison capabilities
- **Model Evaluation**: Comprehensive evaluation metrics and visualization tools
# 🧠💙 Stroke Prediction ML Project

> *Predicting strokes with machine learning, one heartbeat at a time* ✨

Welcome to my MLZoomCamp midterm project! This adorable little ML model helps predict stroke occurrences using health and demographic data. Let's keep people healthy together! 🏥💪

---

## 📊 The Data Story

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
I'm working with the [`healthcare-dataset-stroke-data.csv`](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data) from Kaggle, which contains 5,110 patient records. Here's what we're looking at:

### ✨ Features at a Glance

- 👤 **Who You Are**: `gender`, `age`, `id`
- 🏥 **Health History**: `hypertension`, `heart_disease`
- 🏡 **Life Choices**: `ever_married`, `work_type`, `residence_type`, `smoking_status`
- 📈 **Health Stats**: `avg_glucose_level`, `bmi`
- 🎯 **The Goal**: `stroke` (yes or no?)

---

## 🔍 Exploring the Data

The dataset is pretty clean! Just a few missing BMI values that I filled with the mean. Here's what I discovered:

**Distribution Overview:**
![distribution](https://github.com/liliu-data/mlzoomcamp_mid_project/blob/main/EDA/high_view.jpg)

**Stroke Survivors:**
![survivors](https://github.com/liliu-data/mlzoomcamp_mid_project/blob/main/EDA/Stroke_Survivor_Distribution_Overview.jpg)

**Correlation Heatmap:**
![heatmap](https://github.com/liliu-data/mlzoomcamp_mid_project/blob/main/EDA/correlation_heatmap.png)

### 🤔 What I Learned

The numerical features don't correlate super strongly with strokes. Here's the mutual information breakdown:

```
ever_married      0.007264  💍
work_type         0.006085  💼
heart_disease     0.005525  💔
hypertension      0.005202  🩺
smoking_status    0.002798  🚬
residence_type    0.000094  🏘️
gender            0.000092  ⚧️
```

---

## 🎯 Model Magic: Handling Class Imbalance

Since stroke cases are rare (only ~5% of the data! 😮), I needed a special approach. After consulting ChatGPT and diving into Kaggle discussions, I tested several resampling methods:

- ⚖️ XGBoost's built-in `scale_pos_weight`
- 🔁 Simple minority class duplication
- 🧪 **SMOTE** (Synthetic Minority Over-sampling)
- 🎨 **SMOTE-Tomek** combo

Curious about my decision process? Check out [the full notebook](https://github.com/liliu-data/mlzoomcamp_mid_project/blob/main/mid_project.ipynb)! 📓

---

## 🚀 Let's Get This Running!

You have two super easy options:

### 🐳 Option 1: Docker (Recommended!)

Perfect for consistency across any machine:

```bash
# 1. Clone me!
git clone https://github.com/liliu-data/mlzoomcamp_mid_project.git
cd mlzoomcamp_mid_project

# 2. Start Docker (if using colima)
colima start

# 3. Build the image
docker build -t stroke-prediction-app .

# 4. Run it!
docker run -p 8000:8000 stroke-prediction-app
```

### 🐍 Option 2: Local with UV

For the Python purists:

```bash
# 1. Clone me!
git clone https://github.com/liliu-data/mlzoomcamp_mid_project.git
cd mlzoomcamp_mid_project

# 2. Install uv and sync
pip install uv
uv sync

# 3. Launch!
uv run uvicorn predict:app --host 0.0.0.0 --port 8000 --reload
```

### 🎉 Access Your App

Navigate to: **http://localhost:8000/docs**

You'll see beautiful auto-generated documentation! ✨

![API Screenshot](https://github.com/liliu-data/mlzoomcamp_mid_project/blob/main/screenshots/webservice.png)

---

## 💻 Using the API

### 📮 Endpoint: `POST /predict`

**Send patient data like this:**

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
    "stroke_probability": 0.0737,
    "stroke": false,
    "model_used": "XGBoost"
}
```

### 4. Predict (All Models)

**POST** `/predict/all`

Get predictions from all available models for comparison.

**Response:**
```json
{
    "stroke_probability": 0.0737,
    "stroke": false,
    "model_used": "XGBoost",
    "all_models": {
        "Logistic Regression": {
            "stroke_probability": 0.0650,
            "stroke": false
        },
        "Random Forest": {
            "stroke_probability": 0.0710,
            "stroke": false
        },
        "XGBoost": {
            "stroke_probability": 0.0737,
            "stroke": false
        },
        "LightGBM": {
            "stroke_probability": 0.0720,
            "stroke": false
        }
    }
**Get predictions like this:**

```json
{
  "stroke_probability": 0.0737,
  "stroke": false
}
```
**Try it out on people you know and tell them if they are at risk of stroke (oop!✨)**

### 🧪 Four Ways to Test

#### Using the Interactive Documentation

1. Visit http://localhost:8000/docs
2. Find the endpoint you want to test
3. Click "Try it out"
4. Enter patient data and click "Execute"

#### Using the Test Script
**1️⃣ Interactive Docs** (Easiest!)
- Visit http://localhost:8000/docs
- Click "Try it out" on `/predict`
- Enter data & hit "Execute"

🎥 **Watch it in action:**

[![API Demo Video](https://img.youtube.com/vi/Ozj6kUKUoJI/0.jpg)](https://youtu.be/Ozj6kUKUoJI)

**2️⃣ Test Script**
```bash
python test.py
```

#### Using curl

**3️⃣ cURL**
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

## Google Cloud Platform (GCP) Deployment

### Prerequisites

- Google Cloud SDK (`gcloud`) installed
- GCP project created
- Docker installed

### Quick Deploy to Cloud Run

#### Option 1: Using Deployment Script

```bash
# Set your GCP project ID
export GCP_PROJECT_ID=your-project-id
export GCP_REGION=us-central1

# Make script executable
chmod +x deploy_gcp.sh

# Run deployment
./deploy_gcp.sh
```

#### Option 2: Manual Deployment

```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Build and push image
docker build -t gcr.io/YOUR_PROJECT_ID/stroke-prediction .
docker push gcr.io/YOUR_PROJECT_ID/stroke-prediction

# Deploy to Cloud Run
gcloud run deploy stroke-prediction \
    --image gcr.io/YOUR_PROJECT_ID/stroke-prediction \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1
```

### Using Cloud Build (CI/CD)

```bash
# Submit build
gcloud builds submit --config cloudbuild.yaml
```

This will automatically build, push, and deploy your application.

### Configuration Files

- `app.yaml` - Cloud Run service configuration
- `cloudbuild.yaml` - Cloud Build CI/CD configuration
- `.gcloudignore` - Files to exclude from deployment

## AWS Deployment

### Prerequisites

- AWS CLI installed and configured
- Docker installed
- AWS account with appropriate permissions

### Deploy to AWS ECR

#### Option 1: Using Deployment Script

```bash
# Set your AWS region
export AWS_REGION=us-east-1

# Make script executable
chmod +x deploy_aws.sh

# Run deployment
./deploy_aws.sh
```

#### Option 2: Manual Deployment

```bash
# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Create ECR repository
aws ecr create-repository --repository-name stroke-prediction --region us-east-1

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Build and push
docker build -t stroke-prediction .
docker tag stroke-prediction:latest $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/stroke-prediction:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/stroke-prediction:latest
```

### Deploy to ECS Fargate

1. Update `aws-ecs-task-definition.json` with your ECR image URI
2. Create ECS cluster and service:

```bash
# Create task definition
aws ecs register-task-definition --cli-input-json file://aws-ecs-task-definition.json

# Create service (adjust cluster and subnet IDs)
aws ecs create-service \
    --cluster your-cluster-name \
    --service-name stroke-prediction \
    --task-definition stroke-prediction \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

### Alternative AWS Services

- **AWS App Runner**: Serverless container service
- **AWS Lambda**: Using container images (for smaller workloads)
- **Elastic Beanstalk**: Traditional PaaS deployment

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
├── app.yaml                          # GCP Cloud Run configuration
├── cloudbuild.yaml                   # GCP Cloud Build configuration
├── .gcloudignore                     # GCP ignore file
├── deploy_gcp.sh                     # GCP deployment script
├── deploy_aws.sh                     # AWS deployment script
├── aws-ecs-task-definition.json      # AWS ECS task definition
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
**4️⃣ Python Script**
```python
import requests

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

response = requests.post('http://localhost:8000/predict', json=patient)
result = response.json()

if result['stroke']:
    print(f"⚠️ High risk! Probability: {result['stroke_probability']:.2%}")
else:
    print(f"✅ Low risk! Probability: {result['stroke_probability']:.2%}")
```

---

## 📁 Project Structure

```
📦 mlzoomcamp_mid_project
├── 📓 mid_project.ipynb          # The training journey
├── 📊 healthcare-dataset-stroke-data.csv
├── 🏋️ train.py                  # Model training script
├── 🔮 predict.py                 # FastAPI magic
├── 🧪 test.py                   # API testing
├── 🐳 Dockerfile                # Container config
├── 🔒 uv.lock                   # Dependency lock
└── 📈 EDA/                      # Pretty visualizations
```

---

## 🛑 Stopping the App

- **Docker**: Press `Ctrl+C` or run `docker stop <container_id>`
- **Local**: Press `Ctrl+C`

---

## 🎯 What Could Be Better?

### Current Challenges

- **📉 Imbalanced Data**: Only ~5% stroke cases makes training tricky
- **🔍 Limited Features**: Missing family history, genetics, medications, and detailed lifestyle info
- **🎲 Threshold Sensitivity**: The yes/no cutoff needs careful tuning
- **🌍 Geographic Limits**: May not generalize to all populations

### 🌟 Dream Features (Coming Soon?)

- **🔮 SHAP/LIME Explanations**: "Why did the model predict this?"
- **🎨 Pretty Web Interface**: For non-technical users
- **📊 Risk Levels**: Low/Medium/High instead of just yes/no
- **☁️ Cloud Deployment**: Kubernetes, AWS, GCP ready!
- **📱 Batch Processing**: Predict multiple patients at once
- **🔐 Security**: Authentication, rate limiting, HIPAA compliance
- **📈 Monitoring**: Track model performance over time

---

## 💌 Let's Connect!

I'd love to hear your thoughts, ideas, or feedback! 

**📧 Email**: liting.liu.work@gmail.com  
**💼 LinkedIn**: [Liting Liu](https://www.linkedin.com/in/liting-liu/)

*Made with 💙 and lots of ☕ by Liting*

---

⭐ If you found this helpful, give it a star! ⭐
