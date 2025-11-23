# 🧠💙 Stroke Prediction ML Project

> *Predicting strokes with machine learning, one heartbeat at a time* ✨

Welcome to my MLZoomCamp midterm project! This adorable little ML model helps predict stroke occurrences using health and demographic data. Let's keep people healthy together! 🏥💪

---

## 📊 The Data Story

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

**Get predictions like this:**

```json
{
  "stroke_probability": 0.0737,
  "stroke": false
}
```
**Try it out on people you know and tell them if they are at risk of stroke (oop!✨)**

### 🧪 Four Ways to Test

**1️⃣ Interactive Docs** (Easiest!)
- Visit http://localhost:8000/docs
- Click "Try it out" on `/predict`
- Enter data & hit "Execute"
- [Watch it in action!](https://drive.google.com/file/d/1UiN0Eb22zt1v1giO0n15xG60JwALkcRV/view?usp=sharing) 🎥

**2️⃣ Test Script**
```bash
python test.py
```

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
