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

try:
    response = requests.post(url, json=patient)
    response.raise_for_status()  # Raise an error for bad status codes
    predictions = response.json()

    if predictions['stroke']:
        print(f"The patient is likely to have a stroke with a probability of {predictions['stroke_probability']}")
    else:
        print(f"The patient is unlikely to have a stroke with a probability of {predictions['stroke_probability']}")
except requests.exceptions.ConnectionError:
    print("Error: Could not connect to the server.")
    print("Please make sure the FastAPI server is running:")
    print("  uv run uvicorn predict:app --host 0.0.0.0 --port 8000 --reload")
except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"An error occurred: {e}")