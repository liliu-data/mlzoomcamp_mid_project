import pickle
from typing import Dict, Any

class StrokeModel:
    def __init__(self):
        self.model = None
        self.vectorizer = None
    
    def load_model(self):
        """Load the trained model and vectorizer"""
        try:
            # We'll create these model files later
            with open('models/stroke_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            with open('models/dict_vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            # For now, we'll use a dummy model for testing
            self.model = None
            self.vectorizer = None
            return False
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction on input data"""
        if self.model is None:
            # Return dummy prediction for testing
            return {
                'stroke_probability': 0.25,
                'prediction': 'low_risk',
                'risk_level': 'low'
            }
        
        # Real prediction logic will go here later
        # For now, return a dummy response
        return {
            'stroke_probability': 0.35,
            'prediction': 'low_risk', 
            'risk_level': 'low'
        }

# Global model instance
stroke_model = StrokeModel()