import joblib

class ModelHandler:
    def __init__(self, model_path="models/emg_fatigue_model.pkl"): # Path to the pre-trained model
        self.model = joblib.load(model_path)

    def predict(self, features):
        return self.model.predict([features])[0]