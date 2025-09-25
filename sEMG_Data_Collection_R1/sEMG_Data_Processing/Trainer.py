# trainer.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

class Trainer:
    def __init__(self, model=None):
        self.model = model

    def train(self, epochs=10, df=None):
        """
        Train model on feature DataFrame.
        Args:
            df (pd.DataFrame): features + labels
        """
       

    def save_model(self, path="fatigue_model.pkl"):
        joblib.dump(self.model, path)

    def load_model(self, path="fatigue_model.pkl"):
        self.model = joblib.load(path)