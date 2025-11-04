import torch
# from Fatigue_Model.Model_Training.Models.RawCNN import RawCNN  # example path

import importlib.util
import sys
import os

# dynamically load the RawCNN module
model_path = os.path.join(os.path.dirname(__file__), "../Fatigue_Model/Model_Training/Models/RawCNN.py")
spec = importlib.util.spec_from_file_location("RawCNN", model_path)
RawCNNModule = importlib.util.module_from_spec(spec)
sys.modules["RawCNN"] = RawCNNModule
spec.loader.exec_module(RawCNNModule)

RawCNN = RawCNNModule.RawCNNRegressor


class ModelHandler:
    def __init__(self, model_path):
        # instantiate the model architecture
        self.model = RawCNN()  
        # load weights into it
        state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, features):
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            output = self.model(x)
            return output.item()