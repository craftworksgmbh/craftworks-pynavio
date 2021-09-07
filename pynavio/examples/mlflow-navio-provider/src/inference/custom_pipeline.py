import xgboost as xgb
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler


class CustomPipeline:
    def __init__(self, artifacts: dict):
        self._artifact_validation(artifacts)
        self.xgb_model = pickle.load(open(artifacts["xgb_model"], 'rb'))
        self.standard_scaler = pickle.load(open(artifacts["standard_scaler"], 'rb'))
        
    def _artifact_validation(self, artifacts: dict):
        print("A custom function is validating artifacts for showcase purpouse :)")
        if "xgb_model" not in artifacts:
            raise ValueError("xgb model is missing")
            
        elif "standard_scaler" not in artifacts:
            raise ValueError("standard scaler is missing")
        
    def preprocess(self, X: pd.DataFrame):
        return self.standard_scaler.transform(X.values)
        
    def predict(self, X: pd.DataFrame):
        input_prep = self.preprocess(X)
        pred = self.xgb_model.predict(input_prep)
        return pred

