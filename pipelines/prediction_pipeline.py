import pickle
import pandas as pd


class PredictionPipeline:
    def __init__(self):
        with open("notebooks/artifacts/customer_segmentation_pipeline.pkl", "rb") as f:
            self.model = pickle.load(f)

    def predict(self, input_data: dict):
        """
        input_data: dict with feature_name -> value
        """
        df = pd.DataFrame([input_data])
        prediction = self.model.predict(df)
        return int(prediction[0])
