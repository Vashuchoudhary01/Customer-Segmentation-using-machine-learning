import pandas as pd
import pickle

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def train_pipeline(data_path: str):
    # 1. Load data
    df = pd.read_csv(data_path)

    # 2. Features (already engineered dataframe)
    X = df.copy()

    # 3. Preprocessor (scale all features)
    preprocessor = ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), X.columns)
        ]
    )

    # 4. Full ML pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", KMeans(n_clusters=3, random_state=42))
        ]
    )

    # 5. Train
    pipeline.fit(X)

    # 6. Save pipeline
    with open("notebooks/artifacts/customer_segmentation_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    print("Training completed. Pipeline saved.")


if __name__ == "__main__":
    train_pipeline(r"C:\Users\91906\Desktop\customer segmentation end to end\notebooks\data_1\clustered_data.csv")
