
import os
import joblib
import pandas as pd

from sklearn.cluster import KMeans

from preprocessing import preprocess_data, select_features


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR, "data", "delivery_locations_800.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(
    MODEL_DIR, "kmeans_model.pkl"
)


def train_model():
    try:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(
                f"Dataset not found: {DATA_PATH}"
            )

        # Load dataset.
        df = pd.read_csv(DATA_PATH)

        # Preprocess data.
        df = preprocess_data(df)

        # Select features.
        X = select_features(df)

        if len(X.drop_duplicates()) < 5:
            raise ValueError(
                "At least five unique locations are required."
            )

        model = KMeans(
            n_clusters=5,
            random_state=42,
            n_init=10
        )

        model.fit(X)

        # Create the models directory if necessary.
        os.makedirs(MODEL_DIR, exist_ok=True)

        # Save the trained model.
        joblib.dump(model, MODEL_PATH)

        # Display cluster assignments.
        df["Cluster"] = model.predict(X)

        print("Model training completed successfully!")
        print(f"Model saved at: {MODEL_PATH}")
        print(f"Total valid deliveries: {len(df)}")
        print(f"Total clusters: {model.n_clusters}")

        print("\nDelivery count by cluster:")
        print(df["Cluster"].value_counts().sort_index())

        print("\nCluster centers:")
        print(
            pd.DataFrame(
                model.cluster_centers_,
                columns=["Latitude", "Longitude"]
            )
        )

        return model

    except Exception as exc:
        print(f"Model training failed: {exc}")
        raise


if __name__ == "__main__":
    train_model()
