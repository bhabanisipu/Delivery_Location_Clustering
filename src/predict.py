
import os
import joblib
import pandas as pd


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "kmeans_model.pkl"
)


def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Trained model not found. Run train.py first."
            )

        return joblib.load(MODEL_PATH)

    except Exception as exc:
        raise RuntimeError(
            f"Could not load model: {exc}"
        ) from exc


def predict_delivery_area(latitude, longitude):
    try:
        latitude = float(latitude)
        longitude = float(longitude)

        if not -90 <= latitude <= 90:
            raise ValueError(
                "Latitude must be between -90 and 90."
            )

        if not -180 <= longitude <= 180:
            raise ValueError(
                "Longitude must be between -180 and 180."
            )

        model = load_model()

        # Keep the same feature order used during training.
        new_location = pd.DataFrame(
            [[latitude, longitude]],
            columns=["Latitude", "Longitude"]
        )

        cluster_id = int(model.predict(new_location)[0])

        return {
            "cluster_id": cluster_id,
            "latitude": latitude,
            "longitude": longitude
        }

    except Exception as exc:
        raise ValueError(
            f"Delivery area prediction failed: {exc}"
        ) from exc


if __name__ == "__main__":
    try:
        result = predict_delivery_area(
            latitude=20.2961,
            longitude=85.8245
        )

        print("Prediction Result")
        print("-----------------")
        print(f"Cluster ID: {result['cluster_id']}")
        print(f"Latitude: {result['latitude']}")
        print(f"Longitude: {result['longitude']}")

    except Exception as exc:
        print(exc)
