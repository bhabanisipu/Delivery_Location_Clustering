import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and preprocess delivery location data.
    """

    try:
        required_columns = [
            "Delivery_ID",
            "Latitude",
            "Longitude"
        ]

        missing_columns = [
            column for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        df = df.copy()

        # Convert coordinates to numeric values.
        df["Latitude"] = pd.to_numeric(
            df["Latitude"], errors="coerce"
        )

        df["Longitude"] = pd.to_numeric(
            df["Longitude"], errors="coerce"
        )

        df = df.dropna(
            subset=["Delivery_ID", "Latitude", "Longitude"]
        )

        df = df[
            df["Latitude"].between(-90, 90)
            & df["Longitude"].between(-180, 180)
        ]

        # Remove duplicate delivery IDs.
        df = df.drop_duplicates(
            subset=["Delivery_ID"]
        )

        if df.empty:
            raise ValueError(
                "No valid delivery records remain after preprocessing."
            )

        return df.reset_index(drop=True)

    except Exception as exc:
        raise ValueError(
            f"Preprocessing failed: {exc}"
        ) from exc


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the features required by the K-Means model.
    """

    try:
        required_features = ["Latitude", "Longitude"]

        if not all(
            column in df.columns
            for column in required_features
        ):
            raise ValueError(
                "Latitude and Longitude columns are required."
            )

        X = df[required_features].copy()

        return X

    except Exception as exc:
        raise ValueError(
            f"Feature selection failed: {exc}"
        ) from exc
