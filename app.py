import os

import pandas as pd
import folium

from flask import Flask, render_template, request
from src.preprocessing import preprocess_data
from src.predict import load_model, predict_delivery_area


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "delivery_locations_800.csv"
)

# Its responsibility is to prepare the delivery dataset and assign a cluster to every delivery location.
def get_project_data():
    """Load the dataset, clean it, and assign cluster labels."""

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)
    df = preprocess_data(df)

    model = load_model()

    features = df[["Latitude", "Longitude"]]
    df["Cluster"] = model.predict(features)

    return df, model

# This function attempts to convert numeric cluster IDs into readable names. Like : 
# Cluster 0 — Central Delivery Area
# Cluster 1 — Southern Delivery Area
# Cluster 2 — Eastern Delivery Area

# Cluster 1 has a lower latitude, so it is farther south relative to the reference center.
# Cluster 3 has a higher latitude, so it is farther north.
# Cluster 2 has a higher longitude, so it is farther east.
# Cluster 4 has a lower longitude, so it is farther west.

# {
#     0: "Central Delivery Area",
#     1: "Southern Delivery Area",
#     2: "Eastern Delivery Area",
#     3: "Northern Delivery Area",
#     4: "Western Delivery Area"
# }
def get_area_names(model):
    """
    Create readable directional names from cluster centers.
    These are approximate labels, not verified neighborhood names.
    """

    centers = model.cluster_centers_

    center_latitude = centers[:, 0].mean()
    center_longitude = centers[:, 1].mean()

    distances = (
        (centers[:, 0] - center_latitude) ** 2
        + (centers[:, 1] - center_longitude) ** 2
    )

    central_cluster = int(distances.argmin())

    area_names = {
        central_cluster: "Central Delivery Area"
    }

    used_names = {"Central Delivery Area"}

    for cluster_id, center in enumerate(centers):
        # Skip the central delivery area
        if cluster_id == central_cluster:
            continue

        lat_difference = center[0] - center_latitude
        lon_difference = center[1] - center_longitude
        # Choose the direction
        if abs(lat_difference) >= abs(lon_difference):
            direction = (
                "Northern" if lat_difference >= 0
                else "Southern"
            )
        else:
            direction = (
                "Eastern" if lon_difference >= 0
                else "Western"
            )

        name = f"{direction} Delivery Area"
        suffix = 2

        while name in used_names:
            name = f"{direction} Delivery Area {suffix}"
            suffix += 1

        area_names[cluster_id] = name
        used_names.add(name)

    return area_names

# display all delivery locations on an interactive map and show which K-Means cluster each delivery belongs to.
# Why do we do this? -> So the map initially opens near the delivery locations instead of showing an unrelated place.
def create_map(df, model, area_names):
    """Create an interactive Folium map."""

    delivery_map = folium.Map(
        location=[
            df["Latitude"].mean(),
            df["Longitude"].mean()
        ],
        zoom_start=12
    )

    colors = [
        "blue", "red", "green", "purple", "orange"
    ]

    for _, row in df.iterrows():
        cluster_id = int(row["Cluster"])
        area_name = area_names[cluster_id]
        color = colors[cluster_id % len(colors)]

        popup = (
            f"Delivery ID: {row['Delivery_ID']}<br>"
            f"Cluster: {cluster_id}<br>"
            f"Area: {area_name}<br>"
            f"Latitude: {row['Latitude']:.6f}<br>"
            f"Longitude: {row['Longitude']:.6f}"
        )

        folium.CircleMarker(
            location=[
                row["Latitude"],
                row["Longitude"]
            ],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            tooltip=area_name,
            popup=folium.Popup(popup, max_width=300)
        ).add_to(delivery_map)

    # Mark the center of each cluster.
    for cluster_id, center in enumerate(model.cluster_centers_):
        folium.Marker(
            location=[center[0], center[1]],
            tooltip=area_names[cluster_id],
            popup=area_names[cluster_id],
            icon=folium.Icon(
                color=colors[cluster_id % len(colors)],
                icon="info-sign"
            )
        ).add_to(delivery_map)

    return delivery_map._repr_html_()


@app.route("/", methods=["GET", "POST"])
def home():
    try:
        df, model = get_project_data()
        area_names = get_area_names(model)

        areas = []

        for cluster_id in sorted(df["Cluster"].unique()):
            cluster_id = int(cluster_id)
            cluster_df = df[df["Cluster"] == cluster_id]

            areas.append({
                "cluster_id": cluster_id,
                "name": area_names[cluster_id],
                "count": len(cluster_df),
                "latitude": round(
                    model.cluster_centers_[cluster_id][0], 6
                ),
                "longitude": round(
                    model.cluster_centers_[cluster_id][1], 6
                )
            })

        prediction = None
        error = None

        if request.method == "POST":
            try:
                latitude = float(request.form["latitude"])
                longitude = float(request.form["longitude"])

                result = predict_delivery_area(
                    latitude, longitude
                )

                cluster_id = result["cluster_id"]

                prediction = {
                    **result,
                    "name": area_names[cluster_id]
                }

            except (ValueError, TypeError) as exc:
                error = str(exc)

        map_html = create_map(df, model, area_names)

        return render_template(
            "index.html",
            areas=areas,
            prediction=prediction,
            error=error,
            total_deliveries=len(df),
            total_clusters=len(areas),
            map_html=map_html
        )

    except Exception as exc:
        app.logger.exception("Application failed")
        return f"Application error: {exc}", 500


if __name__ == "__main__":
    app.run(debug=True)
