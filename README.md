# Delivery Location Clustering System

## Project Overview

The **Delivery Location Clustering System** is a machine learning project that groups delivery locations based on their geographical coordinates (latitude and longitude). It uses the K-Means clustering algorithm to identify delivery areas and helps visualize delivery locations through an interactive map.

The application is developed using Python and Flask, allowing users to enter new coordinates and predict the corresponding delivery area.

## Business Problem

Delivery companies often handle deliveries across multiple geographical locations. Managing these locations without proper grouping can make delivery planning and area management more difficult.

This project addresses the problem by clustering nearby geographical locations into groups, making delivery areas easier to visualize and analyze.

## Objectives

- Group delivery locations using geographical coordinates.
- Identify delivery clusters using the K-Means algorithm.
- Visualize delivery locations and cluster centers on an interactive map.
- Predict the cluster for a new delivery location.
- Display delivery area summaries through a web dashboard.

## Technologies Used

- **Python** – Core programming language
- **Pandas** – Data loading and preprocessing
- **Scikit-learn** – K-Means clustering
- **Joblib** – Saving and loading the trained model
- **Flask** – Web application framework
- **Folium** – Interactive geographical map
- **HTML and CSS** – Frontend design

## Project Structure

```text
Delivery_Location_Clustering/
├── data/
│   └── delivery_locations_800.csv
├── notebooks/
│   └── delivery_clustering.ipynb
├── models/
│   └── kmeans_model.pkl
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── app.py
├── requirements.txt
└── README.md
```

## Machine Learning Algorithm

This project uses **K-Means Clustering**, an unsupervised machine learning algorithm.

The algorithm groups delivery locations into five clusters based on latitude and longitude. Each cluster has a center representing the approximate geographic center of its group.

The trained model is saved using Joblib and reused to predict the cluster for new delivery coordinates.

## Key Features

- **Data Preprocessing:** Validates geographical coordinates and removes invalid or duplicate delivery records.
- **Location Clustering:** Groups delivery locations into five clusters.
- **Interactive Map:** Displays delivery locations using different colors for different clusters.
- **Cluster Centers:** Highlights the center of each delivery group.
- **Area Prediction:** Predicts the cluster for newly entered latitude and longitude.
- **Dashboard:** Displays total deliveries, delivery areas, and cluster summaries.
- **Error Handling:** Handles invalid inputs and common application errors.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
cd Delivery_Location_Clustering
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

```bash
python src/train.py
```

This trains the K-Means model and saves it in the `models/` directory.

### 5. Run the Flask Application

```bash
python app.py
```

## How to Use

1. Start the Flask application.
2. View the delivery area summaries on the dashboard.
3. Explore delivery locations and cluster centers on the interactive map.
4. Enter the latitude and longitude of a new delivery location.
5. Click **Predict Area** to see the predicted cluster and its approximate area label.

## Dataset

The project uses a CSV dataset containing delivery IDs, latitude, and longitude.

The included dataset contains 800 synthetic delivery records with coordinates approximately around Bhubaneswar, Odisha. These records are intended for demonstration and do not represent verified real-world deliveries.

## Future Improvements

- Use real-world delivery location data.
- Determine the optimal number of clusters using the Elbow Method or Silhouette Score.
- Use road-network distances instead of straight-line coordinate distances.
- Integrate delivery time and order volume information.
- Develop delivery route optimization features.
- Deploy the application to a cloud platform.

## Conclusion

The Delivery Location Clustering System demonstrates how unsupervised machine learning can organize geographical delivery data into meaningful groups. By combining K-Means clustering, an interactive map, and a Flask dashboard, the project provides a practical foundation for exploring delivery area management.

**Note:** The cluster labels are approximate directional names generated from cluster centers, not verified neighborhood names. The model groups coordinates geographically; it does not directly optimize delivery routes or travel times.
