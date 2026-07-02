# 🍽️ Thamel Eats — Restaurant Recommendation System

ML-powered restaurant finder for Thamel, Kathmandu, Nepal.

---

## 🚀 Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/thamel-eats.git
cd thamel-eats
pip install -r requirements.txt
python app.py
```
Open → `http://127.0.0.1:5000`

---

## 📁 Folder Structure

```
monaj/
├── app.py
├── restaurants_clean.json
├── requirements.txt
├── README.md
├── data/
│   ├── clean_data.py
│   └── Resturant_data_for_thamel.geojson
├── ml/
│   ├── recommender.py
│   └── generate_charts.py
└── templates/
    └── index.html
```

---

## ✨ Features

- 🗺️ Interactive map with 78 restaurant pins
- 🔍 Search by name or street
- 🍽️ Filter by cuisine, rating, WiFi, Veg, Outdoor
- ⭐ User reviews and star ratings
- 🤖 ML-powered similar restaurants
- ✨ Personalized "For You" recommendations
- 📊 Analytics dashboard with live charts
- 🌙 Dark mode toggle

---

## 🤖 ML Models

| Model | Purpose |
|-------|---------|
| Cosine Similarity | Find similar restaurants |
| Content-Based Filtering | Personalized recommendations |
| Random Forest Regressor | Predict restaurant ratings |
| K-Means Clustering | Group restaurants by type |

**Accuracy:**

- MAE: ±0.21 stars
- Correlation: r = 0.78
- 100% predictions within ±0.5 stars

---

## 🛠️ Tech Stack

**Backend**

- Python 3.10+
- Flask 3.1

**Frontend**

- HTML, CSS, JavaScript
- Leaflet.js — interactive map
- OpenStreetMap — free map tiles

**Machine Learning**

- scikit-learn — similarity, clustering, regression
- pandas, NumPy — data processing
- matplotlib, seaborn — chart generation

**Data**

- OpenStreetMap via Overpass Turbo (GeoJSON format)

---

## 📡 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/restaurants` | Get restaurants with filters |
| `GET /api/cuisines` | Get all cuisine types |
| `POST /api/reviews` | Post a review |
| `GET /api/ml/similar/<id>` | Similar restaurants |
| `GET /api/ml/recommend` | Personalized recommendations |
| `POST /api/ml/like` | Like a restaurant |
| `GET /api/ml/clusters` | K-Means cluster groups |
| `GET /api/ml/report` | Model accuracy report |
| `GET /api/ml/chart/<name>` | Get chart as PNG |

---

## 📈 Charts

| Chart | Type |
|-------|------|
| Cuisine Distribution | Pie Chart |
| Feature Importance | Bar Graph |
| Predicted vs Actual Rating | Scatter Chart |
| Rating Prediction Trend | Line Graph |
| Rating Distribution | ggplot-style Histogram |
| K-Means Clusters | Scatter (PCA projection) |

---

## 👤 Author

College ML Project — Kathmandu, Nepal
