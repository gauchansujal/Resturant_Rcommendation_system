ML-powered restaurant finder for Thamel, Kathmandu.

✨ Features

🗺️ Interactive map with 78 restaurant pins
🔍 Search by name or street
🍽️ Filter by cuisine, rating, WiFi, Veg, Outdoor
⭐ User reviews and ratings
🤖 ML-powered similar restaurants
✨ Personalized "For You" recommendations
📊 Analytics dashboard with live charts
🌙 Dark mode

🤖 ML Models Used
Model                                 Purpose
ModelPurposeCosine                    SimilarityFind similar restaurants
Content-Based                         FilteringPersonalized recommendations
Random Forest                         RegressorPredict restaurant ratings
K-Means                               ClusteringGroup restaurants by type

🛠️ Tech Stack
Backend
Python, Flask

Frontend
HTML, CSS, JavaScript
Leaflet.js (map)
OpenStreetMap (map tiles)

Machine Learning
scikit-learn (similarity, clustering, regression)
pandas, NumPy (data processing)
matplotlib, seaborn (charts)


Data


OpenStreetMap via Overpass Turbo (GeoJSON)

project/
├── app.py                        # Flask backend + all API endpoints
├── restaurants_clean.json        # Cleaned restaurant dataset (78 restaurants)
├── requirements.txt              # Python dependencies
│
├── data/
│   ├── clean_data.py             # Data cleaning script (run once)
│   ├── Resturant_data_for_thamel.geojson  # Raw OpenStreetMap data
│   └── reviews.json              # User reviews (auto-created)
│
├── ml/
│   ├── recommender.py            # Core ML engine (all 4 algorithms)
│   ├── generate_charts.py        # Chart generation (matplotlib/seaborn)
│   └── charts/                   # Auto-generated PNG chart images
│
├── templates/
│   └── index.html                # Full frontend (map, filters, ML UI)
│
└── static/                       # Static assets folder

