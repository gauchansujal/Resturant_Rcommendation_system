"""
recommender.py
====================================================================
Machine Learning engine for Thamel Eats.

Implements three classic, well-understood ML techniques on top of
the cleaned restaurant dataset:

1. CONTENT-BASED SIMILARITY
   Every restaurant is converted into a numeric feature vector
   (cuisine one-hot + amenities + normalised rating). Cosine
   similarity between vectors tells us which restaurants are
   "like" a given one. This powers /api/ml/similar/<id>.

2. PERSONALIZED RECOMMENDATIONS
   A user's liked/reviewed restaurants are averaged into a single
   "user profile vector" in the same feature space. We rank every
   restaurant the user hasn't tried by cosine similarity to that
   profile. This is the standard "content-based recommender"
   pattern used by Netflix-style systems before collaborative
   filtering is layered on top. Powers /api/ml/recommend/<user>.

3. RATING PREDICTION (regression)
   A Random Forest Regressor is trained to predict a restaurant's
   rating purely from its structural features (cuisine, amenities,
   completeness of listing). This demonstrates supervised learning
   and gives a sanity check / second opinion on our synthetic
   ratings. Powers /api/ml/predict_rating.

4. CLUSTERING
   K-Means groups restaurants into K "types" (e.g. casual / cafe /
   fine-dining) based on the same feature vectors, purely
   unsupervised. Powers /api/ml/clusters.

All models are intentionally simple and interpretable — this is a
restaurant dataset with ~78 rows, not a Netflix-scale problem, so
we favour algorithms that train in milliseconds and are easy to
explain in a project report over deep learning.
====================================================================
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


class RestaurantRecommender:
    def __init__(self, restaurants_path, reviews_path=None):
        self.restaurants_path = restaurants_path
        self.reviews_path = reviews_path
        self.df = None
        self.feature_matrix = None
        self.feature_names = None
        self.mlb = None
        self.scaler = None
        self.kmeans = None
        self.rf_model = None
        self.rf_metrics = {}
        self._load_and_build()

    # ────────────────────────────────────────────────────────────
    # DATA LOADING & FEATURE ENGINEERING
    # ────────────────────────────────────────────────────────────
    def _load_and_build(self):
        with open(self.restaurants_path, "r", encoding="utf-8") as f:
            raw = json.load(f)["restaurants"]
        self.df = pd.DataFrame(raw)
        self._build_features()
        self._fit_clusters()
        self._fit_rating_model()

    def _build_features(self):
        """
        Turn each restaurant into a numeric feature vector:
          - One-hot encoded cuisines (multi-label, since a place can
            serve several cuisines)
          - Binary amenity flags: wifi, vegetarian, outdoor_seating, takeaway
          - Normalised rating (0-1 scale)
          - "Listing completeness" score (has phone/website/hours/description)
        """
        df = self.df

        # --- cuisine multi-hot encoding ---
        cuisine_lists = df["cuisine"].apply(lambda x: x if isinstance(x, list) else [])
        self.mlb = MultiLabelBinarizer()
        cuisine_matrix = self.mlb.fit_transform(cuisine_lists)
        cuisine_cols = [f"cuisine_{c}" for c in self.mlb.classes_]
        cuisine_df = pd.DataFrame(cuisine_matrix, columns=cuisine_cols, index=df.index)

        # --- amenity flags ---
        amenity_df = pd.DataFrame({
            "wifi":            df["wifi"].fillna(False).astype(int),
            "vegetarian":      df["vegetarian"].fillna(False).astype(int),
            "outdoor_seating": df["outdoor_seating"].fillna(False).astype(int),
            "takeaway":        df["takeaway"].fillna(False).astype(int),
        }, index=df.index)

        # --- listing completeness (proxy for "quality of data", useful signal) ---
        completeness = pd.DataFrame({
            "has_phone":       df["phone"].notna().astype(int),
            "has_website":     df["website"].notna().astype(int),
            "has_hours":       df["opening_hours"].notna().astype(int),
            "has_description": df["description"].notna().astype(int),
        }, index=df.index)

        # --- normalised rating ---
        self.scaler = MinMaxScaler()
        rating_scaled = self.scaler.fit_transform(df[["rating"]])
        rating_df = pd.DataFrame(rating_scaled, columns=["rating_norm"], index=df.index)

        # Combine into the final feature matrix used by similarity / clustering
        self.feature_matrix = pd.concat(
            [cuisine_df, amenity_df, completeness, rating_df], axis=1
        )
        self.feature_names = list(self.feature_matrix.columns)

    # ────────────────────────────────────────────────────────────
    # 1. CONTENT-BASED SIMILARITY  ("restaurants like this one")
    # ────────────────────────────────────────────────────────────
    def similar_restaurants(self, restaurant_id, top_n=5):
        if restaurant_id not in self.df["id"].values:
            return []

        idx = self.df.index[self.df["id"] == restaurant_id][0]
        target_vec = self.feature_matrix.loc[[idx]]

        sims = cosine_similarity(target_vec, self.feature_matrix)[0]
        sim_series = pd.Series(sims, index=self.df.index)
        sim_series = sim_series.drop(idx)  # exclude itself
        top_idx = sim_series.sort_values(ascending=False).head(top_n).index

        results = []
        for i in top_idx:
            row = self.df.loc[i]
            results.append({
                "id":              row["id"],
                "name":            row["name"],
                "primary_cuisine": row["primary_cuisine"],
                "rating":          row["rating"],
                "similarity":      round(float(sim_series.loc[i]), 3),
            })
        return results

    # ────────────────────────────────────────────────────────────
    # 2. PERSONALIZED RECOMMENDATIONS
    # ────────────────────────────────────────────────────────────
    def recommend_for_user(self, liked_restaurant_ids, top_n=8):
        """
        Build a 'user profile vector' by averaging the feature vectors
        of restaurants the user liked / reviewed positively, then
        recommend the most similar restaurants they haven't tried yet.
        Classic content-based recommender pattern.
        """
        liked_idx = self.df.index[self.df["id"].isin(liked_restaurant_ids)]
        if len(liked_idx) == 0:
            # Cold start: no history yet -> fall back to highest rated
            top = self.df.sort_values("rating", ascending=False).head(top_n)
            return [{
                "id": r["id"], "name": r["name"], "primary_cuisine": r["primary_cuisine"],
                "rating": r["rating"], "score": None, "reason": "popular"
            } for _, r in top.iterrows()]

        user_profile = self.feature_matrix.loc[liked_idx].mean(axis=0).values.reshape(1, -1)
        sims = cosine_similarity(user_profile, self.feature_matrix)[0]
        sim_series = pd.Series(sims, index=self.df.index)

        # Don't recommend restaurants already liked
        sim_series = sim_series.drop(liked_idx)
        top_idx = sim_series.sort_values(ascending=False).head(top_n).index

        results = []
        for i in top_idx:
            row = self.df.loc[i]
            results.append({
                "id":              row["id"],
                "name":            row["name"],
                "primary_cuisine": row["primary_cuisine"],
                "rating":          row["rating"],
                "score":           round(float(sim_series.loc[i]), 3),
                "reason":          "personalized",
            })
        return results

    # ────────────────────────────────────────────────────────────
    # 3. RATING PREDICTION (supervised regression)
    # ────────────────────────────────────────────────────────────
    def _fit_rating_model(self):
        """
        Train a Random Forest Regressor to predict `rating` from the
        structural features (cuisine + amenities + completeness),
        EXCLUDING rating_norm itself (that would be cheating/leakage).
        """
        feature_cols = [c for c in self.feature_names if c != "rating_norm"]
        X = self.feature_matrix[feature_cols]
        y = self.df["rating"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=200, max_depth=6, random_state=42, min_samples_leaf=2
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        self.rf_metrics = {
            "mae": round(float(mean_absolute_error(y_test, preds)), 3),
            "r2":  round(float(r2_score(y_test, preds)), 3),
            "n_train": len(X_train),
            "n_test":  len(X_test),
        }

        # Refit on ALL data for actual predictions (common practice once validated)
        model.fit(X, y)
        self.rf_model = model
        self._rf_feature_cols = feature_cols

        # Feature importances -> useful for the report / dashboard
        importances = sorted(
            zip(feature_cols, model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        self.rf_feature_importance = [
            {"feature": f, "importance": round(float(v), 4)} for f, v in importances[:10]
        ]

    def predict_rating(self, restaurant_id):
        if restaurant_id not in self.df["id"].values:
            return None
        idx = self.df.index[self.df["id"] == restaurant_id][0]
        x = self.feature_matrix.loc[[idx], self._rf_feature_cols]
        pred = self.rf_model.predict(x)[0]
        actual = float(self.df.loc[idx, "rating"])
        return {
            "id": restaurant_id,
            "name": self.df.loc[idx, "name"],
            "predicted_rating": round(float(pred), 2),
            "actual_rating": actual,
            "difference": round(float(pred) - actual, 2),
        }

    def predict_all_ratings(self):
        x = self.feature_matrix[self._rf_feature_cols]
        preds = self.rf_model.predict(x)
        out = []
        for i, idx in enumerate(self.df.index):
            out.append({
                "id": self.df.loc[idx, "id"],
                "name": self.df.loc[idx, "name"],
                "predicted_rating": round(float(preds[i]), 2),
                "actual_rating": float(self.df.loc[idx, "rating"]),
            })
        return out

    # ────────────────────────────────────────────────────────────
    # 4. CLUSTERING (unsupervised)
    # ────────────────────────────────────────────────────────────
    def _fit_clusters(self, k=5):
        X = self.feature_matrix.values
        self.kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(X)
        self.df["cluster"] = labels

        # Auto-name clusters based on their dominant cuisine + amenities
        self.cluster_labels = {}
        for c in range(k):
            sub = self.df[self.df["cluster"] == c]
            top_cuisine = sub["primary_cuisine"].mode()
            top_cuisine = top_cuisine.iloc[0] if len(top_cuisine) else "Mixed"
            avg_rating = round(sub["rating"].mean(), 2)
            self.cluster_labels[c] = {
                "label": f"{top_cuisine} cluster",
                "size": int(len(sub)),
                "avg_rating": float(avg_rating),
            }

    def get_clusters(self):
        result = []
        for c, meta in self.cluster_labels.items():
            members = self.df[self.df["cluster"] == c][["id", "name", "primary_cuisine", "rating"]]
            result.append({
                "cluster_id": int(c),
                "label": meta["label"],
                "size": meta["size"],
                "avg_rating": meta["avg_rating"],
                "members": members.to_dict("records"),
            })
        return sorted(result, key=lambda x: -x["size"])

    # ────────────────────────────────────────────────────────────
    # Model report (for an ML-focused dashboard / project writeup)
    # ────────────────────────────────────────────────────────────
    def model_report(self):
        return {
            "dataset_size": int(len(self.df)),
            "feature_count": int(len(self.feature_names)),
            "feature_names_sample": self.feature_names[:15],
            "rating_model": {
                "algorithm": "RandomForestRegressor",
                "mae": self.rf_metrics.get("mae"),
                "r2": self.rf_metrics.get("r2"),
                "train_size": self.rf_metrics.get("n_train"),
                "test_size": self.rf_metrics.get("n_test"),
                "top_features": self.rf_feature_importance,
            },
            "clustering": {
                "algorithm": "KMeans",
                "k": len(self.cluster_labels),
                "clusters": [
                    {"id": c, **meta} for c, meta in self.cluster_labels.items()
                ],
            },
            "similarity": {
                "algorithm": "Cosine similarity over content-based feature vectors",
            },
        }


# ────────────────────────────────────────────────────────────────
# Standalone test / demo when run directly: `python recommender.py`
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    here = os.path.dirname(__file__)
    data_path = os.path.join(here, "..", "restaurants_clean.json")
    if not os.path.exists(data_path):
        data_path = os.path.join(here, "..", "data", "restaurants_clean.json")

    rec = RestaurantRecommender(data_path)

    print("=" * 60)
    print("MODEL REPORT")
    print("=" * 60)
    report = rec.model_report()
    print(json.dumps(report, indent=2))

    print("\n" + "=" * 60)
    print("SIMILAR TO FIRST RESTAURANT")
    print("=" * 60)
    first_id = rec.df.iloc[0]["id"]
    print(f"Base: {rec.df.iloc[0]['name']}")
    for s in rec.similar_restaurants(first_id, top_n=5):
        print(f"  {s['similarity']:.3f}  {s['name']} ({s['primary_cuisine']})")

    print("\n" + "=" * 60)
    print("PERSONALIZED RECS (liked 2 Italian places)")
    print("=" * 60)
    italian_ids = rec.df[rec.df["primary_cuisine"].str.contains("Italian", na=False)]["id"].head(2).tolist()
    for r in rec.recommend_for_user(italian_ids, top_n=5):
        print(f"  {r.get('score')}  {r['name']} ({r['primary_cuisine']})")

    print("\n" + "=" * 60)
    print("RATING PREDICTION SAMPLE")
    print("=" * 60)
    for rid in rec.df["id"].head(5):
        p = rec.predict_rating(rid)
        print(f"  {p['name']:<35} predicted={p['predicted_rating']}  actual={p['actual_rating']}  diff={p['difference']}")