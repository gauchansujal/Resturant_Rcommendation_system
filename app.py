from flask import Flask, render_template, jsonify, request
import json
import os
import sys
from datetime import datetime

# Make the ml/ folder importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ml"))
from recommender import RestaurantRecommender

app = Flask(__name__)

# Auto-detect data folder — works whether JSON is in root or data/ subfolder
_base = os.path.dirname(__file__)
if os.path.exists(os.path.join(_base, "data", "restaurants_clean.json")):
    DATA_PATH    = os.path.join(_base, "data", "restaurants_clean.json")
    REVIEWS_PATH = os.path.join(_base, "data", "reviews.json")
    LIKES_PATH   = os.path.join(_base, "data", "likes.json")
else:
    DATA_PATH    = os.path.join(_base, "restaurants_clean.json")
    REVIEWS_PATH = os.path.join(_base, "reviews.json")
    LIKES_PATH   = os.path.join(_base, "likes.json")

print(f"[INFO] Loading data from: {DATA_PATH}")

# ── ML model — trained once at startup ──────────────────────────────────────
print("[INFO] Training ML models (content-based similarity, clustering, rating prediction)...")
recommender = RestaurantRecommender(DATA_PATH)
print(f"[INFO] ML models ready. {recommender.model_report()['dataset_size']} restaurants, "
      f"{recommender.model_report()['feature_count']} features, "
      f"rating model R²={recommender.rf_metrics.get('r2')}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_restaurants():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["restaurants"]

def load_reviews():
    if not os.path.exists(REVIEWS_PATH):
        return {}
    with open(REVIEWS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_reviews(reviews):
    with open(REVIEWS_PATH, "w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)

def load_likes():
    """likes.json: { "<user_id>": ["node/123", "way/456", ...] }"""
    if not os.path.exists(LIKES_PATH):
        return {}
    with open(LIKES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_likes(likes):
    with open(LIKES_PATH, "w", encoding="utf-8") as f:
        json.dump(likes, f, ensure_ascii=False, indent=2)

def enrich_with_reviews(restaurants, reviews):
    """Merge user review data into each restaurant record."""
    for r in restaurants:
        rid = r["id"]
        r_reviews = reviews.get(rid, [])
        if r_reviews:
            avg = sum(rv["rating"] for rv in r_reviews) / len(r_reviews)
            r["user_rating"]       = round(avg, 1)
            r["user_review_count"] = len(r_reviews)
            r["reviews"]           = r_reviews[-5:]   # last 5 only
        else:
            r["user_rating"]       = None
            r["user_review_count"] = 0
            r["reviews"]           = []
    return restaurants

# ── Unsplash photo mapping by cuisine ─────────────────────────────────────────
# Picsum is a reliable CDN that doesn't block hotlinking (unlike Unsplash).
# Each cuisine gets a fixed seed so the SAME restaurant always gets the SAME photo.
CUISINE_PHOTOS = {
    "Nepali":          "https://picsum.photos/seed/nepali-food/400/300",
    "Indian":          "https://picsum.photos/seed/indian-food/400/300",
    "Chinese":         "https://picsum.photos/seed/chinese-food/400/300",
    "Japanese":        "https://picsum.photos/seed/japanese-food/400/300",
    "Sushi":           "https://picsum.photos/seed/sushi-food/400/300",
    "Italian":         "https://picsum.photos/seed/italian-food/400/300",
    "Italian / Pizza": "https://picsum.photos/seed/pizza-food/400/300",
    "Pizza":           "https://picsum.photos/seed/pizza2-food/400/300",
    "Burger":          "https://picsum.photos/seed/burger-food/400/300",
    "Korean":          "https://picsum.photos/seed/korean-food/400/300",
    "Thai":            "https://picsum.photos/seed/thai-food/400/300",
    "Vietnamese":      "https://picsum.photos/seed/vietnamese-food/400/300",
    "Steak House":     "https://picsum.photos/seed/steak-food/400/300",
    "American":        "https://picsum.photos/seed/american-food/400/300",
    "Cafe / Bakery":   "https://picsum.photos/seed/cafe-food/400/300",
    "Momo":            "https://picsum.photos/seed/momo-food/400/300",
    "default":         "https://picsum.photos/seed/restaurant-default/400/300",
}

def get_photo(restaurant):
    # Use restaurant id as seed so each restaurant gets a unique, consistent photo
    seed = str(restaurant.get("id", "x")).replace("/", "-")
    return f"https://picsum.photos/seed/{seed}/400/300"

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    from flask import make_response
    resp = make_response(render_template("index.html"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

@app.route("/api/restaurants")
def get_restaurants():
    restaurants = load_restaurants()
    reviews     = load_reviews()
    restaurants = enrich_with_reviews(restaurants, reviews)

    # Add photos
    for r in restaurants:
        r["photo"] = get_photo(r)

    cuisine    = request.args.get("cuisine",    "").strip().lower()
    search     = request.args.get("search",     "").strip().lower()
    min_rating = float(request.args.get("min_rating", 0))
    wifi       = request.args.get("wifi",       "false") == "true"
    vegetarian = request.args.get("vegetarian", "false") == "true"
    outdoor    = request.args.get("outdoor",    "false") == "true"

    filtered = []
    for r in restaurants:
        if cuisine and cuisine != "all":
            cuisines_lower = [c.lower() for c in r.get("cuisine", [])]
            primary_lower  = (r.get("primary_cuisine") or "").lower()
            if not any(cuisine == c for c in cuisines_lower) and cuisine != primary_lower:
                continue
        if search:
            name   = (r.get("name")        or "").lower()
            addr   = (r.get("address")     or "").lower()
            native = (r.get("native_name") or "").lower()
            if search not in name and search not in addr and search not in native:
                continue
        display_rating = r.get("user_rating") or r.get("rating", 0)
        if display_rating < min_rating:
            continue
        if wifi       and not r.get("wifi"):            continue
        if vegetarian and not r.get("vegetarian"):      continue
        if outdoor    and not r.get("outdoor_seating"): continue
        filtered.append(r)

    return jsonify({"total": len(filtered), "restaurants": filtered})

@app.route("/api/cuisines")
def get_cuisines():
    restaurants = load_restaurants()
    cuisine_set = set()
    for r in restaurants:
        for c in r.get("cuisine", []):
            if c and c != "Unknown":
                cuisine_set.add(c)
    return jsonify(sorted(cuisine_set))

@app.route("/api/reviews", methods=["POST"])
def post_review():
    data    = request.json
    rest_id = data.get("restaurant_id")
    name    = data.get("name", "Anonymous").strip() or "Anonymous"
    rating  = int(data.get("rating", 3))
    comment = data.get("comment", "").strip()

    if not rest_id or not comment:
        return jsonify({"error": "Missing fields"}), 400
    if not (1 <= rating <= 5):
        return jsonify({"error": "Rating must be 1-5"}), 400

    reviews = load_reviews()
    if rest_id not in reviews:
        reviews[rest_id] = []

    reviews[rest_id].append({
        "name":    name,
        "rating":  rating,
        "comment": comment,
        "date":    datetime.now().strftime("%b %d, %Y"),
    })
    save_reviews(reviews)
    return jsonify({"success": True, "total": len(reviews[rest_id])})

@app.route("/api/analytics")
def analytics():
    restaurants = load_restaurants()
    reviews     = load_reviews()
    restaurants = enrich_with_reviews(restaurants, reviews)

    # Cuisine breakdown
    from collections import Counter
    cuisine_count = Counter()
    for r in restaurants:
        cuisine_count[r["primary_cuisine"]] += 1

    # Top rated
    top = sorted(restaurants, key=lambda x: x.get("user_rating") or x["rating"], reverse=True)[:5]

    # Amenity stats
    total = len(restaurants)
    amenities = {
        "WiFi":          sum(1 for r in restaurants if r.get("wifi")),
        "Vegetarian":    sum(1 for r in restaurants if r.get("vegetarian")),
        "Outdoor":       sum(1 for r in restaurants if r.get("outdoor_seating")),
        "Takeaway":      sum(1 for r in restaurants if r.get("takeaway")),
    }

    # Total reviews
    total_reviews = sum(len(v) for v in reviews.values())

    return jsonify({
        "total_restaurants": total,
        "total_reviews":     total_reviews,
        "cuisine_breakdown": dict(cuisine_count.most_common(10)),
        "top_rated":         [{"name": r["name"], "rating": r.get("user_rating") or r["rating"],
                                "cuisine": r["primary_cuisine"]} for r in top],
        "amenities":         amenities,
    })

# ── Machine Learning Routes ───────────────────────────────────────────────────

@app.route("/api/ml/similar/<path:restaurant_id>")
def ml_similar(restaurant_id):
    """Content-based: restaurants similar to a given one (cosine similarity)."""
    top_n = int(request.args.get("n", 5))
    results = recommender.similar_restaurants(restaurant_id, top_n=top_n)
    return jsonify({"restaurant_id": restaurant_id, "similar": results})

@app.route("/api/ml/recommend")
def ml_recommend():
    """
    Personalized recommendations based on a user's liked restaurants.
    user_id is a simple browser-generated identifier stored in localStorage,
    OR a comma-separated list of restaurant ids can be passed directly via ?liked=.
    """
    user_id = request.args.get("user_id", "").strip()
    top_n = int(request.args.get("n", 8))

    liked_param = request.args.get("liked", "")
    if liked_param:
        liked_ids = [x for x in liked_param.split(",") if x]
    elif user_id:
        likes = load_likes()
        liked_ids = likes.get(user_id, [])
    else:
        liked_ids = []

    results = recommender.recommend_for_user(liked_ids, top_n=top_n)
    return jsonify({
        "user_id": user_id,
        "based_on": liked_ids,
        "recommendations": results,
        "cold_start": len(liked_ids) == 0,
    })

@app.route("/api/ml/like", methods=["POST"])
def ml_like():
    """Record that a user liked a restaurant (used to build their profile vector)."""
    data = request.json or {}
    user_id = data.get("user_id", "").strip()
    restaurant_id = data.get("restaurant_id", "").strip()
    if not user_id or not restaurant_id:
        return jsonify({"error": "user_id and restaurant_id required"}), 400

    likes = load_likes()
    user_likes = likes.setdefault(user_id, [])
    if restaurant_id not in user_likes:
        user_likes.append(restaurant_id)
    save_likes(likes)
    return jsonify({"success": True, "liked_count": len(user_likes)})

@app.route("/api/ml/unlike", methods=["POST"])
def ml_unlike():
    data = request.json or {}
    user_id = data.get("user_id", "").strip()
    restaurant_id = data.get("restaurant_id", "").strip()
    likes = load_likes()
    if user_id in likes and restaurant_id in likes[user_id]:
        likes[user_id].remove(restaurant_id)
        save_likes(likes)
    return jsonify({"success": True, "liked_count": len(likes.get(user_id, []))})

@app.route("/api/ml/likes/<user_id>")
def ml_get_likes(user_id):
    likes = load_likes()
    return jsonify({"user_id": user_id, "liked": likes.get(user_id, [])})

@app.route("/api/ml/predict_rating/<path:restaurant_id>")
def ml_predict_rating(restaurant_id):
    """Supervised regression: predict a restaurant's rating from its features."""
    result = recommender.predict_rating(restaurant_id)
    if result is None:
        return jsonify({"error": "Restaurant not found"}), 404
    return jsonify(result)

@app.route("/api/ml/predict_all")
def ml_predict_all():
    return jsonify({"predictions": recommender.predict_all_ratings()})

@app.route("/api/ml/clusters")
def ml_clusters():
    """Unsupervised K-Means clustering of restaurants by feature similarity."""
    return jsonify({"clusters": recommender.get_clusters()})

@app.route("/api/ml/report")
def ml_report():
    """Full model report — algorithm details, accuracy metrics, feature importances.
    Useful for a project writeup / ML dashboard."""
    return jsonify(recommender.model_report())


if __name__ == "__main__":
    app.run(debug=True)