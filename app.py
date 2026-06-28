from flask import Flask, render_template, jsonify, request
import json
import os
from datetime import datetime

app = Flask(__name__)

DATA_PATH    = os.path.join(os.path.dirname(__file__), "data", "restaurants_clean.json")
REVIEWS_PATH = os.path.join(os.path.dirname(__file__), "data", "reviews.json")

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
CUISINE_PHOTOS = {
    "Nepali":        "https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=400&q=80",
    "Indian":        "https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=400&q=80",
    "Chinese":       "https://images.unsplash.com/photo-1563245372-f21724e3856d?w=400&q=80",
    "Japanese":      "https://images.unsplash.com/photo-1580822184713-fc5400e7fe10?w=400&q=80",
    "Sushi":         "https://images.unsplash.com/photo-1553621042-f6e147245754?w=400&q=80",
    "Italian":       "https://images.unsplash.com/photo-1555396273-367ea4eb4db5?w=400&q=80",
    "Italian / Pizza":"https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=400&q=80",
    "Pizza":         "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=400&q=80",
    "Burger":        "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=400&q=80",
    "Korean":        "https://images.unsplash.com/photo-1590301157890-4810ed352733?w=400&q=80",
    "Thai":          "https://images.unsplash.com/photo-1559314809-0d155014e29e?w=400&q=80",
    "Vietnamese":    "https://images.unsplash.com/photo-1582878826629-29b7ad1cdc43?w=400&q=80",
    "Steak House":   "https://images.unsplash.com/photo-1558030006-450675393462?w=400&q=80",
    "American":      "https://images.unsplash.com/photo-1550547660-d9450f859349?w=400&q=80",
    "Cafe / Bakery": "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?w=400&q=80",
    "Momo":          "https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=400&q=80",
    "default":       "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=400&q=80",
}

def get_photo(restaurant):
    for cuisine in restaurant.get("cuisine", []):
        if cuisine in CUISINE_PHOTOS:
            return CUISINE_PHOTOS[cuisine]
    return CUISINE_PHOTOS["default"]

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
            if not any(cuisine in c for c in cuisines_lower):
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

if __name__ == "__main__":
    app.run(debug=True)