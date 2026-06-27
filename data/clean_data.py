import json
import re
import random

# ── Load raw GeoJSON ──────────────────────────────────────────────────────────
with open("Resturant_data_for_thamel.geojson", "r", encoding="utf-8") as f:
    raw = json.load(f)

features = raw["features"]
print(f"Total raw records: {len(features)}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_name(name):
    """Use English name if available, else keep original; strip extra spaces."""
    if not name:
        return None
    return name.strip()

def pick_display_name(props):
    """Prefer name:en > name (if ASCII-majority) > name:en fallback."""
    en = props.get("name:en", "").strip()
    native = props.get("name", "").strip()
    if en:
        return en
    # If native name is mostly ASCII, keep it
    ascii_ratio = sum(1 for c in native if ord(c) < 128) / max(len(native), 1)
    if ascii_ratio > 0.6:
        return native
    return native  # keep native as fallback (display in UI)

def clean_cuisine(raw_cuisine):
    """
    Split semicolon-separated cuisines, normalise each tag, remove noise tags,
    and return a clean list + a short primary label.
    """
    if not raw_cuisine:
        return [], "Unknown"

    # Tags that are not really cuisine types
    noise = {
        "lunch", "dinner", "breakfast", "coffee_shop", "black_tea", "tea",
        "wine", "beer", "diner", "local", "multi_cusine", "international",
        "yes", ""
    }

    rename = {
        "regional":      "Nepali",
        "nepalese":      "Nepali",
        "nepali":        "Nepali",
        "asian":         "Asian",
        "italian":       "Italian",
        "italian_pizza": "Italian / Pizza",
        "pizza":         "Pizza",
        "chinese":       "Chinese",
        "japanese":      "Japanese",
        "indian":        "Indian",
        "korean":        "Korean",
        "north-korean":  "Korean",
        "vietnamese":    "Vietnamese",
        "sushi":         "Sushi",
        "burger":        "Burger",
        "sandwich":      "Sandwich",
        "grill":         "Grill",
        "steak_house":   "Steak House",
        "fish_and_chips":"Fish & Chips",
        "trout":         "Seafood",
        "noodles":       "Noodles",
        "momo":          "Momo",
        "laphing":       "Nepali",
        "keema_noodles": "Noodles",
        "chowmein":      "Noodles",
        "curry":         "Curry",
        "bagel":         "Cafe / Bakery",
        "czech":         "Czech",
        "chicken":       "Chicken",
    }

    parts = [p.strip().lower() for p in raw_cuisine.split(";")]
    cleaned = []
    seen = set()
    for p in parts:
        if p in noise:
            continue
        label = rename.get(p, p.replace("_", " ").title())
        if label not in seen:
            cleaned.append(label)
            seen.add(label)

    if not cleaned:
        return [], "Unknown"

    primary = cleaned[0]
    return cleaned, primary

def clean_phone(phone):
    """Normalise phone to +977 format where possible."""
    if not phone:
        return None
    phone = phone.strip()
    # Remove spaces inside numbers
    digits_only = re.sub(r"[^\d+]", "", phone)
    if digits_only.startswith("+977"):
        return phone.strip()
    if len(digits_only) == 10 and digits_only.startswith("98"):
        return "+977 " + digits_only
    if len(digits_only) == 7:          # landline without area code
        return "+977 1-" + digits_only
    return phone  # return as-is if we can't normalise

def clean_opening_hours(oh):
    """Return as-is but strip weird whitespace; mark missing as None."""
    if not oh:
        return None
    return oh.strip()

def assign_rating(props, cuisines):
    """
    Assign a realistic synthetic rating (3.0–5.0).
    More info → slightly higher rating to reward complete data.
    """
    score = 3.5  # base

    # Reward completeness
    if props.get("phone"):        score += 0.1
    if props.get("website") or props.get("contact:website"): score += 0.15
    if props.get("opening_hours"): score += 0.1
    if props.get("email"):         score += 0.05
    if props.get("description"):   score += 0.2
    if props.get("outdoor_seating") == "yes": score += 0.1
    if props.get("internet_access") == "wlan": score += 0.05
    if props.get("diet:vegetarian"):  score += 0.05

    # Small cuisine-based nudge (purely illustrative)
    boosts = {"Italian": 0.2, "Italian / Pizza": 0.2, "Sushi": 0.2,
              "Japanese": 0.15, "Nepali": 0.1}
    score += boosts.get(cuisines[0] if cuisines else "", 0)

    # Add controlled noise so not all scores look identical
    random.seed(hash(props.get("@id", "")))   # deterministic per restaurant
    noise = random.uniform(-0.3, 0.3)
    score += noise

    return round(min(max(score, 3.0), 5.0), 1)

def remove_duplicate_coords(records):
    """Keep only the first restaurant at each coordinate pair."""
    seen_coords = {}
    unique = []
    for r in records:
        key = (round(r["longitude"], 6), round(r["latitude"], 6))
        if key not in seen_coords:
            seen_coords[key] = True
            unique.append(r)
        else:
            print(f"  [DUPLICATE] Removed: {r['name']} at {key}")
    return unique

def remove_duplicate_names(records):
    """Where two records share a normalised English name, keep the richer one."""
    seen_names = {}
    for r in records:
        key = r["name"].lower().strip()
        if key not in seen_names:
            seen_names[key] = r
        else:
            # Keep whichever has more filled fields
            existing = seen_names[key]
            if sum(1 for v in r.values() if v) > sum(1 for v in existing.values() if v):
                print(f"  [DUPLICATE NAME] Replacing: {existing['name']} → {r['name']}")
                seen_names[key] = r
    return list(seen_names.values())

# ── Main cleaning loop ────────────────────────────────────────────────────────
cleaned_records = []

for feat in features:
    props = feat.get("properties", {})
    geom  = feat.get("geometry", {})
    coords = geom.get("coordinates", [None, None])

    # 1. Skip if no valid coordinates
    if not coords or coords[0] is None or coords[1] is None:
        print(f"  [SKIP] No coordinates: {props.get('name', 'Unknown')}")
        continue

    lng, lat = coords[0], coords[1]

    # 2. Skip if not actually a restaurant
    if props.get("amenity") != "restaurant":
        print(f"  [SKIP] Not a restaurant: {props.get('name', 'Unknown')}")
        continue

    # 3. Name
    display_name = pick_display_name(props)
    if not display_name:
        print(f"  [SKIP] No name at ({lng}, {lat})")
        continue

    # 4. Cuisine
    cuisines, primary_cuisine = clean_cuisine(props.get("cuisine"))

    # 5. Phone
    phone = clean_phone(props.get("phone") or props.get("contact:phone"))

    # 6. Website
    website = (props.get("website") or
               props.get("contact:website") or
               props.get("contact:facebook") or None)

    # 7. Opening hours
    opening_hours = clean_opening_hours(props.get("opening_hours"))

    # 8. Address
    addr_parts = [
        props.get("addr:housenumber"),
        props.get("addr:street"),
        props.get("addr:city"),
    ]
    address = ", ".join(p for p in addr_parts if p) or None

    # 9. Extra amenities
    outdoor_seating = props.get("outdoor_seating") == "yes"
    wifi            = props.get("internet_access") == "wlan"
    vegetarian      = props.get("diet:vegetarian") in ("yes", "only")
    takeaway        = props.get("takeaway") == "yes"

    # 10. Rating (synthetic, deterministic)
    rating = assign_rating(props, cuisines)

    record = {
        "id":              feat.get("id", props.get("@id")),
        "name":            display_name,
        "native_name":     props.get("name", "").strip() or None,
        "cuisine":         cuisines,
        "primary_cuisine": primary_cuisine,
        "latitude":        round(lat, 7),
        "longitude":       round(lng, 7),
        "rating":          rating,
        "phone":           phone,
        "email":           props.get("email") or None,
        "website":         website,
        "opening_hours":   opening_hours,
        "address":         address,
        "outdoor_seating": outdoor_seating,
        "wifi":            wifi,
        "vegetarian":      vegetarian,
        "takeaway":        takeaway,
        "description":     props.get("description", "").strip() or None,
    }
    cleaned_records.append(record)

print(f"\nAfter basic cleaning:  {len(cleaned_records)} records")

# ── Deduplication ─────────────────────────────────────────────────────────────
cleaned_records = remove_duplicate_coords(cleaned_records)
print(f"After coord dedup:     {len(cleaned_records)} records")

cleaned_records = remove_duplicate_names(cleaned_records)
print(f"After name dedup:      {len(cleaned_records)} records")

# ── Sort by rating descending ─────────────────────────────────────────────────
cleaned_records.sort(key=lambda r: r["rating"], reverse=True)

# ── Save cleaned JSON ─────────────────────────────────────────────────────────
output = {
    "total": len(cleaned_records),
    "source": "OpenStreetMap / Overpass Turbo – Thamel, Kathmandu",
    "restaurants": cleaned_records
}

with open("restaurants_clean.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n✅ Saved {len(cleaned_records)} clean restaurants → restaurants_clean.json")

# ── Quick summary ─────────────────────────────────────────────────────────────
from collections import Counter
cuisine_count = Counter()
for r in cleaned_records:
    cuisine_count[r["primary_cuisine"]] += 1

print("\n📊 Cuisine breakdown:")
for cuisine, count in cuisine_count.most_common():
    print(f"   {cuisine:<25} {count}")

print(f"\n⭐ Rating range: {min(r['rating'] for r in cleaned_records)} – {max(r['rating'] for r in cleaned_records)}")
print(f"📶 Restaurants with WiFi:     {sum(1 for r in cleaned_records if r['wifi'])}")
print(f"🌿 Vegetarian-friendly:       {sum(1 for r in cleaned_records if r['vegetarian'])}")
print(f"🌳 Outdoor seating:           {sum(1 for r in cleaned_records if r['outdoor_seating'])}")