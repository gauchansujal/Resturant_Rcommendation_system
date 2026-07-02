import os
import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed, server-side rendering
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import sys
sys.path.insert(0, os.path.dirname(__file__))
from recommender import RestaurantRecommender

HERE = os.path.dirname(__file__)
CHARTS_DIR = os.path.join(HERE, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# Brand palette to match the Thamel Eats app
RED      = "#E8431A"
RED_DARK = "#C13510"
INK      = "#1A1209"
MUTED    = "#7A6A58"
CREAM    = "#FDF6EE"
PALETTE  = ["#E8431A", "#F59E0B", "#1A1209", "#7A6A58", "#C13510",
            "#E8DDD2", "#F3EDE7", "#8B5A2B", "#D97706", "#92400E"]


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _save_and_encode(fig, filename):
    path = os.path.join(CHARTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return path, encoded


# ────────────────────────────────────────────────────────────────
# 1. PIE CHART — cuisine distribution
# ────────────────────────────────────────────────────────────────
def make_pie_chart(rec: RestaurantRecommender):
    counts = rec.df["primary_cuisine"].value_counts()
    top = counts.head(8)
    other = counts.iloc[8:].sum()
    if other > 0:
        top["Other"] = other

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=CREAM)
    ax.set_facecolor(CREAM)
    wedges, texts, autotexts = ax.pie(
        top.values, labels=top.index, autopct="%1.0f%%",
        colors=PALETTE, startangle=90,
        wedgeprops={"edgecolor": CREAM, "linewidth": 2},
        textprops={"fontsize": 9, "color": INK},
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
        at.set_fontsize(8)
    ax.set_title("Cuisine Distribution — Thamel Restaurants", fontsize=13,
                 fontweight="bold", color=INK, pad=15)
    return _save_and_encode(fig, "pie_cuisine_distribution.png")


# ────────────────────────────────────────────────────────────────
# 2. LINE GRAPH — predicted vs actual rating, sorted by actual rating
# ────────────────────────────────────────────────────────────────
def make_line_chart(rec: RestaurantRecommender):
    preds = rec.predict_all_ratings()
    preds_sorted = sorted(preds, key=lambda p: p["actual_rating"])
    x = range(len(preds_sorted))
    actual = [p["actual_rating"] for p in preds_sorted]
    predicted = [p["predicted_rating"] for p in preds_sorted]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=CREAM)
    ax.set_facecolor(CREAM)
    ax.plot(x, actual, color=INK, linewidth=2, label="Actual rating", marker="o", markersize=3)
    ax.plot(x, predicted, color=RED, linewidth=2, label="Predicted rating (ML)",
            marker="o", markersize=3, alpha=0.85)
    ax.fill_between(x, actual, predicted, color=RED, alpha=0.08)
    ax.set_xlabel("Restaurants (sorted by actual rating)", fontsize=10, color=INK)
    ax.set_ylabel("Rating (stars)", fontsize=10, color=INK)
    ax.set_title("Predicted vs Actual Rating Across All Restaurants", fontsize=13,
                 fontweight="bold", color=INK, pad=12)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", alpha=0.2)
    return _save_and_encode(fig, "line_predicted_vs_actual.png")


# ────────────────────────────────────────────────────────────────
# 3. SCATTER CHART — predicted vs actual (correlation view)
# ────────────────────────────────────────────────────────────────
def make_scatter_chart(rec: RestaurantRecommender):
    preds = rec.predict_all_ratings()
    actual = np.array([p["actual_rating"] for p in preds])
    predicted = np.array([p["predicted_rating"] for p in preds])

    fig, ax = plt.subplots(figsize=(7, 6.5), facecolor=CREAM)
    ax.set_facecolor(CREAM)
    ax.scatter(actual, predicted, color=RED, alpha=0.7, s=60,
               edgecolors=INK, linewidth=0.5, zorder=3)

    # Perfect-prediction reference line (y = x)
    lims = [min(actual.min(), predicted.min()) - 0.1, max(actual.max(), predicted.max()) + 0.1]
    ax.plot(lims, lims, color=INK, linestyle="--", linewidth=1.3, alpha=0.6,
            label="Perfect prediction (y = x)", zorder=2)

    # Correlation coefficient annotation
    corr = float(np.corrcoef(actual, predicted)[0, 1])
    ax.text(0.05, 0.95, f"Correlation: r = {corr:.2f}\nMAE: ±{rec.rf_metrics['mae']} stars",
            transform=ax.transAxes, fontsize=9, color=INK, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=RED, alpha=0.9))

    ax.set_xlabel("Actual rating", fontsize=10, color=INK)
    ax.set_ylabel("Predicted rating (ML model)", fontsize=10, color=INK)
    ax.set_title("Model Accuracy: Predicted vs Actual Rating", fontsize=13,
                 fontweight="bold", color=INK, pad=12)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(alpha=0.2)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect("equal")
    return _save_and_encode(fig, "scatter_model_accuracy.png")


# ────────────────────────────────────────────────────────────────
# 4. BAR GRAPH — feature importance from Random Forest
# ────────────────────────────────────────────────────────────────
def make_bar_chart(rec: RestaurantRecommender):
    feats = rec.rf_feature_importance[:10]
    names = [f["feature"].replace("cuisine_", "") for f in feats][::-1]
    vals = [f["importance"] * 100 for f in feats][::-1]

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=CREAM)
    ax.set_facecolor(CREAM)
    bars = ax.barh(names, vals, color=RED, edgecolor=RED_DARK, linewidth=0.8)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.4, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
                va="center", fontsize=8.5, color=INK)
    ax.set_xlabel("Importance (%) in predicting rating", fontsize=10, color=INK)
    ax.set_title("What Drives a Restaurant's Rating? (Random Forest Feature Importance)",
                 fontsize=12.5, fontweight="bold", color=INK, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", alpha=0.2)
    return _save_and_encode(fig, "bar_feature_importance.png")


# ────────────────────────────────────────────────────────────────
# 5. GGPLOT-STYLE CHART — rating distribution histogram
#    (uses matplotlib's built-in 'ggplot' theme, mimicking R's
#    ggplot2 aesthetic: grey background, white gridlines, no axis box)
# ────────────────────────────────────────────────────────────────
def make_ggplot_chart(rec: RestaurantRecommender):
    with plt.style.context("ggplot"):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ratings = rec.df["rating"].values
        ax.hist(ratings, bins=12, color=RED, edgecolor="white", linewidth=1.2, alpha=0.85)
        mean_r = ratings.mean()
        ax.axvline(mean_r, color=INK, linestyle="--", linewidth=1.5,
                   label=f"Mean = {mean_r:.2f}★")
        ax.set_xlabel("Rating (stars)", fontsize=10)
        ax.set_ylabel("Number of restaurants", fontsize=10)
        ax.set_title("Distribution of Restaurant Ratings (ggplot style)",
                     fontsize=13, fontweight="bold", pad=12)
        ax.legend(fontsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        return _save_and_encode(fig, "ggplot_rating_distribution.png")


# ────────────────────────────────────────────────────────────────
# BONUS: Cluster scatter (2D projection of K-Means clusters)
# ────────────────────────────────────────────────────────────────
def make_cluster_scatter(rec: RestaurantRecommender):
    from sklearn.decomposition import PCA
    X = rec.feature_matrix.values
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    labels = rec.df["cluster"].values

    fig, ax = plt.subplots(figsize=(8, 6.5), facecolor=CREAM)
    ax.set_facecolor(CREAM)
    for c in sorted(set(labels)):
        mask = labels == c
        label_name = rec.cluster_labels[c]["label"]
        ax.scatter(coords[mask, 0], coords[mask, 1], s=70, alpha=0.75,
                   color=PALETTE[c % len(PALETTE)], edgecolors=INK, linewidth=0.4,
                   label=f"{label_name} (n={mask.sum()})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}% variance)", fontsize=10, color=INK)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}% variance)", fontsize=10, color=INK)
    ax.set_title("K-Means Clusters (PCA 2D Projection)", fontsize=13,
                 fontweight="bold", color=INK, pad=12)
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(alpha=0.2)
    return _save_and_encode(fig, "scatter_clusters_pca.png")


# ────────────────────────────────────────────────────────────────
# Master function: generate everything at once
# ────────────────────────────────────────────────────────────────
def generate_all_charts(data_path):
    rec = RestaurantRecommender(data_path)
    charts = {}
    for name, fn in [
        ("pie", make_pie_chart),
        ("line", make_line_chart),
        ("scatter", make_scatter_chart),
        ("bar", make_bar_chart),
        ("ggplot", make_ggplot_chart),
        ("clusters", make_cluster_scatter),
    ]:
        path, encoded = fn(rec)
        charts[name] = {"path": path, "base64": encoded}
        print(f"[chart] {name:<10} -> {path}")
    return charts


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    data_path = os.path.join(here, "..", "restaurants_clean.json")
    if not os.path.exists(data_path):
        data_path = os.path.join(here, "..", "data", "restaurants_clean.json")

    print("Generating charts from:", data_path)
    generate_all_charts(data_path)
    print(f"\nDone! Charts saved in: {CHARTS_DIR}")



    """
generate_charts.py
====================================================================
Generates 5 chart types from the REAL restaurant data and the
trained ML model in recommender.py:

  1. PIE CHART       — cuisine distribution
  2. LINE GRAPH       — rating prediction error across restaurants
                         (sorted by actual rating, predicted vs actual)
  3. SCATTER CHART    — predicted rating vs actual rating
                         (shows model accuracy visually)
  4. BAR GRAPH        — feature importance from the Random Forest
  5. GGPLOT-STYLE     — rating distribution histogram, styled with
                         matplotlib's 'ggplot' theme (R's ggplot2 look)

Every chart is saved as a PNG into ml/charts/ AND returned as a
base64 string (used by the /api/ml/chart/<name> Flask route to
serve them live in the browser without writing to disk).
====================================================================
"""
