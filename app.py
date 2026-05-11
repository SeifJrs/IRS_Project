"""
Intelligent E-Commerce Recommendation System — Flask App
AIE425 Course Project

Features:
  - 3 recommendation approaches: CF, Content-Based, Knowledge-Based
  - 4 CF methods: User-Based Cosine, User-Based Pearson, Item-Based, SVD
  - 4 evaluation metrics: Precision@K, Recall@K, RMSE, NDCG@K
  - Explainability for every recommendation
  - Comprehensive comparison dashboard
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

from data.load_data import load_data, get_train_test_split
from recommenders.collaborative import CollaborativeFiltering
from recommenders.content_based import ContentBasedRecommender
from recommenders.knowledge_based import KnowledgeBasedRecommender
from evaluation.metrics import evaluate_recommender, compare_methods, generate_analysis

app = Flask(__name__, template_folder='templates', static_folder='static')

# ─── Data Loading ─────────────────────────────────────────────────────
print("Loading data...")
products_df, ratings_df, user_prefs = load_data()
train_df, test_df = get_train_test_split(ratings_df)

print("Initializing recommenders...")
cf_engine = CollaborativeFiltering(train_df, products_df)
cb_engine = ContentBasedRecommender(train_df, products_df)
kb_engine = KnowledgeBasedRecommender(ratings_df, products_df)
print("Ready!")

# Cache for evaluation results
eval_cache = {}


# ═══════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    cat_counts = products_df["category"].value_counts()
    cat_data = {"labels": cat_counts.index.tolist(), "values": cat_counts.values.tolist()}

    rating_counts = ratings_df["rating"].value_counts().sort_index()
    rating_data = {"labels": [str(r) for r in rating_counts.index], "values": rating_counts.values.tolist()}

    min_price, max_price = kb_engine.get_price_range()

    return render_template(
        'index.html',
        user_ids=sorted(ratings_df["user_id"].unique().tolist()),
        n_products=len(products_df),
        n_users=ratings_df["user_id"].nunique(),
        n_ratings=f"{len(ratings_df):,}",
        n_ratings_raw=len(ratings_df),
        avg_rating=f"{ratings_df['rating'].mean():.2f}",
        cat_data=cat_data,
        rating_data=rating_data,
        categories=kb_engine.get_available_categories(),
        brands=kb_engine.get_available_brands(),
        min_price=int(min_price),
        max_price=int(max_price),
    )


@app.route("/api/user_info")
def user_info():
    uid = int(request.args.get("user_id", 1))
    n_ratings = len(ratings_df[ratings_df["user_id"] == uid])
    prefs = ", ".join(user_prefs.get(uid, ["N/A"]))
    return jsonify({"n_ratings": n_ratings, "prefs": prefs})


@app.route("/api/recommend/cf")
def recommend_cf():
    uid = int(request.args.get("user_id", 1))
    method = request.args.get("method", "user_cosine")
    top_n = int(request.args.get("top_n", 10))
    recs = cf_engine.recommend(uid, method=method, top_n=top_n)
    return jsonify({"recommendations": recs.to_dict("records") if not recs.empty else []})


@app.route("/api/compare_cf")
def compare_cf():
    uid = int(request.args.get("user_id", 1))
    result = {}
    for method_key, method_name in CollaborativeFiltering.METHOD_NAMES.items():
        recs = cf_engine.recommend(uid, method=method_key, top_n=5)
        if not recs.empty:
            result[method_name] = recs[["product_name", "category", "predicted_rating"]].to_dict("records")
        else:
            result[method_name] = []
    return jsonify(result)


@app.route("/api/recommend/cb")
def recommend_cb():
    uid = int(request.args.get("user_id", 1))
    top_n = int(request.args.get("top_n", 10))
    recs = cb_engine.recommend(uid, top_n=top_n)
    return jsonify({"recommendations": recs.to_dict("records") if not recs.empty else []})


@app.route("/api/recommend/kb")
def recommend_kb():
    uid = int(request.args.get("user_id", 1))
    top_n = int(request.args.get("top_n", 10))
    category = request.args.get("category")
    brand = request.args.get("brand")
    min_price = request.args.get("min_price", type=float)
    max_price = request.args.get("max_price", type=float)
    min_rating = request.args.get("min_rating", type=float)

    recs = kb_engine.recommend(
        user_id=uid,
        category=category if category else None,
        brand=brand if brand else None,
        min_price=min_price,
        max_price=max_price,
        min_rating=min_rating,
        top_n=top_n,
    )
    return jsonify({"recommendations": recs.to_dict("records") if not recs.empty else []})


@app.route("/api/evaluate")
def evaluate():
    k = int(request.args.get("k", 10))

    results = {}

    # CF methods
    for method_key, method_name in CollaborativeFiltering.METHOD_NAMES.items():
        def cf_rec(uid, m=method_key):
            return cf_engine.recommend(uid, method=m, top_n=k)
        res = evaluate_recommender(cf_rec, test_df, ratings_df, k=k)
        results[method_name] = res

    # Content-Based
    def cb_rec(uid):
        return cb_engine.recommend(uid, top_n=k)
    results["Content-Based (TF-IDF)"] = evaluate_recommender(cb_rec, test_df, ratings_df, k=k)

    # Knowledge-Based
    def kb_rec(uid):
        return kb_engine.recommend(user_id=uid, top_n=k)
    results["Knowledge-Based"] = evaluate_recommender(kb_rec, test_df, ratings_df, k=k)

    results_df = compare_methods(results)
    analysis = generate_analysis(results_df)

    # Convert NaN to None for JSON
    results_list = results_df.replace({np.nan: None}).to_dict("records")

    return jsonify({"results": results_list, "analysis": analysis})


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Intelligent Recommender System - AIE425")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60 + "\n")
    app.run(debug=False, port=5000, host="0.0.0.0")
