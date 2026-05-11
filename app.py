"""
Intelligent E-Commerce Recommendation System — Flask API
AIE425 Course Project

Backend API serving the Next.js frontend.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

from data.load_data import load_data, get_train_test_split
from recommenders.collaborative import CollaborativeFiltering
from recommenders.content_based import ContentBasedRecommender
from recommenders.knowledge_based import KnowledgeBasedRecommender
from evaluation.metrics import evaluate_recommender, compare_methods, generate_analysis

app = Flask(__name__)

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
def home():
    return jsonify({
        "message": "Recommender System API is running",
        "version": "Next.js Integrated",
    })


@app.route("/api/user_info")
def user_info():
    uid = int(request.args.get("user_id", 1))
    n_ratings = len(ratings_df[ratings_df["user_id"] == uid])
    prefs = ", ".join(user_prefs.get(uid, ["N/A"]))
    return jsonify({"n_ratings": n_ratings, "prefs": prefs})


@app.route("/api/brands")
def get_brands():
    category = request.args.get("category")
    brands = kb_engine.get_available_brands(category if category else None)
    return jsonify({"brands": brands})


@app.route("/api/bootstrap")
def bootstrap():
    cat_counts = products_df["category"].value_counts()
    rating_counts = ratings_df["rating"].value_counts().sort_index()
    min_price, max_price = kb_engine.get_price_range()

    return jsonify({
        "user_ids": sorted(ratings_df["user_id"].unique().tolist()),
        "stats": {
            "products": len(products_df),
            "users": ratings_df["user_id"].nunique(),
            "ratings": len(ratings_df),
            "avg_rating": round(float(ratings_df["rating"].mean()), 2),
        },
        "category_chart": {
            "labels": cat_counts.index.tolist(),
            "values": cat_counts.values.tolist(),
        },
        "rating_chart": {
            "labels": [str(r) for r in rating_counts.index],
            "values": rating_counts.values.tolist(),
        },
        "categories": kb_engine.get_available_categories(),
        "brands": kb_engine.get_available_brands(),
        "price_range": {
            "min": int(min_price),
            "max": int(max_price),
        },
    })


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
    max_users = request.args.get("max_users", type=int)
    cache_key = f"k={k}|max_users={max_users or 'all'}"

    if cache_key in eval_cache:
        return jsonify(eval_cache[cache_key])

    total_test_users = int(test_df["user_id"].nunique())
    eval_test_df = test_df
    if max_users and max_users > 0:
        sampled_users = sorted(test_df["user_id"].unique().tolist())[:max_users]
        eval_test_df = test_df[test_df["user_id"].isin(sampled_users)].reset_index(drop=True)

    sample_size = int(eval_test_df["user_id"].nunique())

    results = {}

    # CF methods
    for method_key, method_name in CollaborativeFiltering.METHOD_NAMES.items():
        def cf_rec(uid, m=method_key):
            return cf_engine.recommend(uid, method=m, top_n=k)
        res = evaluate_recommender(cf_rec, eval_test_df, ratings_df, k=k)
        results[method_name] = res

    # Content-Based
    def cb_rec(uid):
        return cb_engine.recommend(uid, top_n=k)
    results["Content-Based (TF-IDF)"] = evaluate_recommender(cb_rec, eval_test_df, ratings_df, k=k)

    # Knowledge-Based
    def kb_rec(uid):
        return kb_engine.recommend(user_id=uid, top_n=k)
    results["Knowledge-Based"] = evaluate_recommender(kb_rec, eval_test_df, ratings_df, k=k)

    results_df = compare_methods(results)
    analysis = generate_analysis(results_df)

    # Convert NaN to None for JSON
    results_list = results_df.replace({np.nan: None}).to_dict("records")

    payload = {
        "results": results_list,
        "analysis": analysis,
        "sample_size": sample_size,
        "total_test_users": total_test_users,
    }
    eval_cache[cache_key] = payload
    return jsonify(payload)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Intelligent Recommender System API - AIE425")
    print("  API running on http://localhost:5000")
    print("  Frontend: cd frontend && npm run dev")
    print("=" * 60 + "\n")
    app.run(debug=False, port=5000, host="0.0.0.0")
