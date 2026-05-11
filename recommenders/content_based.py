"""
Content-Based Recommender — uses TF-IDF on product descriptions/categories
to find items similar to a user's past preferences.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """Content-Based Recommendation using TF-IDF + Cosine Similarity."""

    def __init__(self, ratings_df, products_df):
        self.ratings_df = ratings_df
        self.products_df = products_df.copy()

        # Build content features by combining description + category + brand
        self.products_df["content_features"] = (
            self.products_df["description"].fillna("")
            + " " + self.products_df["category"].fillna("")
            + " " + self.products_df["brand"].fillna("")
        )

        # TF-IDF vectorization
        self.tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
        self.tfidf_matrix = self.tfidf.fit_transform(
            self.products_df["content_features"]
        )

        # Product ID to index mapping
        self.pid_to_idx = {
            pid: idx
            for idx, pid in enumerate(self.products_df["product_id"])
        }

    def _build_user_profile(self, user_id):
        """Build a user profile vector from their rated items (weighted by rating)."""
        user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]
        if user_ratings.empty:
            return None

        # Get TF-IDF vectors of rated items, weighted by rating
        profile = np.zeros(self.tfidf_matrix.shape[1])
        total_weight = 0
        for _, row in user_ratings.iterrows():
            pid = row["product_id"]
            if pid in self.pid_to_idx:
                idx = self.pid_to_idx[pid]
                weight = row["rating"]
                profile += weight * self.tfidf_matrix[idx].toarray().flatten()
                total_weight += weight

        if total_weight > 0:
            profile /= total_weight
        return profile.reshape(1, -1)

    def predict_all(self, user_id):
        """Return predicted scores dict {product_id: score} for unrated items."""
        user_profile = self._build_user_profile(user_id)
        if user_profile is None:
            return {}

        # Compute similarity between user profile and all items
        similarities = cosine_similarity(user_profile, self.tfidf_matrix).flatten()

        # Filter out already-rated items
        rated_pids = set(
            self.ratings_df[self.ratings_df["user_id"] == user_id]["product_id"]
        )

        predictions = {}
        for idx, pid in enumerate(self.products_df["product_id"]):
            if pid not in rated_pids:
                # Scale similarity to a 1-5 rating range
                predictions[pid] = 1.0 + 4.0 * similarities[idx]
        return predictions

    def recommend(self, user_id, top_n=10):
        """Return top-N content-based recommendations with explanations."""
        predictions = self.predict_all(user_id)
        if not predictions:
            return pd.DataFrame()

        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]
        rec_pids = [pid for pid, _ in sorted_preds]
        rec_scores = [score for _, score in sorted_preds]

        result = self.products_df[
            self.products_df["product_id"].isin(rec_pids)
        ].copy()
        score_map = dict(zip(rec_pids, rec_scores))
        result["predicted_rating"] = result["product_id"].map(score_map)
        result = result.sort_values("predicted_rating", ascending=False).reset_index(drop=True)

        # Build explanations
        user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]
        liked_cats = set()
        liked_brands = set()
        if not user_ratings.empty:
            high_rated = user_ratings[user_ratings["rating"] >= 4]
            rated_products = self.products_df[
                self.products_df["product_id"].isin(high_rated["product_id"])
            ]
            liked_cats = set(rated_products["category"].unique()[:3])
            liked_brands = set(rated_products["brand"].unique()[:3])

        explanations = []
        for _, row in result.iterrows():
            reasons = []
            if row["category"] in liked_cats:
                reasons.append(f"matches your preferred category '{row['category']}'")
            if row["brand"] in liked_brands:
                reasons.append(f"from a brand you enjoy ('{row['brand']}')")
            if not reasons:
                reasons.append("has similar features to products you've liked")

            explanation = (
                f"📝 Recommended because it {' and '.join(reasons)} "
                f"(content similarity score: {row['predicted_rating']:.2f}/5)"
            )
            explanations.append(explanation)

        result["explanation"] = explanations
        return result

    def get_similar_products(self, product_id, top_n=5):
        """Find products similar to a given product based on content."""
        if product_id not in self.pid_to_idx:
            return pd.DataFrame()
        idx = self.pid_to_idx[product_id]
        sims = cosine_similarity(
            self.tfidf_matrix[idx], self.tfidf_matrix
        ).flatten()
        sim_indices = sims.argsort()[::-1][1:top_n + 1]

        result = self.products_df.iloc[sim_indices].copy()
        result["similarity_score"] = sims[sim_indices]
        return result.reset_index(drop=True)
