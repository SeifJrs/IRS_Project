"""
Collaborative Filtering Recommender — implements 4 methods:
  1. User-Based CF (Cosine Similarity)
  2. User-Based CF (Pearson Correlation)
  3. Item-Based CF (Cosine Similarity)
  4. Matrix Factorization (SVD)
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeFiltering:
    """Collaborative Filtering engine with 4 methods."""

    METHOD_NAMES = {
        "user_cosine":  "User-Based CF (Cosine Similarity)",
        "user_pearson": "User-Based CF (Pearson Correlation)",
        "item_cosine":  "Item-Based CF (Cosine Similarity)",
        "svd":          "Matrix Factorization (SVD)",
    }

    def __init__(self, ratings_df, products_df):
        self.ratings_df = ratings_df
        self.products_df = products_df
        self.user_ids = sorted(ratings_df["user_id"].unique())
        self.product_ids = sorted(ratings_df["product_id"].unique())

        # Build user-item matrix
        self.user_item_matrix = ratings_df.pivot_table(
            index="user_id", columns="product_id", values="rating"
        ).reindex(index=self.user_ids, columns=self.product_ids)

        self.matrix_filled = self.user_item_matrix.fillna(0).values
        self.global_mean = ratings_df["rating"].mean()

        # Precompute
        self._user_cosine_sim = None
        self._user_pearson_sim = None
        self._item_cosine_sim = None
        self._svd_predictions = None

    # ── Similarity computations ───────────────────────────────────

    def _get_user_cosine_sim(self):
        if self._user_cosine_sim is None:
            self._user_cosine_sim = cosine_similarity(self.matrix_filled)
        return self._user_cosine_sim

    def _get_user_pearson_sim(self):
        if self._user_pearson_sim is None:
            # Pearson = cosine similarity on mean-centered vectors
            mean_centered = self.matrix_filled - self.matrix_filled.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(mean_centered, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            normalized = mean_centered / norms
            self._user_pearson_sim = normalized @ normalized.T
        return self._user_pearson_sim

    def _get_item_cosine_sim(self):
        if self._item_cosine_sim is None:
            self._item_cosine_sim = cosine_similarity(self.matrix_filled.T)
        return self._item_cosine_sim

    def _compute_svd(self, n_factors=50):
        if self._svd_predictions is None:
            matrix = self.matrix_filled.copy()
            # Mean-center
            user_means = matrix.mean(axis=1, keepdims=True)
            centered = matrix - user_means
            n_factors = min(n_factors, min(centered.shape) - 1)
            U, sigma, Vt = np.linalg.svd(centered, full_matrices=False)
            U = U[:, :n_factors]
            sigma = sigma[:n_factors]
            Vt = Vt[:n_factors, :]
            self._svd_predictions = user_means + U @ np.diag(sigma) @ Vt
        return self._svd_predictions

    # ── Prediction methods ────────────────────────────────────────

    def _predict_user_based(self, user_id, sim_matrix, top_k_users=30):
        """Predict ratings for a user using user-based CF."""
        if user_id not in self.user_ids:
            return {}
        u_idx = self.user_ids.index(user_id)
        sims = sim_matrix[u_idx]
        # Get top-k similar users (excluding self)
        neighbor_indices = np.argsort(sims)[::-1][1:top_k_users + 1]

        predictions = {}
        for p_idx, pid in enumerate(self.product_ids):
            if not np.isnan(self.user_item_matrix.iloc[u_idx, p_idx]):
                continue  # Already rated
            num, den = 0.0, 0.0
            for n_idx in neighbor_indices:
                if not np.isnan(self.user_item_matrix.iloc[n_idx, p_idx]):
                    num += sims[n_idx] * self.user_item_matrix.iloc[n_idx, p_idx]
                    den += abs(sims[n_idx])
            if den > 0:
                predictions[pid] = num / den
        return predictions

    def _predict_item_based(self, user_id, top_k_items=30):
        """Predict ratings for a user using item-based CF."""
        if user_id not in self.user_ids:
            return {}
        u_idx = self.user_ids.index(user_id)
        item_sim = self._get_item_cosine_sim()

        predictions = {}
        rated_items = []
        for p_idx, pid in enumerate(self.product_ids):
            if not np.isnan(self.user_item_matrix.iloc[u_idx, p_idx]):
                rated_items.append((p_idx, self.user_item_matrix.iloc[u_idx, p_idx]))

        for p_idx, pid in enumerate(self.product_ids):
            if not np.isnan(self.user_item_matrix.iloc[u_idx, p_idx]):
                continue
            # Find most similar items among rated items
            sims_scores = [(item_sim[p_idx, ri], r) for ri, r in rated_items]
            sims_scores.sort(key=lambda x: x[0], reverse=True)
            top = sims_scores[:top_k_items]
            num = sum(s * r for s, r in top)
            den = sum(abs(s) for s, _ in top)
            if den > 0:
                predictions[pid] = num / den
        return predictions

    def _predict_svd(self, user_id):
        """Predict ratings using SVD matrix factorization."""
        if user_id not in self.user_ids:
            return {}
        u_idx = self.user_ids.index(user_id)
        preds = self._compute_svd()

        predictions = {}
        for p_idx, pid in enumerate(self.product_ids):
            if not np.isnan(self.user_item_matrix.iloc[u_idx, p_idx]):
                continue
            predictions[pid] = float(np.clip(preds[u_idx, p_idx], 1.0, 5.0))
        return predictions

    # ── Public API ────────────────────────────────────────────────

    def predict_all(self, user_id, method="user_cosine"):
        """Return predicted ratings dict {product_id: predicted_rating} for unrated items."""
        if method == "user_cosine":
            return self._predict_user_based(user_id, self._get_user_cosine_sim())
        elif method == "user_pearson":
            return self._predict_user_based(user_id, self._get_user_pearson_sim())
        elif method == "item_cosine":
            return self._predict_item_based(user_id)
        elif method == "svd":
            return self._predict_svd(user_id)
        else:
            raise ValueError(f"Unknown method: {method}")

    def recommend(self, user_id, method="user_cosine", top_n=10):
        """Return top-N recommendations as a DataFrame with explanations."""
        predictions = self.predict_all(user_id, method)
        if not predictions:
            return pd.DataFrame()

        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]
        rec_pids = [pid for pid, _ in sorted_preds]
        rec_scores = [score for _, score in sorted_preds]

        result = self.products_df[self.products_df["product_id"].isin(rec_pids)].copy()
        score_map = dict(zip(rec_pids, rec_scores))
        result["predicted_rating"] = result["product_id"].map(score_map)
        result = result.sort_values("predicted_rating", ascending=False).reset_index(drop=True)

        # Add explanations
        method_name = self.METHOD_NAMES.get(method, method)
        explanations = []
        for _, row in result.iterrows():
            if method in ("user_cosine", "user_pearson"):
                explanations.append(
                    f"🤝 Recommended because users with similar rating patterns "
                    f"also liked this {row['category'].lower()} product "
                    f"(predicted rating: {row['predicted_rating']:.1f}/5) — via {method_name}"
                )
            elif method == "item_cosine":
                explanations.append(
                    f"📦 Recommended because this item is similar to products you've rated highly "
                    f"(predicted rating: {row['predicted_rating']:.1f}/5) — via {method_name}"
                )
            else:
                explanations.append(
                    f"🧮 Recommended through matrix factorization of rating patterns "
                    f"(predicted rating: {row['predicted_rating']:.1f}/5) — via {method_name}"
                )
        result["explanation"] = explanations
        return result

    def get_user_ratings(self, user_id):
        """Return products already rated by this user."""
        user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]
        if user_ratings.empty:
            return pd.DataFrame()
        return user_ratings.merge(self.products_df, on="product_id")
