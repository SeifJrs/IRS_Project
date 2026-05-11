"""
Knowledge-Based Recommender — uses explicit user constraints
(budget, brand, category, minimum rating) to filter and rank products.
"""

import pandas as pd
import numpy as np


class KnowledgeBasedRecommender:
    """Knowledge-Based filtering using user-specified constraints."""

    def __init__(self, ratings_df, products_df):
        self.ratings_df = ratings_df
        self.products_df = products_df.copy()

        # Precompute product popularity stats
        product_stats = ratings_df.groupby("product_id").agg(
            avg_rating=("rating", "mean"),
            num_ratings=("rating", "count"),
        ).reset_index()
        self.products_df = self.products_df.merge(
            product_stats, on="product_id", how="left"
        )
        self.products_df["avg_rating"] = self.products_df["avg_rating"].fillna(0)
        self.products_df["num_ratings"] = self.products_df["num_ratings"].fillna(0).astype(int)

    def recommend(
        self,
        user_id=None,
        category=None,
        brand=None,
        min_price=None,
        max_price=None,
        min_rating=None,
        top_n=10,
    ):
        """
        Filter and rank products based on user constraints.

        Parameters:
            user_id:    optional, for context (not used in filtering itself)
            category:   str or list, category filter
            brand:      str or list, brand filter
            min_price:  float, minimum price
            max_price:  float, maximum price
            min_rating: float, minimum average rating threshold
            top_n:      int, number of recommendations to return
        """
        filtered = self.products_df.copy()
        constraints_applied = []

        # Apply constraints
        if category:
            if isinstance(category, str):
                category = [category]
            filtered = filtered[filtered["category"].isin(category)]
            constraints_applied.append(f"category in {category}")

        if brand:
            if isinstance(brand, str):
                brand = [brand]
            filtered = filtered[filtered["brand"].isin(brand)]
            constraints_applied.append(f"brand in {brand}")

        if min_price is not None:
            filtered = filtered[filtered["price"] >= min_price]
            constraints_applied.append(f"price ≥ ${min_price:.2f}")

        if max_price is not None:
            filtered = filtered[filtered["price"] <= max_price]
            constraints_applied.append(f"price ≤ ${max_price:.2f}")

        if min_rating is not None:
            filtered = filtered[filtered["avg_rating"] >= min_rating]
            constraints_applied.append(f"avg rating ≥ {min_rating:.1f}")

        if filtered.empty:
            return pd.DataFrame()

        # Exclude already-rated items for this user
        if user_id is not None:
            rated_pids = set(
                self.ratings_df[self.ratings_df["user_id"] == user_id]["product_id"]
            )
            filtered = filtered[~filtered["product_id"].isin(rated_pids)]

        if filtered.empty:
            return pd.DataFrame()

        # Rank by a score combining average rating and popularity
        filtered = filtered.copy()
        max_ratings_count = filtered["num_ratings"].max()
        if max_ratings_count > 0:
            popularity = filtered["num_ratings"] / max_ratings_count
        else:
            popularity = 0

        filtered["kb_score"] = (
            0.7 * filtered["avg_rating"] + 0.3 * popularity * 5
        )
        filtered["predicted_rating"] = filtered["kb_score"].clip(1.0, 5.0)

        result = filtered.nlargest(top_n, "predicted_rating").reset_index(drop=True)

        # Add explanations
        explanations = []
        for _, row in result.iterrows():
            reasons = []
            if category:
                reasons.append(f"belongs to your preferred category '{row['category']}'")
            if brand:
                reasons.append(f"from your preferred brand '{row['brand']}'")
            if min_price is not None or max_price is not None:
                price_range = ""
                if min_price is not None and max_price is not None:
                    price_range = f"${min_price:.0f}–${max_price:.0f}"
                elif min_price is not None:
                    price_range = f"above ${min_price:.0f}"
                else:
                    price_range = f"under ${max_price:.0f}"
                reasons.append(f"fits your budget ({price_range})")
            if min_rating is not None:
                reasons.append(f"rated {row['avg_rating']:.1f}/5 by other users")

            if not reasons:
                reasons.append("is a popular, well-rated product")

            explanation = (
                f"🎯 Recommended because it {', '.join(reasons)} "
                f"(KB score: {row['predicted_rating']:.2f}/5)"
            )
            explanations.append(explanation)

        result["explanation"] = explanations

        # Drop helper columns from display
        display_cols = [
            "product_id", "product_name", "category", "brand",
            "price", "description", "predicted_rating", "explanation",
            "avg_rating", "num_ratings"
        ]
        result = result[[c for c in display_cols if c in result.columns]]
        return result

    def get_available_categories(self):
        """Return list of all product categories."""
        return sorted(self.products_df["category"].unique().tolist())

    def get_available_brands(self, category=None):
        """Return list of brands, optionally filtered by category."""
        df = self.products_df
        if category:
            if isinstance(category, str):
                category = [category]
            df = df[df["category"].isin(category)]
        return sorted(df["brand"].unique().tolist())

    def get_price_range(self):
        """Return (min_price, max_price) tuple of the dataset."""
        return float(self.products_df["price"].min()), float(self.products_df["price"].max())
