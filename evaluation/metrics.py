"""
Evaluation & Comparison Module — implements 4 evaluation metrics:
  1. Precision@K
  2. Recall@K
  3. RMSE
  4. NDCG@K
"""

import numpy as np
import pandas as pd


def precision_at_k(recommended_items, relevant_items, k=10):
    """
    Precision@K: fraction of top-K recommended items that are relevant.
    relevant_items: set of product_ids the user actually rated highly (>=4).
    """
    top_k = recommended_items[:k]
    if len(top_k) == 0:
        return 0.0
    hits = len(set(top_k) & set(relevant_items))
    return hits / len(top_k)


def recall_at_k(recommended_items, relevant_items, k=10):
    """
    Recall@K: fraction of relevant items that appear in top-K recommendations.
    """
    top_k = recommended_items[:k]
    if len(relevant_items) == 0:
        return 0.0
    hits = len(set(top_k) & set(relevant_items))
    return hits / len(relevant_items)


def rmse(predicted_ratings, actual_ratings):
    """
    Root Mean Squared Error between predicted and actual ratings.
    predicted_ratings: dict {product_id: predicted_rating}
    actual_ratings: dict {product_id: actual_rating}
    """
    common = set(predicted_ratings.keys()) & set(actual_ratings.keys())
    if len(common) == 0:
        return float("nan")
    errors = [(predicted_ratings[pid] - actual_ratings[pid]) ** 2 for pid in common]
    return np.sqrt(np.mean(errors))


def dcg_at_k(scores, k):
    """Discounted Cumulative Gain at K."""
    scores = scores[:k]
    if len(scores) == 0:
        return 0.0
    return scores[0] + np.sum(scores[1:] / np.log2(np.arange(2, len(scores) + 1)))


def ndcg_at_k(recommended_items, relevant_items_with_scores, k=10):
    """
    Normalized Discounted Cumulative Gain at K.
    relevant_items_with_scores: dict {product_id: relevance_score}
    """
    top_k = recommended_items[:k]

    # Actual gains
    gains = [relevant_items_with_scores.get(pid, 0.0) for pid in top_k]
    actual_dcg = dcg_at_k(gains, k)

    # Ideal gains
    ideal_gains = sorted(relevant_items_with_scores.values(), reverse=True)[:k]
    ideal_dcg = dcg_at_k(ideal_gains, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def evaluate_recommender(recommender_func, test_df, ratings_df, k=10):
    """
    Evaluate a recommender function across all test users.

    Parameters:
        recommender_func: callable(user_id) -> list of recommended product_ids
        test_df: DataFrame with columns [user_id, product_id, rating]
        ratings_df: full ratings DataFrame (for building relevant sets)
        k: top-K for evaluation

    Returns:
        dict with average Precision@K, Recall@K, RMSE, NDCG@K
    """
    test_users = test_df["user_id"].unique()

    precisions = []
    recalls = []
    ndcgs = []
    all_rmse_errors = []

    for user_id in test_users:
        # Get test-set items this user rated
        user_test = test_df[test_df["user_id"] == user_id]
        if user_test.empty:
            continue

        # Relevant items = items rated >= 4 in test set
        relevant = set(user_test[user_test["rating"] >= 4]["product_id"])
        relevant_with_scores = dict(
            zip(user_test["product_id"], user_test["rating"])
        )

        # Get recommendations
        try:
            result = recommender_func(user_id)
            if isinstance(result, pd.DataFrame):
                if result.empty:
                    continue
                rec_items = result["product_id"].tolist()
                if "predicted_rating" in result.columns:
                    pred_ratings = dict(
                        zip(result["product_id"], result["predicted_rating"])
                    )
                else:
                    pred_ratings = {}
            elif isinstance(result, dict):
                sorted_preds = sorted(result.items(), key=lambda x: x[1], reverse=True)
                rec_items = [pid for pid, _ in sorted_preds[:k]]
                pred_ratings = result
            elif isinstance(result, list):
                rec_items = result[:k]
                pred_ratings = {}
            else:
                continue
        except Exception:
            continue

        if not rec_items:
            continue

        precisions.append(precision_at_k(rec_items, relevant, k))
        recalls.append(recall_at_k(rec_items, relevant, k))
        ndcgs.append(ndcg_at_k(rec_items, relevant_with_scores, k))

        # RMSE on overlapping predictions
        actual = dict(zip(user_test["product_id"], user_test["rating"]))
        if pred_ratings:
            r = rmse(pred_ratings, actual)
            if not np.isnan(r):
                all_rmse_errors.append(r)

    return {
        "Precision@K": np.mean(precisions) if precisions else 0.0,
        "Recall@K": np.mean(recalls) if recalls else 0.0,
        "NDCG@K": np.mean(ndcgs) if ndcgs else 0.0,
        "RMSE": np.mean(all_rmse_errors) if all_rmse_errors else float("nan"),
        "num_users_evaluated": len(precisions),
    }


def compare_methods(results_dict):
    """
    Format comparison results into a DataFrame.
    results_dict: {method_name: {metric_name: value, ...}, ...}
    """
    rows = []
    for method, metrics in results_dict.items():
        row = {"Method": method}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def generate_analysis(results_df):
    """Generate textual analysis of comparison results."""
    analysis = []

    metrics = ["Precision@K", "Recall@K", "NDCG@K", "RMSE"]

    for metric in metrics:
        if metric not in results_df.columns:
            continue
        col = results_df[metric].dropna()
        if col.empty:
            continue
        if metric == "RMSE":
            best_idx = col.idxmin()
            worst_idx = col.idxmax()
            analysis.append(
                f"**{metric}** (lower is better): "
                f"Best = **{results_df.loc[best_idx, 'Method']}** ({col[best_idx]:.4f}), "
                f"Worst = **{results_df.loc[worst_idx, 'Method']}** ({col[worst_idx]:.4f})"
            )
        else:
            best_idx = col.idxmax()
            worst_idx = col.idxmin()
            analysis.append(
                f"**{metric}** (higher is better): "
                f"Best = **{results_df.loc[best_idx, 'Method']}** ({col[best_idx]:.4f}), "
                f"Worst = **{results_df.loc[worst_idx, 'Method']}** ({col[worst_idx]:.4f})"
            )

    # Overall winner
    score_cols = [c for c in ["Precision@K", "Recall@K", "NDCG@K"] if c in results_df.columns]
    if score_cols:
        results_df_copy = results_df.copy()
        results_df_copy["avg_score"] = results_df_copy[score_cols].mean(axis=1)
        best_overall = results_df_copy.loc[results_df_copy["avg_score"].idxmax(), "Method"]
        analysis.append(f"\n🏆 **Overall Best Method**: **{best_overall}** (highest average across Precision, Recall, and NDCG)")

    return "\n\n".join(analysis)
