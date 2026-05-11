"""
Explainability utilities — generates human-readable explanations
for why each recommendation was made.
"""


def cf_explanation(product, method_name, predicted_rating):
    """Generate explanation for a CF recommendation."""
    if "cosine" in method_name.lower() and "user" in method_name.lower():
        return (
            f"🤝 Recommended because users with similar tastes (measured by cosine similarity) "
            f"also purchased this {product['category'].lower()} product. "
            f"Predicted rating: {predicted_rating:.1f}/5"
        )
    elif "pearson" in method_name.lower():
        return (
            f"🤝 Recommended because users with correlated rating patterns (Pearson correlation) "
            f"enjoyed this {product['category'].lower()} product. "
            f"Predicted rating: {predicted_rating:.1f}/5"
        )
    elif "item" in method_name.lower():
        return (
            f"📦 Recommended because this item is similar to other {product['category'].lower()} "
            f"products you've rated highly. "
            f"Predicted rating: {predicted_rating:.1f}/5"
        )
    elif "svd" in method_name.lower() or "matrix" in method_name.lower():
        return (
            f"🧮 Recommended through latent factor analysis (SVD) of rating patterns. "
            f"The system identified hidden features connecting your preferences to this product. "
            f"Predicted rating: {predicted_rating:.1f}/5"
        )
    return f"Recommended with predicted rating: {predicted_rating:.1f}/5"


def cb_explanation(product, matched_categories, matched_brands, similarity_score):
    """Generate explanation for a content-based recommendation."""
    reasons = []
    if matched_categories:
        reasons.append(f"matches your preferred {'categories' if len(matched_categories) > 1 else 'category'} "
                        f"({', '.join(matched_categories)})")
    if matched_brands:
        reasons.append(f"from {'brands' if len(matched_brands) > 1 else 'a brand'} you enjoy "
                        f"({', '.join(matched_brands)})")
    if not reasons:
        reasons.append("has similar features to products you've liked")

    return (
        f"📝 Recommended because it {' and '.join(reasons)}. "
        f"Content similarity: {similarity_score:.2f}"
    )


def kb_explanation(product, constraints):
    """Generate explanation for a knowledge-based recommendation."""
    reasons = []
    if constraints.get("category"):
        reasons.append(f"belongs to your selected category '{product['category']}'")
    if constraints.get("brand"):
        reasons.append(f"from your preferred brand '{product['brand']}'")
    if constraints.get("min_price") is not None or constraints.get("max_price") is not None:
        min_p = constraints.get("min_price")
        max_p = constraints.get("max_price")
        if min_p is not None and max_p is not None:
            reasons.append(f"fits within your budget (${min_p:.0f}–${max_p:.0f})")
        elif min_p is not None:
            reasons.append(f"is above your minimum price (${min_p:.0f})")
        else:
            reasons.append(f"is within your maximum budget (${max_p:.0f})")
    if constraints.get("min_rating") is not None:
        reasons.append(f"has a high average rating ({product.get('avg_rating', 'N/A')}★)")

    if not reasons:
        reasons.append("is a popular, well-rated product")

    return f"🎯 Recommended because it {', '.join(reasons)}."
