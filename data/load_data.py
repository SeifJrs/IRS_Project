"""
Data loading and preprocessing module.
Generates a realistic synthetic e-commerce dataset with users, products, and ratings.
"""

import pandas as pd
import numpy as np
import os
import json


def generate_synthetic_data(
    n_users=200,
    n_products=500,
    n_ratings=8000,
    seed=42
):
    """Generate a realistic synthetic e-commerce dataset."""
    np.random.seed(seed)

    # ── Product catalog ──────────────────────────────────────────────
    categories = [
        "Electronics", "Clothing", "Home & Kitchen", "Books",
        "Sports & Outdoors", "Beauty", "Toys & Games", "Automotive"
    ]
    brands_by_category = {
        "Electronics":       ["Samsung", "Apple", "Sony", "LG", "Dell", "HP", "Lenovo"],
        "Clothing":          ["Nike", "Adidas", "Zara", "H&M", "Levi's", "Puma", "Uniqlo"],
        "Home & Kitchen":    ["IKEA", "Cuisinart", "KitchenAid", "Dyson", "Philips"],
        "Books":             ["Penguin", "HarperCollins", "Random House", "Scholastic", "O'Reilly"],
        "Sports & Outdoors": ["Nike", "Adidas", "Under Armour", "Columbia", "The North Face"],
        "Beauty":            ["L'Oreal", "Maybelline", "Neutrogena", "Dove", "Olay"],
        "Toys & Games":      ["LEGO", "Hasbro", "Mattel", "Fisher-Price", "Nerf"],
        "Automotive":        ["Bosch", "3M", "Michelin", "Castrol", "Armor All"],
    }
    product_adjectives = [
        "Premium", "Ultra", "Pro", "Essential", "Classic",
        "Advanced", "Smart", "Elite", "Compact", "Deluxe"
    ]
    product_nouns_by_cat = {
        "Electronics":       ["Headphones", "Laptop", "Tablet", "Smartwatch", "Speaker", "Monitor", "Keyboard", "Mouse", "Camera", "Charger"],
        "Clothing":          ["T-Shirt", "Jacket", "Sneakers", "Jeans", "Hoodie", "Shorts", "Cap", "Backpack", "Socks", "Dress"],
        "Home & Kitchen":    ["Blender", "Coffee Maker", "Vacuum", "Air Fryer", "Toaster", "Pan Set", "Knife Set", "Lamp", "Organizer", "Pillow"],
        "Books":             ["Novel", "Textbook", "Guide", "Cookbook", "Biography", "Workbook", "Journal", "Atlas", "Anthology", "Manual"],
        "Sports & Outdoors": ["Running Shoes", "Yoga Mat", "Dumbbell Set", "Tent", "Water Bottle", "Bike Light", "Gloves", "Helmet", "Jersey", "Towel"],
        "Beauty":            ["Moisturizer", "Sunscreen", "Shampoo", "Lipstick", "Foundation", "Serum", "Face Wash", "Perfume", "Mask", "Cream"],
        "Toys & Games":      ["Building Set", "Board Game", "Action Figure", "Puzzle", "RC Car", "Doll", "Card Game", "Drone", "Robot Kit", "Plush Toy"],
        "Automotive":        ["Dash Cam", "Car Charger", "Seat Cover", "Floor Mat", "Wax Kit", "Tool Set", "Air Freshener", "Phone Mount", "Jump Starter", "Tire Gauge"],
    }
    price_ranges = {
        "Electronics":       (29.99, 999.99),
        "Clothing":          (14.99, 199.99),
        "Home & Kitchen":    (19.99, 499.99),
        "Books":             (9.99, 79.99),
        "Sports & Outdoors": (12.99, 299.99),
        "Beauty":            (7.99, 149.99),
        "Toys & Games":      (9.99, 199.99),
        "Automotive":        (9.99, 299.99),
    }

    products = []
    for pid in range(1, n_products + 1):
        cat = np.random.choice(categories)
        brand = np.random.choice(brands_by_category[cat])
        adj = np.random.choice(product_adjectives)
        noun = np.random.choice(product_nouns_by_cat[cat])
        name = f"{brand} {adj} {noun}"
        lo, hi = price_ranges[cat]
        price = round(np.random.uniform(lo, hi), 2)
        desc = (
            f"{name} — a top-rated {cat.lower()} product by {brand}. "
            f"Features {adj.lower()} design with excellent build quality. "
            f"Category: {cat}. Perfect for everyday use."
        )
        products.append({
            "product_id": pid,
            "product_name": name,
            "category": cat,
            "brand": brand,
            "price": price,
            "description": desc,
        })
    products_df = pd.DataFrame(products)

    # ── Users ────────────────────────────────────────────────────────
    user_names = [f"User_{uid}" for uid in range(1, n_users + 1)]
    # Each user has preferred categories (for realistic rating patterns)
    user_prefs = {}
    for uid in range(1, n_users + 1):
        n_pref = np.random.randint(1, 4)
        user_prefs[uid] = list(np.random.choice(categories, n_pref, replace=False))

    # ── Ratings ──────────────────────────────────────────────────────
    ratings = []
    seen = set()
    attempts = 0
    while len(ratings) < n_ratings and attempts < n_ratings * 5:
        attempts += 1
        uid = np.random.randint(1, n_users + 1)
        pid = np.random.randint(1, n_products + 1)
        if (uid, pid) in seen:
            continue
        seen.add((uid, pid))

        product_cat = products_df.loc[products_df["product_id"] == pid, "category"].values[0]
        # Higher ratings for preferred categories
        if product_cat in user_prefs.get(uid, []):
            rating = min(5.0, max(1.0, round(np.random.normal(4.0, 0.8))))
        else:
            rating = min(5.0, max(1.0, round(np.random.normal(3.0, 1.0))))

        ratings.append({
            "user_id": uid,
            "product_id": pid,
            "rating": float(rating),
        })

    ratings_df = pd.DataFrame(ratings)

    return products_df, ratings_df, user_prefs


def load_data(data_dir=None):
    """Load or generate the dataset and return products_df, ratings_df, user_prefs."""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocessed")
    os.makedirs(data_dir, exist_ok=True)

    products_path = os.path.join(data_dir, "products.csv")
    ratings_path = os.path.join(data_dir, "ratings.csv")
    prefs_path = os.path.join(data_dir, "user_prefs.json")

    if os.path.exists(products_path) and os.path.exists(ratings_path):
        products_df = pd.read_csv(products_path)
        ratings_df = pd.read_csv(ratings_path)
        if os.path.exists(prefs_path):
            with open(prefs_path, "r") as f:
                user_prefs = {int(k): v for k, v in json.load(f).items()}
        else:
            user_prefs = {}
    else:
        products_df, ratings_df, user_prefs = generate_synthetic_data()
        products_df.to_csv(products_path, index=False)
        ratings_df.to_csv(ratings_path, index=False)
        with open(prefs_path, "w") as f:
            json.dump(user_prefs, f)

    return products_df, ratings_df, user_prefs


def get_train_test_split(ratings_df, test_ratio=0.2, seed=42):
    """Split ratings into train and test sets."""
    np.random.seed(seed)
    shuffled = ratings_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int(len(shuffled) * (1 - test_ratio))
    train_df = shuffled.iloc[:split_idx].reset_index(drop=True)
    test_df = shuffled.iloc[split_idx:].reset_index(drop=True)
    return train_df, test_df
