import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.load_data import load_data, get_train_test_split
from recommenders.collaborative import CollaborativeFiltering
from recommenders.content_based import ContentBasedRecommender
from recommenders.knowledge_based import KnowledgeBasedRecommender
from evaluation.metrics import evaluate_recommender, compare_methods, generate_analysis

# ─── Configuration ───────────────────────────────────────────────────
st.set_page_config(
    page_title="RecSys Engine — AIE425",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --primary: #667eea;
        --secondary: #764ba2;
        --accent: #f093fb;
        --bg: #0f0f23;
        --card: #1e1e42;
    }

    /* Main background */
    .stApp {
        background-color: #0f0f23;
        color: #e8e8f0;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #12122e;
        border-right: 1px solid rgba(102, 126, 234, 0.15);
    }

    /* Card styling */
    .metric-card {
        background: #1e1e42;
        border: 1px solid rgba(102, 126, 234, 0.15);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(102, 126, 234, 0.4);
    }

    /* Gradient headers */
    .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* Product card */
    .product-card {
        background: #1e1e42;
        border: 1px solid rgba(102, 126, 234, 0.1);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
    }

    .explanation-box {
        background: rgba(102, 126, 234, 0.08);
        border-left: 4px solid #667eea;
        padding: 10px 15px;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        color: #a0a0c0;
        margin-top: 8px;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# ─── State Management & Data Loading ─────────────────────────────────
@st.cache_resource
def get_system_data():
    products_df, ratings_df, user_prefs = load_data()
    train_df, test_df = get_train_test_split(ratings_df)

    cf_engine = CollaborativeFiltering(train_df, products_df)
    cb_engine = ContentBasedRecommender(train_df, products_df)
    kb_engine = KnowledgeBasedRecommender(ratings_df, products_df)

    return {
        "products": products_df,
        "ratings": ratings_df,
        "user_prefs": user_prefs,
        "train_df": train_df,
        "test_df": test_df,
        "cf_engine": cf_engine,
        "cb_engine": cb_engine,
        "kb_engine": kb_engine
    }

data = get_system_data()

# ─── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1 class='gradient-text' style='font-size: 1.8rem; margin-bottom: 0;'>🛒 RecSys Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6b6b8d; font-size: 0.8rem; margin-bottom: 2rem;'>AIE425 · Intelligent Recommender System</p>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Home & Overview", "Collaborative Filtering", "Content-Based", "Knowledge-Based", "Evaluation"],
        index=0
    )

    st.markdown("---")
    st.markdown("### User Settings")
    user_ids = sorted(data["ratings"]["user_id"].unique())
    selected_user = st.selectbox("Select Active User", user_ids, index=0)

    top_n = st.slider("Number of Recommendations", 3, 20, 10)

    # User quick stats
    user_ratings = data["ratings"][data["ratings"]["user_id"] == selected_user]
    st.info(f"**User {selected_user} Stats:**\n\n⭐ {len(user_ratings)} Total Ratings\n\n📦 {len(user_ratings[user_ratings['rating'] >= 4])} Liked Products")

# ─── Helper Functions ────────────────────────────────────────────────
def display_recommendations(df):
    if df.empty:
        st.warning("No recommendations found for this user.")
        return

    for _, row in df.iterrows():
        with st.container():
            st.markdown(f"""
                <div class="product-card">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <span style="font-weight: 700; font-size: 1.1rem; color: #e8e8f0;">{row['product_name']}</span>
                        <span style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 700;">
                            {row['predicted_rating']:.1f} ★
                        </span>
                    </div>
                    <div style="margin: 5px 0; font-size: 0.85rem; color: #a0a0c0;">
                        <span style="margin-right: 15px;">🏷️ {row['category']}</span>
                        <span style="margin-right: 15px;">🏢 {row['brand']}</span>
                        <span>💰 ${row['price']:.2f}</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #6b6b8d; font-style: italic; margin-top: 5px;">
                        {row['description']}
                    </div>
                    <div class="explanation-box">
                        {row['explanation']}
                    </div>
                </div>
            """, unsafe_allow_html=True)

# ─── Home Page ───────────────────────────────────────────────────────
if page == "Home & Overview":
    st.markdown("<h1 class='gradient-text'>Dataset Overview</h1>", unsafe_allow_html=True)

    # Summary Stats
    cols = st.columns(4)
    with cols[0]:
        st.markdown(f'<div class="metric-card"><div style="font-size: 2.5rem; font-weight: 800; color: #667eea;">{len(data["products"])}</div><div style="color: #6b6b8d; text-transform: uppercase; font-size: 0.7rem; font-weight: 600;">Products</div></div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f'<div class="metric-card"><div style="font-size: 2.5rem; font-weight: 800; color: #f093fb;">{len(data["ratings"]["user_id"].unique())}</div><div style="color: #6b6b8d; text-transform: uppercase; font-size: 0.7rem; font-weight: 600;">Users</div></div>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f'<div class="metric-card"><div style="font-size: 2.5rem; font-weight: 800; color: #48bb78;">{len(data["ratings"])}</div><div style="color: #6b6b8d; text-transform: uppercase; font-size: 0.7rem; font-weight: 600;">Ratings</div></div>', unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f'<div class="metric-card"><div style="font-size: 2.5rem; font-weight: 800; color: #ed8936;">{data["ratings"]["rating"].mean():.2f}</div><div style="color: #6b6b8d; text-transform: uppercase; font-size: 0.7rem; font-weight: 600;">Avg Rating</div></div>', unsafe_allow_html=True)

    st.markdown("### System Architecture")
    arch_cols = st.columns(3)
    with arch_cols[0]:
        st.markdown("""
        #### 🤝 Collaborative Filtering
        Recommends products based on user-to-user and item-to-item similarities.
        - **4 Methods:** Cosine, Pearson, Item-Based, SVD.
        """)
    with arch_cols[1]:
        st.markdown("""
        #### 📝 Content-Based
        Analyzes item metadata (description, category) using TF-IDF.
        - **Features:** Brand, Category, Text Description.
        """)
    with arch_cols[2]:
        st.markdown("""
        #### 🎯 Knowledge-Based
        Filters products based on explicit constraints and ranks by popularity.
        - **Filters:** Budget, Brand, Category, Min Rating.
        """)

    st.markdown("### Data Distribution")
    dist_cols = st.columns(2)

    with dist_cols[0]:
        cat_counts = data["products"]["category"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(cat_counts.index, cat_counts.values, color='#667eea')
        ax.set_title("Products per Category", color='white')
        ax.set_facecolor('#1e1e42')
        fig.patch.set_facecolor('#0f0f23')
        ax.tick_params(colors='white', rotation=45)
        st.pyplot(fig)

    with dist_cols[1]:
        rating_counts = data["ratings"]["rating"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie(rating_counts.values, labels=rating_counts.index, autopct='%1.1f%%', 
               colors=['#667eea', '#764ba2', '#f093fb', '#48bb78', '#ed8936'],
               textprops={'color':"white"})
        ax.set_title("Rating Distribution", color='white')
        fig.patch.set_facecolor('#0f0f23')
        st.pyplot(fig)

# ─── Collaborative Filtering ─────────────────────────────────────────
elif page == "Collaborative Filtering":
    st.markdown("<h1 class='gradient-text'>Collaborative Filtering</h1>", unsafe_allow_html=True)
    st.write("Generate recommendations based on patterns from the entire user community.")

    col1, col2 = st.columns([1, 2])
    with col1:
        method_key = st.selectbox(
            "Select CF Method",
            ["user_cosine", "user_pearson", "item_cosine", "svd"],
            format_func=lambda x: data["cf_engine"].METHOD_NAMES.get(x, x)
        )

        if st.button("Generate CF Recommendations"):
            with st.spinner("Analyzing similarities..."):
                recs = data["cf_engine"].recommend(selected_user, method=method_key, top_n=top_n)
                st.session_state["cf_recs"] = recs

        if st.button("Compare All CF Methods"):
            with st.spinner("Running all models..."):
                all_recs = {}
                for m in ["user_cosine", "user_pearson", "item_cosine", "svd"]:
                    all_recs[m] = data["cf_engine"].recommend(selected_user, method=m, top_n=5)
                st.session_state["cf_comparison"] = all_recs

    with col2:
        if "cf_recs" in st.session_state:
            st.subheader(f"Top {top_n} via {data['cf_engine'].METHOD_NAMES.get(method_key)}")
            display_recommendations(st.session_state["cf_recs"])

    if "cf_comparison" in st.session_state:
        st.markdown("---")
        st.subheader("Comparative Analysis (Top 5 per Method)")
        comp_cols = st.columns(4)
        for i, m in enumerate(["user_cosine", "user_pearson", "item_cosine", "svd"]):
            with comp_cols[i]:
                st.markdown(f"**{data['cf_engine'].METHOD_NAMES.get(m)}**")
                df = st.session_state["cf_comparison"][m]
                if not df.empty:
                    for _, row in df.iterrows():
                        st.markdown(f"- {row['product_name']} ({row['predicted_rating']:.1f})")
                else:
                    st.write("No recs.")

# ─── Content-Based ───────────────────────────────────────────────────
elif page == "Content-Based":
    st.markdown("<h1 class='gradient-text'>Content-Based Recommendations</h1>", unsafe_allow_html=True)
    st.write("Personalized suggestions based on the specific attributes of products you've liked.")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### How it works")
        st.info("We build a profile of your tastes by analyzing the descriptions, categories, and brands of items you've rated. We then find products with high TF-IDF similarity to your profile.")

        if st.button("Generate Content-Based"):
            with st.spinner("Processing text features..."):
                recs = data["cb_engine"].recommend(selected_user, top_n=top_n)
                st.session_state["cb_recs"] = recs

    with col2:
        if "cb_recs" in st.session_state:
            st.subheader(f"Suggested for User {selected_user}")
            display_recommendations(st.session_state["cb_recs"])

# ─── Knowledge-Based ─────────────────────────────────────────────────
elif page == "Knowledge-Based":
    st.markdown("<h1 class='gradient-text'>Knowledge-Based Search</h1>", unsafe_allow_html=True)
    st.write("Specify your requirements to find the perfect match.")

    with st.expander("Filter Constraints", expanded=True):
        f_col1, f_col2, f_col3 = st.columns(3)

        with f_col1:
            cats = data["kb_engine"].get_available_categories()
            sel_cats = st.multiselect("Categories", cats)

        with f_col2:
            brands = data["kb_engine"].get_available_brands(sel_cats)
            sel_brands = st.multiselect("Brands", brands)

        with f_col3:
            min_p, max_p = data["kb_engine"].get_price_range()
            sel_price = st.slider("Price Range ($)", float(min_p), float(max_p), (float(min_p), float(max_p)))

        sel_rating = st.slider("Minimum Avg Rating", 0.0, 5.0, 0.0, 0.5)

    if st.button("Apply Knowledge Constraints"):
        with st.spinner("Filtering catalog..."):
            recs = data["kb_engine"].recommend(
                user_id=selected_user,
                category=sel_cats if sel_cats else None,
                brand=sel_brands if sel_brands else None,
                min_price=sel_price[0],
                max_price=sel_price[1],
                min_rating=sel_rating,
                top_n=top_n
            )
            st.session_state["kb_recs"] = recs

    if "kb_recs" in st.session_state:
        st.markdown("---")
        display_recommendations(st.session_state["kb_recs"])

# ─── Evaluation ──────────────────────────────────────────────────────
elif page == "Evaluation":
    st.markdown("<h1 class='gradient-text'>System Performance</h1>", unsafe_allow_html=True)
    st.write("Comprehensive benchmarking across all recommendation strategies.")

    if st.button("🚀 Run Comprehensive Evaluation"):
        with st.spinner("Evaluating models on test set... (this may take 10-20 seconds)"):
            # Setup functions for evaluation
            def get_cf_user_cosine(uid): return data["cf_engine"].predict_all(uid, "user_cosine")
            def get_cf_user_pearson(uid): return data["cf_engine"].predict_all(uid, "user_pearson")
            def get_cf_item_cosine(uid): return data["cf_engine"].predict_all(uid, "item_cosine")
            def get_cf_svd(uid): return data["cf_engine"].predict_all(uid, "svd")
            def get_cb(uid): return data["cb_engine"].predict_all(uid)

            results = {
                "User-Based Cosine":  evaluate_recommender(get_cf_user_cosine, data["test_df"], data["ratings"], k=top_n),
                "User-Based Pearson": evaluate_recommender(get_cf_user_pearson, data["test_df"], data["ratings"], k=top_n),
                "Item-Based CF":      evaluate_recommender(get_cf_item_cosine, data["test_df"], data["ratings"], k=top_n),
                "Matrix Factorization (SVD)": evaluate_recommender(get_cf_svd, data["test_df"], data["ratings"], k=top_n),
                "Content-Based":      evaluate_recommender(get_cb, data["test_df"], data["ratings"], k=top_n),
            }

            st.session_state["eval_results"] = results
            st.session_state["eval_df"] = compare_methods(results)

    if "eval_results" in st.session_state:
        df = st.session_state["eval_df"]

        # Metric cards for best performers
        best_cols = st.columns(4)
        metrics = ["Precision@K", "Recall@K", "NDCG@K", "RMSE"]
        for i, m in enumerate(metrics):
            with best_cols[i]:
                if m == "RMSE":
                    best_val = df[m].min()
                    best_method = df.loc[df[m].idxmin(), "Method"]
                else:
                    best_val = df[m].max()
                    best_method = df.loc[df[m].idxmax(), "Method"]
                st.markdown(f"""
                    <div class="metric-card" style="padding: 10px;">
                        <div style="font-size: 0.7rem; color: #6b6b8d; text-transform: uppercase;">Best {m}</div>
                        <div style="font-size: 1.2rem; font-weight: 800; color: #667eea;">{best_val:.4f}</div>
                        <div style="font-size: 0.8rem; color: #a0a0c0;">{best_method}</div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("### Metrics Comparison")
        st.dataframe(df.style.highlight_max(subset=["Precision@K", "Recall@K", "NDCG@K"], color="#1a3e1a")
                         .highlight_min(subset=["RMSE"], color="#1a3e1a"), use_container_width=True)

        # Matplotlib Bar Chart for Comparison
        st.markdown("### Performance Comparison")
        comp_metrics = ["Precision@K", "Recall@K", "NDCG@K"]
        fig, ax = plt.subplots(figsize=(10, 6))
        df.plot(x='Method', y=comp_metrics, kind='bar', ax=ax, color=['#667eea', '#764ba2', '#f093fb'])
        ax.set_title("Metric Comparison by Method", color='white')
        ax.set_facecolor('#1e1e42')
        fig.patch.set_facecolor('#0f0f23')
        ax.tick_params(colors='white', rotation=45)
        ax.legend(facecolor='#1e1e42', edgecolor='#667eea', labelcolor='white')
        st.pyplot(fig)

        st.markdown("### Analysis Summary")
        analysis_text = generate_analysis(df)
        analysis_html = analysis_text.replace('\n', '<br>')
        st.markdown(f"""
            <div style="background: #1e1e42; border: 1px solid rgba(102, 126, 234, 0.2); border-radius: 12px; padding: 20px;">
                {analysis_html}
            </div>
        """, unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #6b6b8d; font-size: 0.8rem;">
        Intelligent Recommender System Project — AIE425 — Built with Streamlit & Matplotlib
    </div>
""", unsafe_allow_html=True)
