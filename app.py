# app.py
import streamlit as st
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix
import numpy as np

# --- Load structured article data ---
df_articles = pd.read_csv("cnet_structureds.csv")

# --- Load user interaction data ---
df_users = pd.read_csv("Simulated_User_Interactions.csv")

# --- Build ALS model ---
user_map = {u: i for i, u in enumerate(df_users["user_id"].unique())}
item_map = {t: i for i, t in enumerate(df_users["text"].unique())}
item_reverse_map = {v: k for k, v in item_map.items()}

user_ids = df_users["user_id"].map(user_map)
item_ids = df_users["text"].map(item_map)
interactions = df_users["views"]

sparse_matrix = coo_matrix((interactions, (user_ids, item_ids))).tocsr()

als_model = AlternatingLeastSquares(factors=20, regularization=0.1, iterations=15, random_state=42)
als_model.fit(sparse_matrix)

# --- Streamlit UI ---
st.set_page_config(page_title="CNET Recommender", layout="centered")
st.title("🔍 CNET Article Recommender")

user_choices = sorted(df_users["user_id"].unique())
selected_user = st.selectbox("Select a User ID:", user_choices)

if st.button("Get Recommendations"):
    try:
        user_index = user_map[selected_user]
        item_ids, scores = als_model.recommend(
            userid=user_index,
            user_items=sparse_matrix[user_index],
            N=5,
            filter_already_liked_items=True
        )

        st.subheader(f"📚 Top 5 Recommendations for User {selected_user}:")
        for item_id, score in zip(item_ids, scores):
            article = item_reverse_map.get(item_id, "Unknown Article")
            article_info = df_articles[df_articles["text"] == article].iloc[0]
            st.markdown(f"**📰 {article}**")
            st.markdown(f"*Topic:* {article_info['topic_label']}")
            st.markdown(f"*Score:* {score:.4f}")
            st.markdown(f"[🔗 Visit]({article_info['url']})")
            st.markdown("---")
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
