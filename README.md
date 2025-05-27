# Smart-Content-Recommendation-System

🧠 SmartContent: CNET Topic Modeling & Recommender System
This project scrapes the CNET homepage to extract article headlines, applies NLP + LDA Topic Modeling to categorize the content, and builds a personalized recommender system using ALS (Alternating Least Squares). A simple Streamlit app allows users to explore trending topics and receive article recommendations.

## 🚀 Features

🔍 Scrapes headlines (<h1>, <h2>, <h3>) and paragraphs from the CNET homepage

🧼 Cleans and tokenizes text using spaCy

📊 Extracts latent topics with Gensim's LDA

🏷 Assigns human-readable topic labels

🧠 Builds a recommender using Implicit ALS

🎯 Interactive Streamlit app with:

Topic filtering

Article recommendations

Optional user feedback system (👍 / 👎)


🧠 How It Works

Scraping: Extracts structured headlines and text from https://www.cnet.com/

Preprocessing: Uses spaCy for lemmatization, stopword removal, and tokenization

Topic Modeling: Applies LDA (Latent Dirichlet Allocation) to detect 12 major content themes

Labeling: Manually maps topics to human-readable labels like “AI & Tech Advice”

Recommender Engine: Uses implicit ALS on a simulated user interaction dataset

App: Runs a Streamlit UI to interactively display recommendations by topic




📈 Example Output

Topic #1: Politics & Tech Market Impact

Topic #5: Home Tech Essentials & Appliances

User 7 Recommendations:

📰 Energy and Utilities

📰 Crossing the Broadband Divide

📰 Memorial Day Sales

