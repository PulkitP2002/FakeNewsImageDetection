import requests
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
from urllib.parse import urlencode
from sentence_transformers import SentenceTransformer, util


# Load .env file
load_dotenv()
SERP_API_KEY = os.getenv("SERP_API_KEY")

def fetch_serp_news(query):
    url = "https://serpapi.com/search"
    params = {
        "engine": "google_news",
        "q": query,
        "api_key": SERP_API_KEY,
        "num": 5,
        "hl": "en",
        "gl": "in"
    }
    full_url = f"{url}?{urlencode(params)}"
    print(f"[INFO] 🔗 SerpAPI URL: {full_url}")
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        news_results = data.get("news_results", [])
        headlines = [article.get("title") for article in news_results]
        return headlines
    except Exception as e:
        print(f"[ERROR] SerpAPI News fetch failed: {e}")
        return []

model = SentenceTransformer("all-MiniLM-L6-v2")

def check_similarity(text, headlines):
    """
    Calculates the semantic similarity between the extracted/generated text
    and a list of fetched news headlines.
    Returns a score between 0 and 1 (1 means highly similar).
    """
    if not headlines:
        return 0.0

    text_embedding = model.encode(text, convert_to_tensor=True)
    headline_embeddings = model.encode(headlines, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(text_embedding, headline_embeddings)[0]
    avg_score = float(similarities.mean())  # Get average similarity across headlines

    return avg_score