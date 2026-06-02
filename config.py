"""
Configuration file for Netflix AI Recommendation System
"""

import os

# =================================
# PATHS
# =================================
DATA_DIR = "data"
PROCESSED_DATA_DIR = "data/processed"
MODELS_DIR = "models"
EMBEDDINGS_FILE = f"{MODELS_DIR}/v2_embeddings.npy"
INDEX_FILE = f"{MODELS_DIR}/v2_index.faiss"
DATAFRAME_FILE = f"{MODELS_DIR}/v2_df.pkl"

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# =================================
# DATA FILES
# =================================
LOCAL_CSV = f"{DATA_DIR}/movies_on_netflix.csv"
PROCESSED_CSV = f"{PROCESSED_DATA_DIR}/netflix_processed.csv"

# =================================
# API KEYS
# =================================
import os

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# =================================
# MODEL SETTINGS
# =================================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast & lightweight
EMBEDDING_DIMENSION = 384

# =================================
# FAISS SETTINGS
# =================================
FAISS_INDEX_TYPE = "IndexFlatL2"  # L2 distance
SEARCH_TOP_K = 10  # Number of recommendations

# =================================
# DATA VALIDATION SETTINGS
# =================================
MIN_DESCRIPTION_LENGTH = 10  # Minimum characters in description
REQUIRED_COLUMNS = ["title", "type", "listed_in", "description"]

# =================================
# API SETTINGS
# =================================
API_TIMEOUT = 5  # seconds
API_MAX_RETRIES = 3
API_REQUESTS_PER_BATCH = 20

# =================================
# STREAMLIT SETTINGS
# =================================
PAGE_TITLE = "Netflix AI Dashboard"
PAGE_ICON = "🎬"
LAYOUT = "wide"

# =================================
# DATA COLLECTION SETTINGS
# =================================
FETCH_TRENDING = True  # Fetch trending movies
FETCH_BOLLYWOOD = True  # Fetch Bollywood movies
FETCH_POPULAR = True  # Fetch popular movies

# Bollywood keywords for filtering
BOLLYWOOD_KEYWORDS = ["bollywood", "hindi", "india", "indian"]