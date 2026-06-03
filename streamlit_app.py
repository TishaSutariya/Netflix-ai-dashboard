"""
Netflix AI Recommendation Engine - Streamlit App
FINAL BUG-FREE VERSION - June 2026
Fixed: set slicing error + poster loading issues
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import requests
import numpy as np
from embeddings import ModelManager
from sklearn.neighbors import NearestNeighbors
from recommender import RecommendationEngine
from config import (
    PAGE_TITLE,
    PAGE_ICON,
    LAYOUT,
    TMDB_API_KEY,
    TMDB_BASE_URL
)

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)
import os

TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", os.getenv("TMDB_API_KEY", ""))

if "user" not in st.session_state:
    st.session_state.user = ""

if "history" not in st.session_state:
    st.session_state.history = []

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

if "poster_cache" not in st.session_state:
    st.session_state.poster_cache = {}

@st.cache_resource
def load_models():
    df, embeddings, index = ModelManager.load_all()
    
    if df is None:
        st.error("❌ Models not found! Run: python build_models.py")
        st.stop()
    
    required_cols = ['title', 'type', 'listed_in', 'release_year', 'description']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing columns: {missing_cols}")
        st.stop()
    
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce').fillna(2000).astype(int)
    
    engine = RecommendationEngine(df, embeddings, index)
    return df, engine

df, recommender = load_models()

@st.cache_data(ttl=3600)
def safe_tmdb_api_cached(query):
    url = f"{TMDB_BASE_URL}/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": query,
        "language": "en-US"
    }
    
    try:
        response = requests.get(url, params=params, timeout=8)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {"results": []}

def get_poster(title):
    """Get movie poster with caching and better error handling"""
    if title in st.session_state.poster_cache:
        return st.session_state.poster_cache[title]
    
    # Check if API key exists
    if not TMDB_API_KEY:
        placeholder = "https://via.placeholder.com/300x450/1a1a2e/ffffff?text=No+API+Key"
        st.session_state.poster_cache[title] = placeholder
        return placeholder
    
    try:
        data = safe_tmdb_api_cached(title)
        
        # Debug: Check what we got
        if not data.get("results"):
            placeholder = "https://via.placeholder.com/300x450/1a1a2e/ffffff?text=No+Result"
            st.session_state.poster_cache[title] = placeholder
            return placeholder
        
        if len(data["results"]) > 0:
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                st.session_state.poster_cache[title] = poster_url
                return poster_url
    except Exception as e:
        st.error(f"❌ Poster error for {title}: {str(e)}")  # Shows exact error
        pass
    
    # Professional fallback placeholder
    placeholder = "https://via.placeholder.com/300x450/1a1a2e/ffffff?text=No+Image"
    st.session_state.poster_cache[title] = placeholder
    return placeholder

def safe_get_value(row, column, default="N/A"):
    try:
        if hasattr(row, 'get'):
            value = row.get(column, default)
        else:
            value = getattr(row, column, default)
        
        if pd.isna(value) or value is None:
            return default
        return str(value).strip()
    except:
        return default

def filter_trending_last_10_years(df_trending):
    current_year = 2026
    min_year = current_year - 10
    
    try:
        df_trending['release_year'] = pd.to_numeric(df_trending['release_year'], errors='coerce').fillna(current_year).astype(int)
        filtered = df_trending[df_trending['release_year'] >= min_year].copy()
        return filtered
    except:
        return df_trending

st.sidebar.title("👤 User Panel")

user_input = st.sidebar.text_input("Enter Username", key="sidebar_user_input")
if user_input and user_input.strip():
    st.session_state.user = user_input.strip()
    st.sidebar.success(f"Welcome {st.session_state.user} 🎉")

if st.session_state.user:
    st.sidebar.info(f"✨ Active User: {st.session_state.user}")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📑 Navigation",
    ["🏠 Home", "🎯 Recommend", "🔍 Search", "🔥 Trending (2016-2026)", "📥 My Watchlist", "📊 Analytics"],
    key="sidebar_navigation"
)
banner_found = False

for banner_name in ["banner.PNG", "banner.png", "Banner.png"]:
    banner = Path(banner_name)

    if banner.exists():
        try:
            # Use absolute path and add error handling
            banner_abs = str(banner.resolve())
            st.image(banner_abs, use_container_width=True)
            banner_found = True
            break
        except Exception as e:
            # If this banner fails to load, try next one
            print(f"Could not load {banner_name}: {e}")  # Debug
            continue
    # else: Continue to next banner name

if not banner_found:
    st.title("🎬 Netflix AI Dashboard")

if not banner_found:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 40px; border-radius: 15px; margin-bottom: 30px; 
                text-align: center; color: white;'>
        <h1 style='margin: 0; font-size: 48px;'>🎬 Netflix AI</h1>
        <p style='margin: 10px 0 0 0; opacity: 0.9; font-size: 20px;'>
            Intelligent Movie Recommendation Engine
        </p>
    </div>
    """, unsafe_allow_html=True)
if page == "🏠 Home":
    st.title("🎬 Netflix AI Recommendation Engine")
    
    if st.session_state.user:
        st.success(f"👋 Hello {st.session_state.user}, enjoy your recommendations! 🍿")
    else:
        st.info("💡 Enter username in sidebar for personalized experience!")
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    try:
        stats = recommender.get_statistics()
        total_titles = stats.get('total_movies', len(df))
        movies_count = stats.get('total_movies_count', (df['type'] == 'Movie').sum() if 'type' in df.columns else 0)
        shows_count = stats.get('total_shows_count', (df['type'] == 'TV Show').sum() if 'type' in df.columns else 0)
        
        col1.metric("📺 Total Titles", f"{total_titles:,}")
        col2.metric("🎬 Movies", f"{movies_count:,}")
        col3.metric("📺 TV Shows", f"{shows_count:,}")
    except:
        col1.metric("📺 Total Titles", f"{len(df):,}")
        col2.metric("🎬 Movies", "N/A")
        col3.metric("📺 TV Shows", "N/A")
    
    st.divider()
    
    st.markdown("""
    ### 🚀 Features
    
    | Feature | Description |
    |---------|-------------|
    | **🎯 AI Recommendations** | Smart suggestions using AI embeddings |
    | **🔍 Advanced Search** | Find movies by title, genre, or description |
    | **🔥 Trending (2016-2026)** | Latest popular movies from last 10 years |
    | **📥 My Watchlist** | Save movies to watch later (no refresh!) |
    | **📊 Analytics Dashboard** | Visual insights into movie catalog |
    """)
    
    st.divider()
    st.info("💡 **Try:** 'Batman', 'Avengers', 'Stranger Things', 'Interstellar'")

elif page == "🎯 Recommend":
    st.title("🎯 AI-Powered Recommendations")
    st.markdown("Select a movie you like, and our AI will suggest similar content!")
    st.divider()
    
    try:
        unique_titles = sorted(df['title'].dropna().unique())
    except:
        st.error("❌ Could not load titles")
        st.stop()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_movie = st.selectbox(
            "🎬 Select a Movie or Show You Like",
            unique_titles,
            index=len(unique_titles)//2 if len(unique_titles) > 0 else 0
        )
    
    with col2:
        top_n = st.slider("Number of Recommendations", 1, 20, 5)
    
    st.divider()
    
    if st.button("✨ Get AI Recommendations", use_container_width=True, type="primary"):
        with st.spinner("🤖 AI is finding similar movies..."):
            results = None
            try:
                results = recommender.recommend(selected_movie, top_n)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        if results is None or len(results) == 0:
            st.error("❌ Could not find recommendations")
        else:
            if selected_movie not in st.session_state.history:
                st.session_state.history.append(selected_movie)
            
            st.success(f"✅ Found {len(results)} matches for *'{selected_movie}'*!")
            st.divider()
            
            for idx, (_, row) in enumerate(results.iterrows(), 1):
                with st.container():
                    title = safe_get_value(row, 'title', 'Unknown')
                    movie_type = safe_get_value(row, 'type', 'Movie')
                    genres = safe_get_value(row, 'listed_in', 'N/A')
                    year = safe_get_value(row, 'release_year', 'N/A')
                    description = safe_get_value(row, 'description', 'No description')
                    
                    if len(description) > 250:
                        description = description[:250] + "..."
                    
                    col1, col2 = st.columns([0.18, 0.82])
                    
                    with col1:
                        try:
                            poster = get_poster(title)
                            st.image(poster, width=140)
                        except:
                            st.image("https://via.placeholder.com/300x450/1a1a2e/ffffff?text=No+Image", width=140)
                    
                    with col2:
                        st.subheader(f"#{idx} {title}")
                        
                        c1, c2, c3 = st.columns(3)
                        c1.markdown(f"**📺 Type:** {movie_type}")
                        c2.markdown(f"**🎭 Genres:** {genres[:50]}")
                        c3.markdown(f"**📅 Year:** {year}")
                        
                        st.markdown(f"**📝 Description:** {description}")
                        
                        if 'similarity_score' in row.index:
                            try:
                                score = min(max(float(row['similarity_score']), 0), 100)
                                st.progress(score / 100)
                                st.caption(f"AI Match: **{score:.1f}%**")
                            except:
                                pass
                        
                        st.divider()
                        
                        in_watchlist = title in st.session_state.watchlist
                        
                        if in_watchlist:
                            st.markdown("✅ **In Watchlist**")
                        else:
                            if st.button("📥 Add to Watchlist", key=f"watch_{idx}"):
                                st.session_state.watchlist.append(title)
                                st.info(f"✅ Added '{title}' to watchlist!", icon="✅")
                    
                    st.divider()

elif page == "🔍 Search":
    st.title("🔍 Search Movies & Shows")
    st.divider()
    
    search_query = st.text_input("🔎 Enter search query", placeholder="e.g., Batman, Action, 2020", key="search_query_input")
    
    if search_query and search_query.strip():
        with st.spinner("🔍 Searching..."):
            results = pd.DataFrame()
            try:
                results = recommender.search_movies(search_query.strip(), limit=25)
            except:
                pass
        
        if len(results) == 0:
            st.warning("❌ No results found")
        else:
            st.success(f"✅ Found {len(results)} results")
            st.divider()
            
            for idx, (_, row) in enumerate(results.iterrows(), 1):
                with st.container():
                    title = safe_get_value(row, 'title', 'Unknown')
                    movie_type = safe_get_value(row, 'type', 'Movie')
                    year = safe_get_value(row, 'release_year', 'N/A')
                    genres = safe_get_value(row, 'listed_in', 'N/A')
                    description = safe_get_value(row, 'description', 'No description')
                    
                    if len(description) > 200:
                        description = description[:200] + "..."
                    
                    col1, col2 = st.columns([0.22, 0.78])
                    
                    with col1:
                        try:
                            poster = get_poster(title)
                            st.image(poster, width=130)
                        except:
                            st.image("https://via.placeholder.com/300x450/1a1a2e/ffffff?text=No+Image", width=130)
                    
                    with col2:
                        st.subheader(f"{idx}. {title}")
                        st.markdown(f"**📺 Type:** {movie_type} | **📅 Year:** {year}")
                        st.markdown(f"**🎭 Genres:** {genres}")
                        st.markdown(f"**📝 Description:** {description}")
                        
                        in_watchlist = title in st.session_state.watchlist
                        if in_watchlist:
                            st.caption("✅ In watchlist")
                        elif st.button("📥 Add", key=f"search_watch_{idx}"):
                            st.session_state.watchlist.append(title)
                            st.info(f"✅ Added to watchlist!")
                    
                    st.divider()

elif page == "🔥 Trending (2016-2026)":
    st.title("🔥 Trending Movies (2016-2026)")
    st.markdown("Latest popular movies from the last 10 years only")
    st.divider()
    
    with st.spinner("🔥 Loading trending..."):
        trending = pd.DataFrame()
        try:
            trending = recommender.get_trending(limit=30)
            if len(trending) > 0:
                trending = filter_trending_last_10_years(trending)
        except:
            pass
    
    if len(trending) == 0:
        st.warning("⚠️ No trending data. Run `python build_models.py`")
    else:
        st.success(f"✨ Showing {len(trending)} trending movies from **2016-2026**")
        st.divider()
        
        for idx, (_, row) in enumerate(trending.iterrows(), 1):
            with st.container():
                title = safe_get_value(row, 'title', 'Unknown')
                year = safe_get_value(row, 'release_year', 'N/A')
                movie_type = safe_get_value(row, 'type', 'Movie')
                genres = safe_get_value(row, 'listed_in', 'N/A')
                description = safe_get_value(row, 'description', 'No description')
                
                if len(description) > 200:
                    description = description[:200] + "..."
                
                try:
                    if int(year) < 2016:
                        continue
                except:
                    pass
                
                col1, col2 = st.columns([0.22, 0.78])
                
                with col1:
                    try:
                        poster = get_poster(title)
                        st.image(poster, width=130)
                    except:
                        st.image("https://via.placeholder.com/300x450/1a1a2e/ffffff?text=No+Image", width=130)
                
                with col2:
                    st.subheader(f"#{idx} {title}")
                    st.markdown(f"**📺 Type:** {movie_type} | **📅 Year:** {year}")
                    st.markdown(f"**🎭 Genres:** {genres}")
                    st.markdown(f"**📝 Description:** {description}")
                    
                    in_watchlist = title in st.session_state.watchlist
                    if not in_watchlist and st.button("📥 Add", key=f"trend_watch_{idx}"):
                        st.session_state.watchlist.append(title)
                        st.info("✅ Added to watchlist!")
                
                st.divider()

elif page == "📥 My Watchlist":
    st.title("📥 My Watchlist")
    st.markdown("Movies you want to watch later (no page refresh when adding!)")
    st.divider()
    
    if not st.session_state.watchlist:
        st.info("🎬 **Watchlist is empty!**")
        st.markdown("""
        Add movies to your watchlist by clicking **"📥 Add to Watchlist"** on:
        - Recommendation page
        - Search results  
        - Trending movies
        """)
        st.divider()
        st.info("💡 Your watchlist will appear here automatically")
    else:
        st.success(f"✨ You have **{len(st.session_state.watchlist)}** movie(s) in watchlist")
        st.divider()
        
        if st.button("🗑️ Clear Watchlist"):
            st.session_state.watchlist = []
            st.rerun()
        
        st.divider()
        
        for idx, movie_title in enumerate(st.session_state.watchlist, 1):
            movie_info = pd.DataFrame()
            try:
                movie_info = df[df['title'] == movie_title]
                if movie_info.empty:
                    movie_info = df[df['title'].str.lower() == movie_title.lower()]
            except:
                pass
            
            if not movie_info.empty:
                row = movie_info.iloc[0]
                title = safe_get_value(row, 'title', movie_title)
                movie_type = safe_get_value(row, 'type', 'Movie')
                genres = safe_get_value(row, 'listed_in', 'N/A')
                year = safe_get_value(row, 'release_year', 'N/A')
                
                col1, col2, col3 = st.columns([0.22, 0.70, 0.08])
                
                with col1:
                    try:
                        poster = get_poster(title)
                        st.image(poster, width=120)
                    except:
                        st.image("https://via.placeholder.com/300x450/1a1a2e/ffffff?text=No+Image", width=120)
                
                with col2:
                    st.subheader(f"{idx}. {title}")
                    st.markdown(f"**📺 Type:** {movie_type} | **📅 Year:** {year}")
                    st.markdown(f"**🎭 Genres:** {genres}")
                
                with col3:
                    if st.button("❌", key=f"remove_{idx}"):
                        st.session_state.watchlist = [m for m in st.session_state.watchlist if m != movie_title]
                        st.rerun()
                
                st.divider()

elif page == "📊 Analytics":
    st.title("📊 Analytics Dashboard")
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📺 Total Titles", len(df))
    col2.metric("🎬 Movies", (df['type'] == 'Movie').sum() if 'type' in df.columns else 0)
    col3.metric("📺 TV Shows", (df['type'] == 'TV Show').sum() if 'type' in df.columns else 0)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'type' in df.columns:
            type_counts = df['type'].value_counts()
            if len(type_counts) > 0:
                fig = px.pie(values=type_counts.values, names=type_counts.index, title="Movies vs TV Shows")
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'listed_in' in df.columns:
            try:
                genres = df['listed_in'].str.split(", ", expand=True).stack().value_counts().head(15)
                if len(genres) > 0:
                    fig = px.bar(x=genres.index, y=genres.values, title="Top 15 Genres")
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            except:
                pass
    
    st.divider()
    
    if 'release_year' in df.columns:
        try:
            year_data = pd.to_numeric(df['release_year'], errors='coerce').dropna()
            if len(year_data) > 0:
                year_counts = year_data.value_counts().sort_index().tail(20)
                if len(year_counts) > 0:
                    fig = px.bar(x=year_counts.index.astype(int), y=year_counts.values, title="Titles by Year (Last 20)")
                    st.plotly_chart(fig, use_container_width=True)
        except:
            pass
    
    st.divider()
    st.subheader("🧠 Your Activity")
    if st.session_state.history:
        # FIX: Convert set to list first before slicing
        unique_history = list(dict.fromkeys(st.session_state.history))
        st.write(f"**Recommendations requested:** {len(unique_history)}")
        for i, movie in enumerate(unique_history[:10], 1):
            st.write(f"{i}. {movie}")

st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
Netflix AI Recommendation Engine | Professional Dashboard | June 2026
</div>
""", unsafe_allow_html=True)