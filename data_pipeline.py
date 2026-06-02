"""
Unified Data Pipeline
Loads data from local CSV and TMDB API, combines them
"""

import pandas as pd
import requests
import numpy as np
from config import (
    LOCAL_CSV,
    TMDB_API_KEY,
    TMDB_BASE_URL,
    API_TIMEOUT,
    API_MAX_RETRIES
)

class TMDBLoader:
    """Fetch movies from TMDB API"""
    
    def __init__(self, api_key=TMDB_API_KEY):
        self.api_key = api_key
        self.base_url = TMDB_BASE_URL
    
    def safe_request(self, endpoint, params):
        """Safe API request with retry logic"""
        params['api_key'] = self.api_key
        
        for attempt in range(API_MAX_RETRIES):
            try:
                response = requests.get(
                    f"{self.base_url}{endpoint}",
                    params=params,
                    timeout=API_TIMEOUT
                )
                
                if response.status_code == 200:
                    return response.json()
                
            except Exception as e:
                if attempt == API_MAX_RETRIES - 1:
                    print(f"⚠️  API Error (attempt {attempt+1}): {str(e)}")
        
        return {"results": []}
    
    def format_movie(self, movie):
        """Format TMDB movie to our standard format"""
        return {
            'title': movie.get('title', 'Unknown'),
            'type': 'Movie',
            'listed_in': 'Movies, Action, Adventure',  # Default genres
            'description': movie.get('overview', 'No description'),
            'release_year': int(movie.get('release_date', '2024')[:4]) if movie.get('release_date') else 2024,
            'rating': movie.get('vote_average', 0),
            'source': 'TMDB'
        }
    
    def get_trending(self, page=1):
        """Get trending movies"""
        print("📡 Fetching trending movies from TMDB...")
        
        data = self.safe_request(
            "/movie/now_playing",
            {"page": page}
        )
        
        movies = [self.format_movie(m) for m in data.get("results", [])]
        print(f"   ✓ Fetched {len(movies)} trending movies")
        
        return pd.DataFrame(movies)
    
    def get_popular(self, page=1):
        """Get popular movies"""
        print("📡 Fetching popular movies from TMDB...")
        
        data = self.safe_request(
            "/movie/popular",
            {"page": page}
        )
        
        movies = [self.format_movie(m) for m in data.get("results", [])]
        print(f"   ✓ Fetched {len(movies)} popular movies")
        
        return pd.DataFrame(movies)
    
    def get_bollywood(self):
        """Get Bollywood/Hindi movies"""
        print("📡 Fetching Bollywood movies from TMDB...")
        
        # Try multiple strategies
        strategies = [
            {"with_original_language": "hi"},  # Hindi language
            {"with_keywords": "bollywood|indian"},
            {"with_companies": "15983|4|3889"},  # Indian production companies
        ]
        
        all_movies = []
        
        for i, params in enumerate(strategies):
            data = self.safe_request("/discover/movie", params)
            movies = [self.format_movie(m) for m in data.get("results", [])]
            all_movies.extend(movies)
            
            if len(movies) > 0:
                print(f"   ✓ Strategy {i+1}: Got {len(movies)} movies")
        
        # Remove duplicates
        df = pd.DataFrame(all_movies)
        df = df.drop_duplicates(subset=['title'], keep='first')
        
        print(f"   ✓ Total Bollywood movies: {len(df)}")
        return df


class LocalDataLoader:
    """Load data from local CSV files"""
    
    @staticmethod
    def load_csv(filepath):
        """Load CSV file"""
        print(f"📂 Loading local CSV: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            df['source'] = 'LocalDB'
            print(f"   ✓ Loaded {len(df)} records")
            return df
        
        except FileNotFoundError:
            print(f"   ❌ File not found: {filepath}")
            return pd.DataFrame()
        
        except Exception as e:
            print(f"   ❌ Error loading file: {str(e)}")
            return pd.DataFrame()


class DataPipeline:
    """Main pipeline that combines all data sources"""
    
    def __init__(self):
        self.tmdb = TMDBLoader()
        self.local = LocalDataLoader()
    
    def load_all_data(self, 
                     use_local=True, 
                     use_trending=True, 
                     use_popular=False,
                     use_bollywood=True):
        """Load data from all sources and combine"""
        
        print("\n" + "="*50)
        print("📥 LOADING DATA FROM ALL SOURCES")
        print("="*50)
        
        dataframes = []
        
        # Load local data
        if use_local:
            local_df = self.local.load_csv(LOCAL_CSV)
            if len(local_df) > 0:
                dataframes.append(local_df)
        
        # Load trending
        if use_trending:
            try:
                trending_df = self.tmdb.get_trending()
                if len(trending_df) > 0:
                    dataframes.append(trending_df)
            except Exception as e:
                print(f"   ⚠️  Error fetching trending: {str(e)}")
        
        # Load popular
        if use_popular:
            try:
                popular_df = self.tmdb.get_popular()
                if len(popular_df) > 0:
                    dataframes.append(popular_df)
            except Exception as e:
                print(f"   ⚠️  Error fetching popular: {str(e)}")
        
        # Load Bollywood
        if use_bollywood:
            try:
                bollywood_df = self.tmdb.get_bollywood()
                if len(bollywood_df) > 0:
                    dataframes.append(bollywood_df)
            except Exception as e:
                print(f"   ⚠️  Error fetching Bollywood: {str(e)}")
        
        # Combine all data
        if len(dataframes) == 0:
            print("   ❌ No data loaded!")
            return pd.DataFrame()
        
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"\n✓ Combined {len(combined_df)} records from all sources")
        
        return combined_df