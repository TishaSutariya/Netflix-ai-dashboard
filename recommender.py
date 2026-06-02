"""
Recommendation Engine Module
Handles searching and generating recommendations
"""

import pandas as pd
import numpy as np
from difflib import get_close_matches
from embeddings import EmbeddingGenerator, FAISSIndexManager

class RecommendationEngine:
    """Main recommendation engine"""
    
    def __init__(self, df, embeddings, index):
        """
        Initialize recommender
        
        Args:
            df: DataFrame with movie/show data
            embeddings: NumPy array of embeddings
            index: FAISS index
        """
        self.df = df
        self.embeddings = embeddings
        self.index = index
        self.embedding_generator = EmbeddingGenerator()
    
    def find_title(self, query, threshold=0.4):
        """
        Find exact or similar title match
        
        Args:
            query: Movie title to search for
            threshold: Similarity threshold (0-1)
        
        Returns:
            Matched title or None
        """
        # Exact match first
        titles = self.df['title'].astype(str).tolist()
        
        # Try exact match
        if query in titles:
            return query
        
        # Try fuzzy match
        matches = get_close_matches(
            query,
            titles,
            n=1,
            cutoff=threshold
        )
        
        return matches[0] if matches else None
    
    def get_movie_index(self, title):
        """Get the index of a movie in the dataframe"""
        try:
            idx = self.df[self.df['title'] == title].index[0]
            return idx
        except IndexError:
            return None
    
    def recommend(self, title, top_n=5):
        """
        Get recommendations for a movie
        
        Args:
            title: Movie title to get recommendations for
            top_n: Number of recommendations to return
        
        Returns:
            DataFrame with recommended movies, or None if not found
        """
        # Find the movie
        matched_title = self.find_title(title)
        
        if matched_title is None:
            return None
        
        # Get index
        movie_idx = self.get_movie_index(matched_title)
        
        if movie_idx is None:
            return None
        
        # Get embedding
        query_embedding = self.embeddings[movie_idx].reshape(1, -1)
        
        # Search FAISS
        distances, indices = FAISSIndexManager.search(
            self.index,
            query_embedding,
            k=top_n + 1  # +1 to skip the movie itself
        )
        
        # Get recommended movies (skip first which is the query movie itself)
        recommended_indices = indices[1:]
        
        # Return dataframe
        results = self.df.iloc[recommended_indices][[
            'title',
            'type',
            'listed_in',
            'description',
            'release_year'
        ]].reset_index(drop=True)
        
        # Add similarity scores
        results['similarity_score'] = 100 - (distances[1:] / 10)  # Convert to percentage
        
        return results
    
    def get_by_genre(self, genre, limit=10):
        """Get movies by genre"""
        mask = self.df['listed_in'].str.contains(genre, case=False, na=False)
        results = self.df[mask][['title', 'type', 'listed_in', 'description']].head(limit)
        return results
    
    def get_by_type(self, content_type='Movie', limit=10):
        """Get movies by type (Movie or TV Show)"""
        results = self.df[self.df['type'] == content_type][
            ['title', 'type', 'listed_in', 'description']
        ].head(limit)
        return results
    
    def search_movies(self, query, limit=10):
        """Search movies by title or description"""
        # Search in title
        title_mask = self.df['title'].str.contains(query, case=False, na=False)
        
        # Search in description
        desc_mask = self.df['description'].str.contains(query, case=False, na=False)
        
        # Combine
        results = self.df[title_mask | desc_mask][
            ['title', 'type', 'listed_in', 'description', 'release_year']
        ].head(limit)
        
        return results
    
    def get_trending(self, limit=15):
        df = self.df.copy()

        # FORCE CLEAN (MOST IMPORTANT)
        df['rating'] = df['rating'].astype(str)
        df['rating'] = df['rating'].str.replace(',', '.', regex=False)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

        # drop invalid ratings
        df = df.dropna(subset=['rating'])

        # NOW SAFE
        results = df.sort_values('rating', ascending=False).head(limit)

        return results
    
    def get_statistics(self):
        """Get statistics about the dataset"""
        stats = {
            'total_movies': len(self.df),
            'total_movies_count': (self.df['type'] == 'Movie').sum(),
            'total_shows_count': (self.df['type'] == 'TV Show').sum(),
            'genres': self.df['listed_in'].str.split(', ').explode().unique().tolist(),
        }
        
        return stats