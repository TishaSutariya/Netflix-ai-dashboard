"""
Data Preprocessing & Validation Module
Handles cleaning, NULL handling, and data validation
"""

import pandas as pd
import numpy as np
from config import (
    REQUIRED_COLUMNS,
    MIN_DESCRIPTION_LENGTH,
    BOLLYWOOD_KEYWORDS
)

class DataValidator:
    """Validates and cleans Netflix data"""
    
    @staticmethod
    def validate_row(row):
        """Check if a row has all required data"""
        try:
            # Check required columns exist and not null
            for col in REQUIRED_COLUMNS:
                if col not in row or pd.isna(row[col]):
                    return False, f"Missing {col}"
            
            # Validate title
            if not isinstance(row['title'], str) or len(str(row['title']).strip()) == 0:
                return False, "Invalid title"
            
            # Validate description length
            description = str(row['description']).strip()
            if len(description) < MIN_DESCRIPTION_LENGTH:
                return False, f"Description too short (< {MIN_DESCRIPTION_LENGTH} chars)"
            
            # Validate type
            valid_types = ["Movie", "TV Show"]
            if row['type'] not in valid_types:
                return False, f"Invalid type: {row['type']}"
            
            return True, "Valid"
        
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def clean_dataframe(df):
        """Clean and validate entire dataframe"""
        print(f"📊 Starting with {len(df)} rows")
        
        # Drop rows with NULL in critical columns
        df = df.dropna(subset=REQUIRED_COLUMNS)
        print(f"✓ After NULL removal: {len(df)} rows")
        
        # Remove duplicates by title
        df = df.drop_duplicates(subset=['title'], keep='first')
        print(f"✓ After duplicate removal: {len(df)} rows")
        
        # Validate each row
        valid_rows = []
        invalid_count = 0
        
        for idx, row in df.iterrows():
            is_valid, msg = DataValidator.validate_row(row)
            if is_valid:
                valid_rows.append(row)
            else:
                invalid_count += 1
        
        if invalid_count > 0:
            print(f"⚠️  Removed {invalid_count} invalid rows")
        
        df = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"✓ Final cleaned dataset: {len(df)} rows")
        
        return df
    
    @staticmethod
    def fill_null_values(df):
        """Fill remaining NULL values with defaults"""
        df = df.copy()
        
        # Fill missing descriptions with title
        df['description'] = df['description'].fillna(df['title'])
        
        # Fill missing genres with "Unknown"
        df['listed_in'] = df['listed_in'].fillna("Unknown")
        
        # Fill missing release_year with 2024
        if 'release_year' in df.columns:
            df['release_year'] = df['release_year'].fillna(2024).astype(int)
        
        return df
    
    @staticmethod
    def create_combined_text(df):
        """Create combined text for embeddings"""
        df = df.copy()
        
        # Combine title + genre + description
        df["combined"] = (
            df["title"].astype(str) + " " +
            df["listed_in"].astype(str) + " " +
            df["description"].astype(str)
        )
        
        return df
    
    @staticmethod
    def get_bollywood_movies(df):
        """Filter Bollywood movies from dataframe"""
        mask = df["listed_in"].str.lower().str.contains(
            "|".join(BOLLYWOOD_KEYWORDS), 
            na=False
        )
        return df[mask]


class DataCleaner:
    """Main cleaner that orchestrates the cleaning process"""
    
    @staticmethod
    def process(df):
        """Full processing pipeline"""
        print("\n" + "="*50)
        print("🧹 CLEANING DATA")
        print("="*50)
        
        # Step 1: Validate and clean
        df = DataValidator.clean_dataframe(df)
        
        # Step 2: Fill NULL values
        df = DataValidator.fill_null_values(df)
        
        # Step 3: Create combined text for embeddings
        df = DataValidator.create_combined_text(df)
        
        print("✅ Data cleaning complete!\n")
        
        return df


def check_data_quality(df):
    """Print data quality statistics"""
    print("\n📈 DATA QUALITY REPORT:")
    print(f"  Total records: {len(df)}")
    print(f"  Missing titles: {df['title'].isna().sum()}")
    print(f"  Missing descriptions: {df['description'].isna().sum()}")
    print(f"  Missing genres: {df['listed_in'].isna().sum()}")
    print(f"  Movie count: {(df['type'] == 'Movie').sum()}")
    print(f"  TV Show count: {(df['type'] == 'TV Show').sum()}")
    
    # Sample genres
    if 'listed_in' in df.columns:
        genres = df['listed_in'].str.split(", ").explode().unique()
        print(f"  Unique genres: {len(genres)}")