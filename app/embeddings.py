"""
Embedding Generation Module
Creates embeddings using SentenceTransformer and manages FAISS indices
"""

import numpy as np
import faiss
from sklearn.neighbors import NearestNeighbors
import joblib
from sentence_transformers import SentenceTransformer
from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EMBEDDINGS_FILE,
    INDEX_FILE,
    DATAFRAME_FILE
)


class EmbeddingGenerator:
    """Generate embeddings using SentenceTransformer"""
    
    def __init__(self, model_name=EMBEDDING_MODEL):
        print(f"🤖 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"   ✓ Model loaded (dimension: {EMBEDDING_DIMENSION})")
    
    def generate(self, texts):
        """Generate embeddings for list of texts"""
        print(f"\n📊 Generating embeddings for {len(texts)} texts...")
        
        # Encode texts
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        
        # Convert to float32 (required by FAISS)
        embeddings = np.array(embeddings).astype('float32')
        
        print(f"   ✓ Generated embeddings shape: {embeddings.shape}")
        
        return embeddings


class FAISSIndexManager:
    """Manage FAISS indices"""
    
    @staticmethod
    def create_index(embeddings):
        """Create FAISS index from embeddings"""
        print(f"\n🔍 Creating FAISS index...")
        
        dimension = embeddings.shape[1]
        
        # Create index
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        print(f"   ✓ Index created with {index.ntotal} vectors")
        
        return index
    
    @staticmethod
    def save_index(index, filepath=INDEX_FILE):
        """Save FAISS index"""
        print(f"\n💾 Saving FAISS index to {filepath}")
        faiss.write_index(index, filepath)
        print(f"   ✓ Index saved")
    
    @staticmethod
    def load_index(filepath=INDEX_FILE):
        """Load FAISS index"""
        print(f"📂 Loading FAISS index from {filepath}")
        
        # ✅ FIX 1: ADD FILE CHECK (prevents crash)
        import os
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"FAISS index file missing: {filepath}. "
                "Run build_models.py and commit the file."
            )
        
        index = faiss.read_index(filepath)
        print(f"   ✓ Index loaded ({index.ntotal} vectors)")
        return index
    
    @staticmethod
    def search(index, query_embedding, k=5):
        """Search for similar embeddings"""
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = index.search(query_embedding, k)
        
        return distances[0], indices[0]


class ModelManager:
    """Manage all model files (embeddings, index, dataframe)"""
    
    @staticmethod
    def save_all(df, embeddings, index):
        """Save all model artifacts"""
        print("\n" + "="*50)
        print("💾 SAVING ALL MODELS")
        print("="*50)
        
        # Save dataframe
        print(f"Saving dataframe to {DATAFRAME_FILE}")
        joblib.dump(df, DATAFRAME_FILE)
        print(f"   ✓ Dataframe saved ({len(df)} records)")
        
        # Save embeddings
        print(f"Saving embeddings to {EMBEDDINGS_FILE}")
        np.save(EMBEDDINGS_FILE, embeddings)
        print(f"   ✓ Embeddings saved ({embeddings.shape})")
        
        # Save FAISS index
        FAISSIndexManager.save_index(index)
        
        print("✅ All models saved successfully!\n")
    
    @staticmethod
    def load_all():
        """Load all model artifacts"""
        print("\n" + "="*50)
        print("📂 LOADING ALL MODELS")
        print("="*50)
        
        try:
            # Load dataframe
            print(f"Loading dataframe from {DATAFRAME_FILE}")
            df = joblib.load(DATAFRAME_FILE)
            print(f"   ✓ Dataframe loaded ({len(df)} records)")
            
            # Load embeddings
            print(f"Loading embeddings from {EMBEDDINGS_FILE}")
            embeddings = np.load(EMBEDDINGS_FILE)
            print(f"   ✓ Embeddings loaded ({embeddings.shape})")
            
            # Load FAISS index
            # (unchanged logic, only safer inside load_index now)
            index = FAISSIndexManager.load_index()
            
            print("✅ All models loaded successfully!\n")
            
            return df, embeddings, index
        
        except FileNotFoundError as e:
            print(f"   ❌ Error: {str(e)}")
            print("   Please run build_models.py first to create models!")
            return None, None, None
        
        # ✅ FIX 2: catch unexpected FAISS/runtime crashes
        except Exception as e:
            print(f"   ❌ Unexpected error: {str(e)}")
            return None, None, None