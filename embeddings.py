"""
Embedding Generation + FAISS Manager (PRODUCTION SAFE VERSION)
Fixes:
- Missing file crashes
- Streamlit Cloud deployment issues
- FAISS loading errors
"""

import numpy as np
import faiss
import joblib
import os
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    EMBEDDINGS_FILE,
    INDEX_FILE,
    DATAFRAME_FILE
)


class EmbeddingGenerator:
    """Generate embeddings using SentenceTransformer"""

    def __init__(self, model_name=EMBEDDING_MODEL):
        print(f"🤖 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("   ✓ Model loaded successfully")

    def generate(self, texts):
        """Generate embeddings for list of texts"""

        print(f"\n📊 Generating embeddings for {len(texts)} texts...")

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )

        embeddings = np.array(embeddings).astype("float32")

        print(f"   ✓ Embeddings shape: {embeddings.shape}")

        return embeddings


class FAISSIndexManager:
    """Manage FAISS index safely"""

    @staticmethod
    def create_index(embeddings):
        """Create FAISS index"""

        print("\n🔍 Creating FAISS index...")

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        print(f"   ✓ Index created with {index.ntotal} vectors")

        return index

    @staticmethod
    def save_index(index, filepath=INDEX_FILE):
        """Save FAISS index"""

        print(f"\n💾 Saving FAISS index to {filepath}")
        faiss.write_index(index, filepath)
        print("   ✓ Index saved successfully")

    @staticmethod
    def load_index(filepath=INDEX_FILE):
        """Load FAISS index safely"""

        print(f"\n📂 Loading FAISS index from {filepath}")

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"""
❌ FAISS INDEX FILE NOT FOUND

Expected file: {filepath}

👉 Fix:
1. Run build_models.py locally
2. Ensure index file is created
3. Commit it to GitHub

Streamlit Cloud does NOT generate this automatically.
"""
            )

        index = faiss.read_index(filepath)

        print(f"   ✓ Index loaded ({index.ntotal} vectors)")

        return index

    @staticmethod
    def search(index, query_embedding, k=5):
        """Search similar vectors"""

        query_embedding = query_embedding.reshape(1, -1).astype("float32")

        distances, indices = index.search(query_embedding, k)

        return distances[0], indices[0]


class ModelManager:
    """Load and manage all ML artifacts"""

    @staticmethod
    def save_all(df, embeddings, index):
        """Save everything locally"""

        print("\n" + "=" * 50)
        print("💾 SAVING ALL MODELS")
        print("=" * 50)

        # Save dataframe
        print(f"Saving dataframe → {DATAFRAME_FILE}")
        joblib.dump(df, DATAFRAME_FILE)
        print(f"   ✓ Saved {len(df)} rows")

        # Save embeddings
        print(f"Saving embeddings → {EMBEDDINGS_FILE}")
        np.save(EMBEDDINGS_FILE, embeddings)
        print(f"   ✓ Saved shape {embeddings.shape}")

        # Save FAISS index
        FAISSIndexManager.save_index(index)

        print("\n✅ ALL MODELS SAVED SUCCESSFULLY!\n")

    @staticmethod
    def load_all():
        """Load everything safely (Streamlit Cloud ready)"""

        print("\n" + "=" * 50)
        print("📂 LOADING ALL MODELS")
        print("=" * 50)

        # ---------------------------
        # LOAD DATAFRAME
        # ---------------------------
        if not os.path.exists(DATAFRAME_FILE):
            raise FileNotFoundError(
                f"Missing dataframe file: {DATAFRAME_FILE}"
            )

        df = joblib.load(DATAFRAME_FILE)
        print(f"   ✓ DataFrame loaded ({len(df)} rows)")

        # ---------------------------
        # LOAD EMBEDDINGS
        # ---------------------------
        if not os.path.exists(EMBEDDINGS_FILE):
            raise FileNotFoundError(
                f"Missing embeddings file: {EMBEDDINGS_FILE}"
            )

        embeddings = np.load(EMBEDDINGS_FILE)
        print(f"   ✓ Embeddings loaded {embeddings.shape}")

        # ---------------------------
        # LOAD FAISS INDEX
        # ---------------------------
        index = FAISSIndexManager.load_index()

        print("\n✅ ALL MODELS LOADED SUCCESSFULLY!\n")

        return df, embeddings, index