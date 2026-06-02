"""
BUILD MODELS SCRIPT
This script runs the complete pipeline: load → clean → embed → index → save
Run this once to create your model, then it updates automatically
"""

import sys
from data_pipeline import DataPipeline
from preprocessing import DataCleaner, check_data_quality
from embeddings import EmbeddingGenerator, FAISSIndexManager, ModelManager

def main():
    """Main pipeline"""
    
    print("\n" + "="*70)
    print("🚀 NETFLIX AI RECOMMENDATION SYSTEM - MODEL BUILDER")
    print("="*70)
    
    # ==========================================
    # STEP 1: LOAD DATA
    # ==========================================
    pipeline = DataPipeline()
    
    df = pipeline.load_all_data(
        use_local=True,
        use_trending=True,
        use_popular=False,
        use_bollywood=True
    )
    
    if len(df) == 0:
        print("❌ No data loaded. Exiting.")
        sys.exit(1)
    
    # ==========================================
    # STEP 2: CLEAN DATA
    # ==========================================
    df = DataCleaner.process(df)
    
    # Print quality report
    check_data_quality(df)
    
    # ==========================================
    # STEP 3: GENERATE EMBEDDINGS
    # ==========================================
    generator = EmbeddingGenerator()
    embeddings = generator.generate(df['combined'].tolist())
    
    # ==========================================
    # STEP 4: CREATE FAISS INDEX
    # ==========================================
    index = FAISSIndexManager.create_index(embeddings)
    
    # ==========================================
    # STEP 5: SAVE EVERYTHING
    # ==========================================
    ModelManager.save_all(df, embeddings, index)
    
    # ==========================================
    # SUMMARY
    # ==========================================
    print("="*70)
    print("✅ MODEL BUILDING COMPLETE!")
    print("="*70)
    print(f"\n📊 Final Statistics:")
    print(f"   Total titles: {len(df)}")
    print(f"   Movies: {(df['type'] == 'Movie').sum()}")
    print(f"   TV Shows: {(df['type'] == 'TV Show').sum()}")
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   FAISS index vectors: {index.ntotal}")
    
    print(f"\n🎯 You can now run:")
    print(f"   streamlit run streamlit_app.py")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)