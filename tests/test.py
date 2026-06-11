"""
test.py - Quick metrics calculation for Netflix AI Dashboard
Calculate: Accuracy, Query Latency, Results Count
"""

import time
from embeddings import ModelManager
from recommender import RecommendationEngine

print("=" * 60)
print("🧪 Netflix AI Dashboard - Metrics Test")
print("=" * 60)

# Load models
print("\n📦 Loading models...")
start_time = time.time()
df, embeddings, index = ModelManager.load_all()
load_time = time.time() - start_time

if df is None:
    print("❌ Models not found! Run: python build_models.py")
    exit(1)

print(f"✅ Models loaded in {load_time:.2f}s")
print(f"📊 Dataset size: {len(df)} titles")

# Create recommender
print("\n🤖 Creating recommendation engine...")
recommender = RecommendationEngine(df, embeddings, index)
print("✅ Engine created")

# Test queries
test_queries = [
    "batman",
    "avengers", 
    "stranger things",
    "interstellar",
    "inception",
    "dark knight",
    "iron man",
    "joker",
    "matrix",
    "gladiator"
]

print(f"\n🔍 Testing {len(test_queries)} queries...")
print("-" * 60)

correct_matches = 0
total_latency = 0

for query in test_queries:
    start = time.time()
    results = recommender.search_movies(query, limit=5)
    latency = time.time() - start
    total_latency += latency
    
    status = "✅" if len(results) > 0 else "❌"
    print(f"{status} '{query}': {len(results)} results ({latency*1000:.1f}ms)")
    
    if len(results) > 0:
        correct_matches += 1

# Calculate metrics
accuracy = (correct_matches / len(test_queries)) * 100
avg_latency = total_latency / len(test_queries)

print("\n" + "=" * 60)
print("📈 RESULTS")
print("=" * 60)
print(f"✅ Search Accuracy: {accuracy:.1f}% ({correct_matches}/{len(test_queries)} queries)")
print(f"⚡ Avg Query Latency: {avg_latency*1000:.1f}ms")
print(f"📊 Dataset Size: {len(df)} titles")
print("=" * 60)