"""
Main test runner for Medical RAG System
Run all parts and demonstrate functionality.
"""
import json
import data_pipeline
import evaluation
from retrieval.bm25 import BM25Retrieval
from retrieval.semantic import SemanticRetrieval
from retrieval.hybrid_rrf import HybridRRFRetrieval
from rag.generator import generate_answer

print("="*60)
print("DoctorFollow AI - Medical RAG System")
print("="*60)

print("\n[Step 1] Running Data Pipeline...")
corpus_data = data_pipeline.run_pipeline()
print(f"✓ Retrieved {corpus_data['unique_articles']} unique articles")

print("\n[Step 2] Building Retrieval Index...")
corpus = corpus_data["articles"]

print("\n--- Running Evaluation ---")
best_method = evaluation.run_evaluation()

print("\n[Step 3] Testing RAG Generation...")
bm25 = BM25Retrieval(corpus)

queries = [
    "What are the latest guidelines for managing type 2 diabetes?",
    "Çocuklarda akut otitis media tedavisi nasıl yapılır?"
]

for query in queries:
    print(f"\nQuery: {query}")
    results = bm25.retrieve(query, top_k=3)
    
    print("Retrieved docs:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"  {i}. PMID:{doc.get('pmid')} - {doc.get('title', '')[:50]}...")
    
    answer = generate_answer(query, results)
    print(f"\nAnswer:\n{answer}")

print("\n" + "="*60)
print("DONE!")
print("="*60)