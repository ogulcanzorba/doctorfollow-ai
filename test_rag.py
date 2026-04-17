"""
Test RAG Generation with Groq
"""
import os
import sys

from retrieval.bm25 import BM25Retrieval
from retrieval.semantic import SemanticRetrieval
from retrieval.hybrid_rrf import HybridRRFRetrieval
from rag.generator import generate_answer
import json

os.chdir(r"C:\Users\PC\Desktop\DoctorFollow.AI")

with open("corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)["articles"]

print(f"Loaded {len(corpus)} articles")

bm25 = BM25Retrieval(corpus)
semantic = SemanticRetrieval(corpus)
hybrid = HybridRRFRetrieval(bm25, semantic, k=60)

queries = [
    "What are the latest guidelines for managing type 2 diabetes?",
    "Çocuklarda akut otitis media tedavisi nasıl yapılır?"
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)
    
    results = hybrid.retrieve(query, top_k=5)
    
    print(f"\nRetrieved {len(results)} documents:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"  {i}. PMID:{doc.get('pmid')} - {doc.get('title', '')[:60]}...")
    
    answer = generate_answer(query, results)
    print(f"\n--- ANSWER ---\n{answer}\n")