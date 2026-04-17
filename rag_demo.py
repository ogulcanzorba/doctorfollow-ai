"""
Test RAG Generation with Groq - UTF-8 output
"""
import os
import sys
import json

os.chdir(r"C:\Users\PC\Desktop\DoctorFollow.AI")
sys.path.insert(0, ".")

from retrieval.bm25 import BM25Retrieval
from retrieval.semantic import SemanticRetrieval
from retrieval.hybrid_rrf import HybridRRFRetrieval
from rag.generator import generate_answer

with open("corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)["articles"]

print(f"Loaded {len(corpus)} articles")

bm25 = BM25Retrieval(corpus)
semantic = SemanticRetrieval(corpus)
hybrid = HybridRRFRetrieval(bm25, semantic, k=60)

for query in ["What are the latest guidelines for managing type 2 diabetes?", "Acute otitis media treatment in children"]:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)
    
    results = hybrid.retrieve(query, top_k=5)
    answer = generate_answer(query, results)
    
    with open("rag_output.txt", "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Query: {query}\n")
        f.write(f"{'='*60}\n")
        f.write(answer)
        f.write("\n")

print("\nSaved to rag_output.txt")