"""
Hybrid Retrieval using Reciprocal Rank Fusion (RRF)
Combines BM25 and Semantic search results.
"""
import json
import numpy as np
from typing import Optional

class HybridRRFRetrieval:
    def __init__(self, bm25_retrieval, semantic_retrieval, k: int = 60):
        self.bm25 = bm25_retrieval
        self.semantic = semantic_retrieval
        self.k = k
    
    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """
        Combine BM25 and Semantic results using RRF.
        
        RRF Score = Σ 1 / (k + rank(d))
        - k=60 (default): balanced fusion
        - k=0: only #1 result gets reward
        - k=1000: more distributed reward
        """
        bm25_results = self.bm25.retrieve(query, top_k=50)
        semantic_results = self.semantic.retrieve(query, top_k=50)
        
        rrf_scores = {}
        doc_map = {}
        
        for rank, (doc, score) in enumerate(bm25_results, 1):
            pmid = doc.get('pmid', '')
            if pmid:
                rrf_scores[pmid] = rrf_scores.get(pmid, 0) + 1.0 / (self.k + rank)
                doc_map[pmid] = doc
        
        for rank, (doc, score) in enumerate(semantic_results, 1):
            pmid = doc.get('pmid', '')
            if pmid:
                rrf_scores[pmid] = rrf_scores.get(pmid, 0) + 1.0 / (self.k + rank)
                doc_map[pmid] = doc
        
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for pmid, score in sorted_docs[:top_k]:
            if pmid in doc_map:
                results.append((doc_map[pmid], score))
        
        return results


def load_corpus(path: str = "corpus.json") -> list[dict]:
    """Load corpus from JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("articles", [])


def demo():
    """Demo hybrid retrieval."""
    from bm25 import BM25Retrieval
    from semantic import SemanticRetrieval
    
    corpus = load_corpus()
    print(f"Loaded {len(corpus)} articles")
    
    bm25 = BM25Retrieval(corpus)
    semantic = SemanticRetrieval(corpus)
    hybrid = HybridRRFRetrieval(bm25, semantic, k=60)
    
    queries = [
        "What are the latest guidelines for managing type 2 diabetes?",
        "Çocuklarda akut otitis media tedavisi nasıl yapılır?",
        "Iron supplementation dosing for anemia during pregnancy"
    ]
    
    for query in queries:
        print(f"\n=== Query: {query} ===")
        results = hybrid.retrieve(query, top_k=3)
        for i, (doc, score) in enumerate(results, 1):
            print(f"{i}. [{score:.3f}] {doc.get('title', '')[:80]}...")


if __name__ == "__main__":
    demo()