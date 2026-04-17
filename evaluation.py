"""
Evaluation Module
Compares BM25, Semantic, and Hybrid retrieval methods using various metrics.
"""
import json
import numpy as np
from typing import Optional
from retrieval.bm25 import BM25Retrieval
from retrieval.semantic import SemanticRetrieval
from retrieval.hybrid_rrf import HybridRRFRetrieval

TEST_QUERIES = [
    "What are the latest guidelines for managing type 2 diabetes?",
    "Çocuklarda akut otitis media tedavisi nasıl yapılır?",
    "Iron supplementation dosing for anemia during pregnancy",
    "Çölyak hastalığı tanı kriterleri nelerdir?",
    "Antibiotic resistance patterns in community acquired pneumonia"
]

EXPECTED_TERMS = [
    "type 2 diabetes mellitus",
    "acute otitis media",
    "iron deficiency anemia",
    "celiac disease diagnosis",
    "community acquired pneumonia"
]

def load_corpus(path: str = "corpus.json") -> list[dict]:
    """Load corpus from JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("articles", [])

def get_relevant_docs(corpus: list[dict], query_idx: int) -> set[str]:
    """Get relevant document PMIDs for a query."""
    term = EXPECTED_TERMS[query_idx].lower()
    relevant = set()
    
    for article in corpus:
        matched = article.get("matched_terms", [])
        if any(term in str(m).lower() for m in matched):
            relevant.add(article.get("pmid"))
    
    return relevant

def calculate_ndcg(scores: list[float], relevant_set: set[str], retrieved_pmids: list[str], k: int = 5) -> float:
    """Calculate NDCG@k."""
    dcg = 0.0
    for i, pmid in enumerate(retrieved_pmids[:k], 1):
        if pmid in relevant_set:
            dcg += 1.0 / np.log2(i + 1)
    
    ideal = 0.0
    for i in range(1, min(k, len(relevant_set)) + 1):
        ideal += 1.0 / np.log2(i + 1)
    
    if ideal == 0:
        return 0.0
    
    return dcg / ideal

def calculate_mrr(relevant_set: set[str], retrieved_pmids: list[str]) -> float:
    """Calculate Mean Reciprocal Rank."""
    for i, pmid in enumerate(retrieved_pmids, 1):
        if pmid in relevant_set:
            return 1.0 / i
    return 0.0

def calculate_hit_rate(relevant_set: set[str], retrieved_pmids: list[str], k: int = 5) -> float:
    """Calculate Hit Rate@k."""
    for pmid in retrieved_pmids[:k]:
        if pmid in relevant_set:
            return 1.0
    return 0.0

def calculate_precision(relevant_set: set[str], retrieved_pmids: list[str], k: int = 5) -> float:
    """Calculate Precision@k."""
    hits = sum(1 for pmid in retrieved_pmids[:k] if pmid in relevant_set)
    return hits / k

def evaluate_method(corpus, retrieval_obj, method_name: str, top_k: int = 5):
    """Evaluate a retrieval method."""
    results = {
        "method": method_name,
        "ndcg": [],
        "mrr": [],
        "hit_rate": [],
        "precision": []
    }
    
    for i, query in enumerate(TEST_QUERIES):
        relevant_set = get_relevant_docs(corpus, i)
        
        if not relevant_set:
            continue
        
        retrieved = retrieval_obj.retrieve(query, top_k=top_k)
        retrieved_pmids = [doc.get("pmid") for doc, score in retrieved]
        
        ndcg = calculate_ndcg([], relevant_set, retrieved_pmids, top_k)
        mrr = calculate_mrr(relevant_set, retrieved_pmids)
        hit = calculate_hit_rate(relevant_set, retrieved_pmids, top_k)
        prec = calculate_precision(relevant_set, retrieved_pmids, top_k)
        
        results["ndcg"].append(ndcg)
        results["mrr"].append(mrr)
        results["hit_rate"].append(hit)
        results["precision"].append(prec)
    
    results["ndcg"] = np.mean(results["ndcg"])
    results["mrr"] = np.mean(results["mrr"])
    results["hit_rate"] = np.mean(results["hit_rate"])
    results["precision"] = np.mean(results["precision"])
    
    return results

def run_evaluation():
    """Run full evaluation."""
    corpus = load_corpus()
    print(f"Loaded {len(corpus)} articles")
    
    print("\nLoading retrieval models...")
    bm25 = BM25Retrieval(corpus)
    semantic = SemanticRetrieval(corpus)
    hybrid = HybridRRFRetrieval(bm25, semantic, k=60)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    methods = [
        (bm25, "BM25"),
        (semantic, "Semantic (e5-small)"),
        (hybrid, "Hybrid RRF (k=60)")
    ]
    
    for retrieval_obj, name in methods:
        print(f"\nEvaluating {name}...")
        results = evaluate_method(corpus, retrieval_obj, name)
        
        print(f"\n{name}:")
        print(f"  NDCG@5:      {results['ndcg']:.4f}")
        print(f"  MRR:         {results['mrr']:.4f}")
        print(f"  Hit Rate@5:   {results['hit_rate']:.4f}")
        print(f"  Precision@5: {results['precision']:.4f}")
    
    best_method = None
    best_ndcg = -1
    
    print("\n" + "="*60)
    print("BEST METHOD (by NDCG@5)")
    print("="*60)
    
    for retrieval_obj, name in methods:
        results = evaluate_method(corpus, retrieval_obj, name)
        if results['ndcg'] > best_ndcg:
            best_ndcg = results['ndcg']
            best_method = name
    
    print(f"\nBest: {best_method} with NDCG@5 = {best_ndcg:.4f}")
    
    return best_method

def demo_queries():
    """Show sample queries for each method."""
    corpus = load_corpus()
    bm25 = BM25Retrieval(corpus)
    semantic = SemanticRetrieval(corpus)
    hybrid = HybridRRFRetrieval(bm25, semantic, k=60)
    
    print("\n" + "="*60)
    print("SAMPLE QUERY RESULTS")
    print("="*60)
    
    for query in TEST_QUERIES[:2]:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        for name, retriever in [("BM25", bm25), ("Semantic", semantic), ("Hybrid", hybrid)]:
            results = retriever.retrieve(query, top_k=3)
            print(f"\n{name}:")
            for i, (doc, score) in enumerate(results, 1):
                print(f"  {i}. {doc.get('title', '')[:60]}...")


if __name__ == "__main__":
    run_evaluation()
    demo_queries()