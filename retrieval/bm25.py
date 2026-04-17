"""
BM25 Search using rank_bm25
"""
import json
import math
import re
from rank_bm25 import BM25Okapi
from typing import Optional

def simple_tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric."""
    if not text:
        return []
    text = text.lower()
    tokens = re.findall(r'[a-z]+', text)
    return tokens

class BM25Retrieval:
    def __init__(self, corpus: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.documents = []
        self.tokenized_corpus = []
        
        self._build_index()
    
    def _build_index(self):
        """Build BM25 index from corpus."""
        self.documents = []
        self.tokenized_corpus = []
        
        for article in self.corpus:
            doc_text = f"{article.get('title', '')} {article.get('abstract', '')}"
            self.documents.append(doc_text)
            tokens = simple_tokenize(doc_text)
            self.tokenized_corpus.append(tokens)
        
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
    
    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """Retrieve top-k documents for query."""
        query_tokens = simple_tokenize(query)
        
        if not query_tokens:
            return []
        
        scores = self.bm25.get_scores(query_tokens)
        
        doc_scores = list(enumerate(scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in doc_scores[:top_k]:
            if score > 0:
                results.append((self.corpus[idx], score))
        
        return results
    
    def get_scores(self, query: str) -> list[float]:
        """Get raw scores for query."""
        query_tokens = simple_tokenize(query)
        return self.bm25.get_scores(query_tokens).tolist()


def load_corpus(path: str = "corpus.json") -> list[dict]:
    """Load corpus from JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("articles", [])


def demo():
    """Demo BM25 retrieval."""
    corpus = load_corpus()
    print(f"Loaded {len(corpus)} articles")
    
    bm25 = BM25Retrieval(corpus, k1=1.5, b=0.75)
    
    queries = [
        "What are the latest guidelines for managing type 2 diabetes?",
        "Çocuklarda akut otitis media tedavisi nasıl yapılır?",
        "Iron supplementation dosing for anemia during pregnancy"
    ]
    
    for query in queries:
        print(f"\n=== Query: {query} ===")
        results = bm25.retrieve(query, top_k=3)
        for i, (doc, score) in enumerate(results, 1):
            print(f"{i}. [{score:.3f}] {doc.get('title', '')[:80]}...")


if __name__ == "__main__":
    demo()