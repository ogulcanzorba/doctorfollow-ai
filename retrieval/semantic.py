"""
Semantic Search using sentence-transformers (intfloat/multilingual-e5-small)
"""
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional

class SemanticRetrieval:
    def __init__(self, corpus: list[dict], model_name: str = "intfloat/multilingual-e5-small"):
        self.corpus = corpus
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        
        self._build_index()
    
    def _build_index(self):
        """Build embeddings index from corpus."""
        texts = []
        for article in self.corpus:
            doc_text = f"{article.get('title', '')} {article.get('abstract', '')}"
            texts.append(doc_text)
        
        print(f"Building embeddings with {self.model_name}...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Embeddings shape: {self.embeddings.shape}")
    
    def _encode_query(self, query: str) -> np.ndarray:
        """Encode query into embedding."""
        return self.model.encode(query)
    
    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """Retrieve top-k documents for query."""
        query_embedding = self._encode_query(query)
        
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.embeddings
        )[0]
        
        doc_scores = list(enumerate(similarities))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in doc_scores[:top_k]:
            if score > 0:
                results.append((self.corpus[idx], float(score)))
        
        return results
    
    def get_scores(self, query: str) -> list[float]:
        """Get raw cosine similarity scores."""
        query_embedding = self._encode_query(query)
        return cosine_similarity(
            query_embedding.reshape(1, -1),
            self.embeddings
        )[0].tolist()


def load_corpus(path: str = "corpus.json") -> list[dict]:
    """Load corpus from JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("articles", [])


def demo():
    """Demo semantic retrieval."""
    corpus = load_corpus()
    print(f"Loaded {len(corpus)} articles")
    
    semantic = SemanticRetrieval(corpus)
    
    queries = [
        "What are the latest guidelines for managing type 2 diabetes?",
        "Çocuklarda akut otitis media tedavisi nasıl yapılır?",
        "Iron supplementation dosing for anemia during pregnancy"
    ]
    
    for query in queries:
        print(f"\n=== Query: {query} ===")
        results = semantic.retrieve(query, top_k=3)
        for i, (doc, score) in enumerate(results, 1):
            print(f"{i}. [{score:.3f}] {doc.get('title', '')[:80]}...")


if __name__ == "__main__":
    demo()