"""
RAG Generator using Groq API
Generates cited answers from retrieved context.
"""
import os
import json
from groq import Groq
from typing import Optional

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

SYSTEM_PROMPT = """Sen bir tıbbi asistanst. 
Yanıtlarını YALNIZCA verilen bağlamdan oluştur.
Her zaman kaynakları PMID numarası ile cite et.
Eğer verilen bağlamda yeterli bilgi yoksa, 'Bu konuda yeterli bilgi bulunmamaktadır.' de.
Türkçe yanıt ver."""

def create_client(api_key: str = None) -> Groq:
    """Create Groq client."""
    key = api_key or GROQ_API_KEY
    return Groq(api_key=key)

def format_context(articles: list[dict]) -> str:
    """Format articles as context for LLM."""
    context_parts = []
    for i, article in enumerate(articles, 1):
        title = article.get("title", "No title")
        abstract = article.get("abstract", "")
        pmid = article.get("pmid", "")
        year = article.get("year", "")
        journal = article.get("journal", "")
        
        context_parts.append(
            f"[{i}] {title}\n"
            f"PMID: {pmid} | Year: {year} | Journal: {journal}\n"
            f"Abstract: {abstract}\n"
        )
    
    return "\n---\n".join(context_parts)

def build_user_prompt(query: str, context: str) -> str:
    """Build user prompt with query and context."""
    return f"""Soru: {query}

Makaleler:
{context}

Lütfen yukarıdaki makalelere dayanarak soruyu yanıtla. Her kaynak için PMID numarasını belirt."""

def generate_answer(
    query: str,
    retrieved_docs: list[tuple[dict, float]],
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.3,
    max_tokens: int = 1024
) -> str:
    """
    Generate answer using Groq LLM.
    
    Args:
        query: User query
        retrieved_docs: List of (article, score) tuples
        model: Groq model name
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
    
    Returns:
        Generated answer string
    """
    client = create_client()
    
    if not retrieved_docs:
        return "Verilen bağlamda soruyla ilgili makale bulunamadı."
    
    articles = [doc for doc, score in retrieved_docs]
    context = format_context(articles[:5])
    
    user_prompt = build_user_prompt(query, context)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Hata oluştu: {str(e)}"


def demo():
    """Demo RAG generation."""
    import sys
    sys.path.insert(0, "..")
    from retrieval.bm25 import BM25Retrieval, load_corpus
    
    corpus = load_corpus()
    print(f"Loaded {len(corpus)} articles")
    
    bm25 = BM25Retrieval(corpus)
    
    queries = [
        "What are the latest guidelines for managing type 2 diabetes?",
        "Çocuklarda akut otitis media tedavisi nasıl yapılır?"
    ]
    
    for query in queries:
        print(f"\n=== Query: {query} ===")
        
        results = bm25.retrieve(query, top_k=5)
        
        print(f"Retrieved {len(results)} documents")
        
        answer = generate_answer(query, results)
        
        print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    demo()