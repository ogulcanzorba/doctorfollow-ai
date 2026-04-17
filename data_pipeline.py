"""
Part 1: Data Pipeline - PubMed API
Fetches 5 most recent abstracts for each medical term.
"""
import json
import os
import time
import csv
from pathlib import Path
from Bio import Entrez
from typing import Optional

EMAIL = ""
NCBI_API_KEY = ""
Entrez.email = EMAIL
Entrez.api_key = NCBI_API_KEY

def read_medical_terms(csv_path: str = "medical_terms.csv") -> list[str]:
    terms = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            terms.append(row['term'])
    return terms

def search_pubmed(term: str, retmax: int = 5) -> list[str]:
    """Search PubMed and return PMIDs."""
    try:
        handle = Entrez.esearch(
            db="pubmed",
            term=term,
            retmax=retmax,
            sort="relevance"
        )
        result = Entrez.read(handle)
        handle.close()
        return result.get("IdList", [])
    except Exception as e:
        print(f"Error searching {term}: {e}")
        return []

def fetch_article_details(pmid: str) -> Optional[dict]:
    """Fetch article details from PubMed."""
    try:
        handle = Entrez.efetch(
            db="pubmed",
            id=pmid,
            rettype="xml",
            retmode="xml"
        )
        records = Entrez.read(handle, validate=False)
        handle.close()
        
        article = records.get("PubmedArticle", [{}])[0]
        medline_citation = article.get("MedlineCitation", {})
        article_data = medline_citation.get("Article", {})
        
        pmid_value = medline_citation.get("PMID", "")
        title = article_data.get("ArticleTitle", "")
        abstract = article_data.get("Abstract", {})
        abstract_texts = abstract.get("AbstractText", [])
        abstract_text = " ".join([str(t) for t in abstract_texts]) if abstract_texts else ""
        
        authors_list = article_data.get("AuthorList", [])
        first_author = ""
        if authors_list:
            first_author_obj = authors_list[0]
            last_name = first_author_obj.get("LastName", "")
            fore_name = first_author_obj.get("ForeName", "")
            first_author = f"{fore_name} {last_name}".strip()
        
        journal = article_data.get("Journal", {})
        journal_title = journal.get("Title", "")
        
        journal_issue = journal.get("JournalIssue", {})
        pub_date = journal_issue.get("PubDate", {})
        year = pub_date.get("Year", "")
        
        article_id_list = article.get("ArticleIdList", [])
        doi = ""
        for aid in article_id_list:
            if aid.get("IdType") == "doi":
                doi = aid.get("Id", "")
                break
        
        return {
            "pmid": pmid_value,
            "title": title,
            "abstract": abstract_text,
            "authors": first_author,
            "journal": journal_title,
            "year": year,
            "doi": doi
        }
    except Exception as e:
        print(f"Error fetching PMID {pmid}: {e}")
        return None

def run_pipeline(terms: list[str] = None, retmax: int = 5):
    """Run the data pipeline."""
    if terms is None:
        terms = read_medical_terms()
    
    print(f"Processing {len(terms)} terms...")
    
    all_articles = []
    term_article_map = {}
    seen_pmids = set()
    errors = []
    
    for term in terms:
        print(f"Searching: {term}")
        pmids = search_pubmed(term, retmax)
        
        term_article_map[term] = []
        
        for pmid in pmids:
            time.sleep(0.35)
            
            article = fetch_article_details(pmid)
            if article is None:
                errors.append({"term": term, "pmid": pmid, "error": "fetch failed"})
                continue
            
            term_article_map[term].append(pmid)
            
            if pmid in seen_pmids:
                continue
            
            seen_pmids.add(pmid)
            article["matched_terms"] = [term]
            all_articles.append(article)
        
        print(f"  Found {len(pmids)} articles, {len([p for p in pmids if p in seen_pmids])} duplicates")
    
    for article in all_articles:
        for term, pmids in term_article_map.items():
            if article["pmid"] in pmids:
                if "matched_terms" not in article:
                    article["matched_terms"] = []
                if term not in article["matched_terms"]:
                    article["matched_terms"].append(term)
    
    output = {
        "terms_processed": len(terms),
        "unique_articles": len(all_articles),
        "duplicates_removed": sum(len(v) for v in term_article_map.values()) - len(all_articles),
        "errors": errors,
        "articles": all_articles
    }
    
    print(f"\n=== Summary ===")
    print(f"Terms processed: {output['terms_processed']}")
    print(f"Unique articles: {output['unique_articles']}")
    print(f"Duplicates removed: {output['duplicates_removed']}")
    print(f"Errors: {len(errors)}")
    
    with open("corpus.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved to corpus.json")
    return output

if __name__ == "__main__":
    run_pipeline()