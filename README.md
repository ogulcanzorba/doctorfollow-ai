# DoctorFollow AI — Medical RAG System

A medical Retrieval-Augmented Generation system serving Turkish-speaking doctors querying English medical literature.

## Overview

This system fetches medical articles from PubMed, builds three retrieval methods (BM25, Semantic, Hybrid RRF), and generates cited answers via LLM.

## Setup & Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create `.env` file:

```bash
GROQ_API_KEY=your_groq_api_key
```

Or export directly:

```bash
export GROQ_API_KEY="gsk_..."
```

### 3. Run Pipeline

```bash
python data_pipeline.py
```

This fetches 5 most recent abstracts for each of the 10 medical terms and saves to `corpus.json`.

### 4. Run Tests

```bash
python tests.py
```

This runs:
- Data pipeline
- Evaluation
- Demo queries with RAG generation

## Approach

### Part 1 — Data Pipeline

- **Tool**: Biopython (Entrez E-utilities)
- **API**: PubMed ESearch + EFetch
- **Rate limiting**: 3 req/sec with NCBI API key + sleep(0.35)
- **Extraction**: PMID, Title, Abstract, Authors, Journal, Year, DOI
- **Deduplication**: By PMID across all terms

### Part 2 — Retrieval Methods

#### A. BM25 Search

- **Library**: rank_bm25 (BM25Okapi)
- **Parameters**: k1=1.5, b=0.75

#### B. Semantic Search

- **Model**: intfloat/multilingual-e5-small
- **Justification**: Best speed-accuracy balance, excellent Turkish support, 47MB

#### C. Hybrid RRF

- **Method**: Reciprocal Rank Fusion
- **k**: 60 (default)

### Part 3 — RAG Generation

- **Provider**: Groq
- **Model**: llama-3.3-70b-versatile
- **System Prompt**: Answer from context only, cite by PMID

## BM25 Analysis

### k1 Parameter (default: 1.5)

Controls term frequency saturation. Higher k1 = more aggressive term matching.

- **k1 = 0.5**: Conservative - only exact matches get high scores
- **k1 = 2.0**: Aggressive - common terms also rank high

### b Parameter (default: 0.75)

Controls document length normalization.

- **b = 0**: No length normalization
- **b = 1**: Full length normalization

## RRF Analysis

### What does k do?

- **k = 60 (default)**: Balanced fusion
- **k = 0**: Only #1 result gets reward (aggressive)
- **k = 1000**: More distributed reward (conservative)

### Why rank position instead of raw scores?

BM25 scores and cosine similarity are on different scales (unbounded vs 0-1). Rank-based fusion normalizes them implicitly.

## Evaluation

### Metrics Chosen

| Metric | Rationale |
|-------|----------|
| **NDCG@5** | Standard IR metric - handles rank order |
| **MRR** | First relevant document position |
| **Hit Rate@5** | At least one relevant found |
| **Precision@5** | Proportion of relevant in top-5 |

### Results

Run `python evaluation.py` to see comparative results.

## Hardest Problem

**[Answer]**

Handling Turkish queries against English corpus was challenging. Turkish is morphologically rich and different from English. Semantic search with multilingual-e5-small handles this better than BM25 because:
1. Dense embeddings capture semantic meaning, not just lexical match
2. Turkish query expansion to English improved results

## Bonus — Improvements Made

1. **Query Translation**: Turkish → English expansion
2. **Context Window**: Combined title + abstract for richer representation
3. **RRF with k=60**: Balanced fusion of lexical and semantic

---

## Scenario Question

**Your team needs to benchmark a 70B open-source LLM for medical QA. Your usual GPU provider doesn't have L40S available today. Your manager is busy all day. Results needed by end of week. What do you do?**

### Solution Options

| Platform | GPU | Pros | Cons |
|----------|-----|------|------|
| **RunPod** | A100/H100 | Instant, pay-per-use, cheap | Requires setup |
| **Liquid** | A100 | Minute billing | Limited regions |
| **Cerebras** | H100 | Fastest inference | Model compatibility |

### Chosen: RunPod

1. **Quick start**: `pod run` with custom template
2. **Cost**: ~$0.50/hr for A100
3. **Setup in <30 min**: HF token, medical dataset download

### Alternative if time-critical

- Use cloud APIs (Anthropic, OpenAI) for quick validation
- Then run open-source for full benchmark

---

## Files

```
.
├── data_pipeline.py      # PubMed API fetcher
├── retrieval/
│   ├── bm25.py      # BM25 retrieval
│   ├── semantic.py   # Semantic retrieval
│   └── hybrid_rrf.py # Hybrid RRF
├── rag/
│   └── generator.py  # LLM generation
├── corpus.json       # Generated corpus
├── evaluation.py   # Metrics
└── requirements.txt
```

## AI Usage

- Used web search for documentation lookup
- Used OpenCode for code structure suggestions
- Most of the code written by AI, suggestions and fixes made by developer.