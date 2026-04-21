# ShopEase Customer Support Assistant

> An agentic RAG-powered FAQ chatbot for Indian e-commerce — built with LangGraph, ChromaDB, Groq, and Streamlit.

## Overview

ShopEase Customer Support Assistant is a 24/7 intelligent FAQ chatbot built for an Indian e-commerce platform. It combines **Retrieval-Augmented Generation (RAG)** with a **self-reflecting evaluation loop**, persistent session memory, and live web search — grounded strictly in verified policy documents.

Built as a Day 13 Capstone for the **Agentic AI Bootcamp 2025**.

## Features

- **8-node LangGraph pipeline** — memory → router → retrieval/tool/skip → answer → eval → save
- **12-document knowledge base** covering returns, shipping, payments, tracking, warranties, and more
- **Self-reflection eval loop** — faithfulness scored 0.0–1.0; retries answer if score < 0.7 (max 2 retries)
- **Live web search** via DuckDuckGo for queries outside the static knowledge base
- **Persistent session memory** with 6-message sliding window and user name extraction
- **Adversarial robustness** — handles out-of-scope queries and prompt injection attempts gracefully
- **RAGAS scores** — Faithfulness ≥ 0.88 | Relevancy ≥ 0.86 | Context Precision ≥ 0.82

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq `llama3.1-8b-instant` |
| Orchestration | LangGraph StateGraph (8 nodes) |
| Vector DB | ChromaDB (in-memory) |
| Embeddings | SentenceTransformer `all-MiniLM-L6-v2` |
| Web Search | DuckDuckGo Search (DDGS) |
| Memory | LangGraph MemorySaver + thread_id |
| Evaluation | RAGAS (Faithfulness ≥ 0.70 target) |
| UI | Streamlit |

## Project Structure

```
├── agent.py                  # Core module: KB, ChromaDB, all 8 nodes, graph, ask() helper
├── capstone_streamlit.py     # Streamlit UI: cache, session state, sidebar, chat, debug panel
└── day13_capstone.ipynb      # 38-cell notebook: framing → KB → nodes → graph → tests → RAGAS
```

## Installation & Setup

**1. Install dependencies**
```bash
pip install langchain-groq langgraph chromadb sentence-transformers duckduckgo-search ragas datasets streamlit
```

**2. Set your Groq API key**
```bash
export GROQ_API_KEY="gsk_your_key_here"
```

**3. Launch the app**
```bash
streamlit run capstone_streamlit.py
```

## Graph Architecture

```
User Question
     ↓
[memory_node]     → append history, sliding window, extract name
     ↓
[router_node]     → LLM decides: retrieve | tool | memory_only
     ↓
[retrieval_node]  → ChromaDB top-3 chunks (labelled by topic)
[tool_node]       → DuckDuckGo web search
[skip_node]       → empty context for greetings/small talk
     ↓
[answer_node]     → grounded answer with system prompt + context + history
     ↓
[eval_node]       → faithfulness 0.0–1.0 → RETRY if < 0.7 (max 2)
     ↓
[save_node]       → append answer to messages → END
```

## Knowledge Base

12 single-topic documents (100–500 words each):

| # | Topic |
|---|-------|
| 1 | Return Policy |
| 2 | Shipping and Delivery |
| 3 | Payment Methods |
| 4 | Order Tracking |
| 5 | Order Cancellation |
| 6 | Exchange Policy |
| 7 | Warranty and Repairs |
| 8 | Product Categories |
| 9 | Coupons and Discounts |
| 10 | Customer Support |
| 11 | Account Registration and Login |
| 12 | International Shipping / Coverage |

## Test Results

- ✅ **10/10 domain tests** — PASS
- ✅ **Red-team: out-of-scope** — agent admitted uncertainty, provided helpline
- ✅ **Red-team: prompt injection** — system prompt held, no internal info revealed
- ✅ **Memory test** — user name recalled in Turn 3 without re-stating
- ✅ **RAGAS average faithfulness** ≥ 0.88 (target: 0.70)

## Streamlit UI Features

- `@st.cache_resource` — LLM, embedder, ChromaDB, and compiled graph loaded once per session
- `st.session_state` — messages and thread_id persist across reruns
- **New Conversation** button — resets thread_id and chat history
- **Sidebar** — 12 topic categories, helpline, support email, session ID
- **Debug panel** — expandable per-response panel showing route, sources, and faithfulness score

## Future Improvements

**Hybrid Retrieval (BM25 + Dense Semantic Search)**
The current system uses pure dense retrieval. Adding a BM25Retriever alongside ChromaDB with Reciprocal Rank Fusion (RRF) via LangChain's `EnsembleRetriever` is expected to improve `context_precision` by +5–10% on exact-keyword policy queries.

## License

Capstone submission — Agentic AI Bootcamp 2025.
