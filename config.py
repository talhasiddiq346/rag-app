"""
config.py - All settings for the RAG app live in one place.

Change values here instead of hunting through the code.
"""

import os

# python-dotenv is optional: if installed, it loads GROQ_API_KEY from a local .env file.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ---------------- Embedding model (Stage 2) ----------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # free, open-source, 384-dimensional vectors

# ---------------- Chunking (Stage 1) ----------------
DEFAULT_CHUNK_SIZE = 150      # words per chunk (value from the assignment)
DEFAULT_CHUNK_OVERLAP = 30    # words shared between neighbouring chunks

# ---------------- Retrieval (Stage 4) ----------------
DEFAULT_TOP_K = 3
MIN_TOP_K = 1
MAX_TOP_K = 5
NEIGHBOR_WINDOW = 1          # also send the chunk before/after each hit (keeps sections whole)
CONTEXT_WORD_BUDGET = 2400   # max words of context sent to the LLM (Groq free tier ~8k tokens/min)
MAX_SUBQUERIES = 4           # multi-query retrieval: extra search queries written by the LLM
QUERY_MODEL = "openai/gpt-oss-20b"   # small, fast model used only to write those queries

# ---------------- Generation (Stage 5) ----------------
# The PDF used "llama-3.3-70b-versatile" and "llama-3.1-8b-instant". Groq has since
# retired both (the PDF's "model_decommissioned" watch-out). These are open-weight
# chat models that were live on Groq's free tier when this project was built.
# Check https://console.groq.com/docs/models if one stops working.
GROQ_MODELS = [
    "openai/gpt-oss-120b",   # default: largest, best answers
    "openai/gpt-oss-20b",    # smaller and faster
    "qwen/qwen3.8-27b",      # alternative open-weight model
]
DEFAULT_MODEL = GROQ_MODELS[0]
TEMPERATURE = 0.1   # low = factual, sticks to the context

SYSTEM_PROMPT = (
    "You are a document question-answering assistant. Answer using only the "
    "provided context. If the information is unavailable in the context, "
    "explicitly say that you do not know. Never invent facts. When useful, "
    "mention which source file the answer came from."
)

# ---------------- Files ----------------
SUPPORTED_EXTENSIONS = ["pdf", "docx", "txt"]


def get_api_key_from_environment() -> str:
    """
    Look for the Groq key WITHOUT hard-coding it:
      1. Environment variable / .env file  (local use)
      2. Streamlit secrets                 (Hugging Face Spaces / Streamlit Cloud)
    Returns "" if nothing is found. The key is never printed.
    """
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st

        return str(st.secrets.get("GROQ_API_KEY", "")).strip()
    except Exception:
        # No secrets.toml present - that is fine.
        return ""
