"""
app.py - Streamlit user interface for the Open-Source RAG Application.

Run with:   streamlit run app.py
"""

import streamlit as st

import config
from document_processor import process_uploaded_files
from rag_engine import RAGError, VectorStore, expand_queries, generate_answer, load_embedder, plan_retrieval

st.set_page_config(page_title="Open-Source RAG Application", page_icon="📚", layout="wide")

SEARCH_MODES = {
    "Auto (detects comparison questions)": "auto",
    "Search all documents together": "all",
    "Compare documents side by side": "compare",
}


# ---------- Stage 2: load the embedding model ONCE (cached across reruns) ----------
@st.cache_resource(show_spinner="Loading the embedding model (first run downloads ~90 MB)...")
def get_embedder():
    return load_embedder()


# ---------- Session state (remembers things between button clicks) ----------
DEFAULT_STATE = {
    "store": None,          # VectorStore once at least one document is indexed
    "last_result": None,    # last answer + sources, so they stay on screen
    "uploader_key": 0,      # changing this empties the file uploader widget
    "kb_version": 0,        # changes whenever the document list changes (resets the filter)
    "flash": [],            # messages to show after a rerun: [(kind, text)]
}
for key, value in DEFAULT_STATE.items():
    st.session_state.setdefault(key, value)


def clear_knowledge_base():
    """Remove the index, chunks, answers and uploaded files from this session."""
    st.session_state.store = None
    st.session_state.last_result = None
    st.session_state.uploader_key += 1
    st.session_state.kb_version += 1
    st.session_state.pop("question", None)


def remove_document(name: str):
    store = st.session_state.store
    store.remove_document(name)
    if not store.sources:
        st.session_state.store = None
    st.session_state.kb_version += 1
    st.session_state.last_result = None


# =============================== SIDEBAR ===============================
with st.sidebar:
    st.header("⚙️ Settings")

    env_key = config.get_api_key_from_environment()
    typed_key = st.text_input(
        "Groq API key",
        type="password",
        placeholder="Loaded from environment" if env_key else "gsk_...",
        help="Get a free key at console.groq.com → API Keys. It is never stored or displayed.",
    ).strip()
    api_key = typed_key or env_key
    if typed_key:
        st.caption("🔑 Using the key you entered.")
    elif env_key:
        st.caption("🔑 Using GROQ_API_KEY from environment / secrets.")
    else:
        st.caption("⚠️ No API key yet.")

    model = st.selectbox("Groq model", config.GROQ_MODELS, index=0)

    st.subheader("Retrieval")
    mode = SEARCH_MODES[st.radio(
        "Search mode", list(SEARCH_MODES),
        help="Comparison mode retrieves the best chunks from EACH document, so every "
             "document is represented in the context sent to the LLM.",
    )]
    top_k = st.slider("Chunks to retrieve (k)", config.MIN_TOP_K, config.MAX_TOP_K, config.DEFAULT_TOP_K,
                      help="When comparing, k chunks are retrieved from EACH document.")

    st.subheader("Chunking")
    chunk_size = st.number_input("Chunk size (words)", min_value=20, max_value=1000,
                                 value=config.DEFAULT_CHUNK_SIZE, step=10)
    chunk_overlap = st.number_input("Chunk overlap (words)", min_value=0, max_value=500,
                                    value=config.DEFAULT_CHUNK_OVERLAP, step=5)
    overlap_ok = chunk_overlap < chunk_size
    store = st.session_state.store
    if not overlap_ok:
        st.error("Overlap must be smaller than the chunk size.")
    elif store and store.settings != (chunk_size, chunk_overlap):
        st.info("Chunk settings changed - click 'Add to knowledge base' to re-chunk all documents.")

    st.divider()
    st.button("🗑️ Clear knowledge base", on_click=clear_knowledge_base,
              help="Removes the index, chunks, answer and uploaded files from this session.")


# =============================== MAIN PAGE ===============================
st.title("📚 Open-Source RAG Application")
st.markdown("**Talha Siddiq** (b04-0923-0038) &nbsp;·&nbsp; **Ajwad Muhammad Waseem** (b04-0923-000004)")
st.caption("CS4106 - Foundations of Generative AI · Groq · Sentence-Transformers · FAISS · Streamlit")

with st.expander("How does RAG work?"):
    st.markdown(
        """
1. **Chunk** - documents are split into small overlapping passages.
2. **Embed** - each chunk becomes a 384-number vector using `all-MiniLM-L6-v2`.
3. **Index** - the vectors are stored in a **FAISS** search index.
4. **Retrieve** - your question is embedded too, and the top-*k* closest chunks are found.
   For **comparison** questions, the top-*k* chunks are taken from **each** document.
5. **Generate** - a **Groq** LLM answers using *only* those chunks as context.

The answer comes from **your documents**, not the model's memory. If the answer
isn't in them, the app should say it doesn't know.
"""
    )

# ---------------- Step 1: upload & add ----------------
st.subheader("1. Upload documents")
uploaded = st.file_uploader(
    "Upload PDF / DOCX / TXT files - you can add more files later, one or many at a time",
    type=config.SUPPORTED_EXTENSIONS,
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}",
)

for kind, text in st.session_state.flash:      # messages from the previous click
    getattr(st, kind)(text)
st.session_state.flash = []

if st.button("➕ Add to knowledge base", type="primary", disabled=not overlap_ok):
    if not uploaded:
        st.warning("Please upload at least one document first.")
    else:
        messages = []
        with st.spinner("Reading files, chunking, embedding and indexing..."):
            results = process_uploaded_files(uploaded)
            good = [r for r in results if not r["error"]]
            messages += [("warning", f"**{r['filename']}** skipped - {r['error']}") for r in results if r["error"]]
            if not good and not store:
                messages.append(("error", "None of the uploaded files contained usable text. Knowledge base not built."))
            else:
                try:
                    if store is None:
                        store = VectorStore(get_embedder())
                    report = store.add_documents(good, int(chunk_size), int(chunk_overlap))
                except (RAGError, ValueError) as exc:
                    messages.append(("error", str(exc)))
                else:
                    st.session_state.store = store
                    st.session_state.kb_version += 1
                    st.session_state.last_result = None
                    st.session_state.uploader_key += 1  # empty the uploader for the next round
                    if report["unchanged"]:
                        messages.append(("info", "Already in the knowledge base (skipped): "
                                         + ", ".join(report["unchanged"])))
                    if report["updated"]:
                        messages.append(("info", "Replaced with the new version: " + ", ".join(report["updated"])))
                    messages.append(("success",
                                     f"Added {len(report['added']) + len(report['updated'])} file(s) "
                                     f"({report['chunks']} new chunks). Knowledge base now has "
                                     f"{len(store.sources)} file(s) and {len(store.records)} chunks."))
        st.session_state.flash = messages
        st.rerun()

store = st.session_state.store
if store:
    with st.expander(f"📂 Knowledge base: {len(store.sources)} document(s), {len(store.records)} chunks "
                     f"(chunk size {store.settings[0]}, overlap {store.settings[1]})", expanded=True):
        for name, count in store.chunk_counts().items():
            col_name, col_btn = st.columns([5, 1])
            col_name.markdown(f"📄 **{name}** - {count} chunks")
            col_btn.button("Remove", key=f"remove_{name}", on_click=remove_document, args=(name,))
else:
    st.caption("Knowledge base is empty.")

# ---------------- Step 2: ask ----------------
st.subheader("2. Ask a question")
question = st.text_input(
    "Ask a question about your documents:", key="question",
    placeholder="e.g. Compare the tuition fees in both documents",
)
selected_docs = None
if store and len(store.sources) > 1:
    selected_docs = st.multiselect(
        "Documents to search", store.sources, default=store.sources,
        key=f"doc_filter_{st.session_state.kb_version}",
        help="Untick documents to leave them out. Naming a file in your question also focuses on it.",
    )

if st.button("💬 Answer"):
    q = (question or "").strip()
    if not q:
        st.warning("Please type a question.")
    elif not api_key:
        st.error("Enter your Groq API key in the sidebar.")
    elif store is None:
        st.error("Upload a document and click 'Add to knowledge base' first.")
    elif selected_docs is not None and not selected_docs:
        st.warning("Select at least one document to search.")
    else:
        available = selected_docs or store.sources
        compare, targets = plan_retrieval(q, available, mode)
        label = f"comparing {len(targets)} documents" if compare else f"top-{top_k} chunks"
        with st.spinner(f"Retrieving {label} and asking {model}..."):
            try:
                extra = expand_queries(q, api_key)  # multi-query retrieval (falls back to [] on error)
                if compare:
                    hits = store.search_per_document(q, top_k, targets, expand=True, extra_queries=extra)
                else:
                    hits = store.search(q, top_k, targets, expand=True, extra_queries=extra)
                if not hits:
                    raise RAGError("No chunks were retrieved. Try re-adding your documents.")
                answer = generate_answer(q, hits, api_key, model, compare=compare)
            except RAGError as exc:
                st.error(str(exc))
            except Exception as exc:  # last safety net - no raw traceback for users
                st.error(f"Something went wrong ({type(exc).__name__}). Please try again.")
            else:
                st.session_state.last_result = {"question": q, "answer": answer, "hits": hits,
                                                "model": model, "k": top_k, "compare": compare,
                                                "targets": targets}

# ---------------- Step 3: show answer + sources ----------------
result = st.session_state.last_result
if result:
    st.subheader("Answer")
    how = (f"compared {len(result['targets'])} documents, k = {result['k']} per document"
           if result["compare"] else f"k = {result['k']}")
    st.caption(f"Question: {result['question']} · model: {result['model']} · {how}")
    with st.container(border=True):
        st.markdown(result["answer"])
    st.download_button("⬇️ Download answer (.md)", data=f"# {result['question']}\n\n{result['answer']}\n",
                       file_name="rag_answer.md", mime="text/markdown")

    with st.expander(f"Sources ({len(result['hits'])} retrieved chunks)", expanded=False):
        for i, hit in enumerate(result["hits"], start=1):
            st.markdown(
                f"**Passage {i}** · 📄 `{hit['source']}` · page {hit.get('page', '-')}"
                f"{'-' + str(hit['page_end']) if hit.get('page_end') not in (None, hit.get('page')) else ''} · "
                f"section *{hit.get('section') or '-'}* · chunks {hit.get('chunk_nos', [hit['chunk_no']])} · "
                f"similarity score **{hit['score']:.3f}**"
            )
            st.progress(min(max(hit["score"], 0.0), 1.0))
            st.write(hit["text"])
            if i < len(result["hits"]):
                st.divider()
