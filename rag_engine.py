"""
rag_engine.py - The five RAG stages.

    1. CHUNK     chunk_document()         split text at sentence/section boundaries
                 chunk_text()             (simple word-window version from the handout)
    2. EMBED     load_embedder()          all-MiniLM-L6-v2 turns text into vectors
    3. INDEX     VectorStore.add_documents()  normalised vectors in FAISS IndexFlatIP
                                              + a BM25 keyword index
    4. RETRIEVE  VectorStore.search()     hybrid search: FAISS (meaning) + BM25 (keywords)
                 VectorStore.search_per_document()   best chunks from EACH document
    5. GENERATE  generate_answer()        Groq LLM answers from those chunks only

Why these upgrades? Long handbooks broke the simple version:
  - word windows cut policy sections in half (the "75%" sentence ended up in another chunk)
  - questions full of names ("IBA, KSBL and DHA Suffa University") matched cover pages,
    addresses and contact lists instead of the actual policy
  - a section's table or follow-up paragraph was in the NEXT chunk and got lost.
Fixes: sentence/section-aware chunks, keyword+vector (hybrid) search, a "focused" query
without names for comparisons, and neighbour expansion that sends whole sections.

This file has no Streamlit code, so you can also test it from a plain Python script.
"""

import hashlib
import math
import re
from collections import Counter
from typing import Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config


class RAGError(Exception):
    """An error with a message that is safe and friendly to show to the user."""


# ======================= Stage 1 - CHUNK =======================
def chunk_text(text: str, size: int = config.DEFAULT_CHUNK_SIZE,
               overlap: int = config.DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """
    Simple version from the handout: chunks of `size` words, each repeating the last
    `overlap` words of the previous one.
    Example (size=150, overlap=30): chunk 1 = words 0-149, chunk 2 = words 120-269, ...
    """
    _check_chunk_settings(size, overlap)
    words = text.split()
    chunks = []
    step = size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + size])
        if chunk:                       # never store empty chunks
            chunks.append(chunk)
        if start + size >= len(words):  # last window reached the end - stop
            break
    return chunks


def _check_chunk_settings(size: int, overlap: int):
    if size <= 0:
        raise ValueError("Chunk size must be positive.")
    if not 0 <= overlap < size:
        raise ValueError("Overlap must be at least 0 and smaller than the chunk size.")


def _is_heading(line: str) -> bool:
    """Short line that looks like a title: 'ATTENDANCE', 'Class Attendance Policy:'."""
    s = line.strip().rstrip(":").strip()
    words = s.split()
    if not 1 <= len(words) <= 8 or len(s) < 3 or s[-1] in ".,;?!":
        return False
    if sum(c.isalpha() for c in s) < 3:
        return False
    if s.isupper():
        return True
    return len(words) <= 6 and all(w[0].isupper() or not w[0].isalpha() for w in words)


def _units(page_text: str):
    """Yield (text, is_heading) units: headings, and sentences of normal paragraphs."""
    paragraph = []

    def flush():
        text = " ".join(paragraph).strip()
        paragraph.clear()
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if sentence.strip():
                yield sentence.strip(), False

    for line in page_text.split("\n"):
        if not line.strip():
            yield from flush()
        elif _is_heading(line):
            yield from flush()
            yield line.strip().rstrip(":").strip(), True
        else:
            paragraph.append(line.strip())
    yield from flush()


def chunk_document(pages: List[str], size: int = config.DEFAULT_CHUNK_SIZE,
                   overlap: int = config.DEFAULT_CHUNK_OVERLAP) -> List[Dict]:
    """
    Structure-aware chunking.
      - chunks end at sentence boundaries (never mid-sentence)
      - a new section heading starts a new chunk, and every chunk remembers its section
      - overlap = whole sentences (about `overlap` words) repeated from the previous chunk
      - repeated page headers/footers (same line on 3+ pages) are dropped as noise
    Returns [{"text", "body", "section", "page", "page_end"}]; "body" is the text
    without the overlap, used to stitch neighbouring chunks back together.
    """
    _check_chunk_settings(size, overlap)
    units = []
    for page_no, page in enumerate(pages, start=1):
        for text, heading in _units(page):
            if not heading and len(text.split()) > size:  # very long "sentence" (tables, lists)
                units += [{"text": piece, "heading": False, "page": page_no}
                          for piece in chunk_text(text, size, 0)]
            else:
                units.append({"text": text, "heading": heading, "page": page_no})
    if not units:
        return []
    noise = {t for t, n in Counter(u["text"].lower() for u in units).items() if n >= 3 and len(t) < 80}
    units = [u for u in units if u["text"].lower() not in noise]

    chunks, current, carried, words, section = [], [], 0, 0, ""

    def flush(keep_overlap=True):
        nonlocal current, carried, words
        body = current[carried:]
        if body:
            chunks.append({
                "text": " ".join(u["text"] for u in current),
                "body": " ".join(u["text"] for u in body),
                "section": current_section,
                "page": body[0]["page"],
                "page_end": current[-1]["page"],
            })
        tail = []
        if keep_overlap and overlap:
            for u in reversed(current):
                if sum(len(t["text"].split()) for t in tail) + len(u["text"].split()) > overlap:
                    break
                tail.insert(0, u)
        current, carried = tail, len(tail)
        words = sum(len(u["text"].split()) for u in current)

    current_section = ""
    for u in units:
        n = len(u["text"].split())
        if u["heading"]:
            if words - sum(len(t["text"].split()) for t in current[:carried]) >= size * 0.4:
                flush(keep_overlap=False)       # a new section starts a new chunk
            if len(current) == carried:          # chunk (re)starts here -> label it
                current_section = u["text"]
            section = u["text"]
        elif words + n > size and len(current) > carried:
            flush()
            current_section = section
        if not current or len(current) == carried:
            current_section = section
        current.append(u)
        words += n
    flush(keep_overlap=False)
    return chunks


# ======================= Stage 2 - EMBED =======================
def load_embedder(model_name: str = config.EMBEDDING_MODEL) -> SentenceTransformer:
    """Load the embedding model (downloaded once, ~90 MB). The app caches this."""
    return SentenceTransformer(model_name)


def embed(embedder: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """Text -> L2-normalised float32 vectors (so inner product = cosine similarity)."""
    vectors = embedder.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False)
    vectors = vectors.astype("float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid divide-by-zero
    return vectors / norms


# ============ Keyword search (BM25) - complements the embeddings ============
STOPWORDS = set("""a an the of and or to in on for is are be been by with as at from that this
these those it its was were will shall may can do does did what which who whom how why when where
all any each their his her they them you your i we our about into than then there here not no
please tell me give list explain describe""".split())


def tokenize(text: str) -> List[str]:
    """Lower-case words without stop-words, with a tiny stemmer (policies -> policy)."""
    out = []
    for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()):
        if w in STOPWORDS:
            continue
        if len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


class BM25:
    """Classic BM25 ranking: rewards chunks containing the query's rarer words."""

    def __init__(self, texts: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(t)) for t in texts]
        self.lengths = np.array([sum(tf.values()) for tf in self.tfs], dtype="float32")
        self.avg = float(self.lengths.mean()) if len(texts) else 0.0
        df = Counter(w for tf in self.tfs for w in tf)
        n = len(texts)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query: str) -> np.ndarray:
        result = np.zeros(len(self.tfs), dtype="float32")
        for w in set(tokenize(query)):
            idf = self.idf.get(w)
            if not idf:
                continue
            tf = np.array([t.get(w, 0) for t in self.tfs], dtype="float32")
            norm = self.k1 * (1 - self.b + self.b * self.lengths / max(self.avg, 1e-9))
            result += idf * tf * (self.k1 + 1) / (tf + norm)
        return result


# ================= Stage 3 + 4 - INDEX & RETRIEVE =================
class VectorStore:
    """
    FAISS index + BM25 index + a parallel list of chunk records.
    Row i in both indexes  <->  self.records[i]  (text, source filename, page, section).

    Documents can be ADDED in several rounds: only new or changed files are embedded,
    files that are already indexed are skipped (this keeps rebuilding fast).
    """

    def __init__(self, embedder: SentenceTransformer):
        self.embedder = embedder
        self.documents: Dict[str, Dict] = {}  # filename -> {"pages": [...], "hash": ...}
        self.records: List[Dict] = []
        self.vectors = None                   # numpy array, row i = records[i]
        self.index = None
        self.bm25 = None
        self.settings = None                  # (chunk_size, overlap) used for the chunks

    # ---------- information ----------
    @property
    def sources(self) -> List[str]:
        return list(self.documents)

    def chunk_counts(self) -> Dict[str, int]:
        counts = {name: 0 for name in self.documents}
        for r in self.records:
            counts[r["source"]] += 1
        return counts

    # ---------- building ----------
    def clear(self):
        self.documents, self.records, self.vectors, self.index, self.bm25, self.settings = {}, [], None, None, None, None

    def build(self, documents: List[Dict], size: int, overlap: int) -> int:
        """Start from scratch with these documents. Returns the number of chunks."""
        self.clear()
        self.add_documents(documents, size, overlap)
        return len(self.records)

    def add_documents(self, documents: List[Dict], size: int, overlap: int) -> Dict:
        """
        Add documents ([{"filename", "text", "pages"(optional)}]) to the knowledge base.
        - same filename + same content  -> skipped (already indexed)
        - same filename + new content   -> old version replaced
        - chunk settings changed        -> everything already stored is re-chunked
        Returns {"added": [...], "updated": [...], "unchanged": [...], "chunks": n_new_chunks}
        """
        _check_chunk_settings(size, overlap)
        report = {"added": [], "updated": [], "unchanged": [], "chunks": 0}
        if self.records and self.settings != (size, overlap):
            old_docs = [{"filename": f, "pages": d["pages"]} for f, d in self.documents.items()]
            self.clear()
            self._add(old_docs, size, overlap, report={"added": [], "updated": [], "unchanged": [], "chunks": 0})
        self._add(documents, size, overlap, report)
        if not self.records:
            raise RAGError("No text chunks were produced from the uploaded documents.")
        return report

    def _add(self, documents, size, overlap, report):
        new_records = []
        for doc in documents:
            name = doc["filename"]
            pages = doc.get("pages") or [doc.get("text", "")]
            digest = hashlib.sha1("\f".join(pages).encode("utf-8")).hexdigest()
            old = self.documents.get(name)
            if old and old["hash"] == digest:
                report["unchanged"].append(name)
                continue
            chunks = chunk_document(pages, size, overlap)
            if not chunks:
                continue
            if old:
                self.remove_document(name)
                report["updated"].append(name)
            else:
                report["added"].append(name)
            self.documents[name] = {"pages": pages, "hash": digest}
            new_records += [{**c, "source": name, "chunk_no": i} for i, c in enumerate(chunks, start=1)]

        self.settings = (size, overlap)
        if not new_records:
            return
        try:
            # The section title is embedded with the chunk, so "ATTENDANCE" helps find its text.
            new_vectors = embed(self.embedder, [f"{r['section']}: {r['text']}" if r["section"] else r["text"]
                                                for r in new_records])
        except Exception as exc:
            for r in new_records:  # roll back so the store stays consistent
                self.documents.pop(r["source"], None)
            raise RAGError(f"Failed to create embeddings: {exc}") from exc
        self.records += new_records
        self.vectors = new_vectors if self.vectors is None else np.vstack([self.vectors, new_vectors])
        self._rebuild_index()
        report["chunks"] += len(new_records)

    def remove_document(self, name: str):
        keep = [i for i, r in enumerate(self.records) if r["source"] != name]
        self.records = [self.records[i] for i in keep]
        self.vectors = self.vectors[keep] if self.vectors is not None and keep else None
        self.documents.pop(name, None)
        self._rebuild_index()

    def _rebuild_index(self):
        try:
            if self.vectors is None or len(self.vectors) == 0:
                self.index, self.bm25 = None, None
                return
            index = faiss.IndexFlatIP(self.vectors.shape[1])  # inner product on unit vectors = cosine
            index.add(self.vectors)
            self.index = index
            self.bm25 = BM25([f"{r['section']} {r['text']}" for r in self.records])
        except Exception as exc:
            raise RAGError(f"Failed to build the search index: {exc}") from exc

    # ---------- searching ----------
    def _rank(self, question: str, rows: List[int], k: int, compare: bool = False,
              extra_queries: Optional[List[str]] = None) -> List[Dict]:
        """
        Hybrid, multi-query ranking of the given rows using Reciprocal Rank Fusion (RRF).
        Ranked lists that vote:
          - FAISS cosine similarity for the question, for the focused question
            (names / "compare" removed) and for every extra sub-query
          - BM25 keyword scores for the (focused) question and every sub-query
        Each list gives a chunk 1 / (60 + rank); chunks that rank well in several lists win.
        Returns the overall top-k PLUS the best chunk of every sub-query, so each aspect of
        the question (requirement, penalty, exceptions, tables...) is represented.
        """
        if not rows:
            return []
        extras = [q for q in (extra_queries or []) if q.strip()]
        focused = focus_query(question, self.sources)
        variants = [question] + ([focused] if focused != question else []) + extras
        q_vectors = embed(self.embedder, variants)
        scores, ids = self.index.search(q_vectors, self.index.ntotal)  # FAISS: all rows, sorted
        dense = np.zeros((len(variants), self.index.ntotal), dtype="float32")
        for v in range(len(variants)):
            dense[v, ids[v]] = scores[v]
        row_arr = np.array(rows)

        def rrf(score_lists, keyword_flags):
            fused = np.zeros(len(rows), dtype="float32")
            for values, is_keyword in zip(score_lists, keyword_flags):
                order = np.argsort(-values, kind="stable")
                ranks = np.empty(len(rows), dtype="float32")
                ranks[order] = np.arange(1, len(rows) + 1)
                contribution = 1.0 / (60.0 + ranks)
                if is_keyword:
                    contribution[values <= 0] = 0.0  # no shared keyword -> no keyword credit
                fused += contribution
            return fused

        dense_lists = [dense[v, row_arr] for v in range(len(variants))]
        main_keyword = self.bm25.scores(focused if compare else question)[row_arr]
        extra_keywords = [self.bm25.scores(q)[row_arr] for q in extras]
        overall = rrf(dense_lists + [main_keyword] + extra_keywords,
                      [False] * len(dense_lists) + [True] * (1 + len(extras)))
        n_base = len(variants) - len(extras)
        per_query = [rrf([dense_lists[n_base + i], extra_keywords[i]], [False, True]) for i in range(len(extras))]

        picked = []
        overall_order = [int(j) for j in np.argsort(-overall, kind="stable")]
        if overall_order:
            picked.append(overall_order[0])
        for fused in per_query:                       # best chunk for every sub-query
            for j in np.argsort(-fused, kind="stable"):
                if int(j) not in picked:
                    picked.append(int(j))
                    break
        for j in overall_order:                        # fill up with the overall best
            if len(picked) >= k + len(extras):
                break
            if j not in picked:
                picked.append(j)

        cosine = np.max(np.stack(dense_lists), axis=0)
        keyword = np.max(np.stack([main_keyword] + extra_keywords), axis=0)
        return [{**self.records[rows[j]], "row": rows[j], "score": float(cosine[j]),
                 "keyword_score": float(keyword[j]), "rank_score": float(overall[j])}
                for j in picked[:k + len(extras)]]

    def _rows(self, sources: Optional[List[str]]) -> List[int]:
        allowed = set(sources) if sources is not None else None
        return [i for i, r in enumerate(self.records) if allowed is None or r["source"] in allowed]

    def search(self, query: str, k: int = config.DEFAULT_TOP_K, sources: Optional[List[str]] = None,
               expand: bool = False, word_budget: int = config.CONTEXT_WORD_BUDGET,
               extra_queries: Optional[List[str]] = None) -> List[Dict]:
        """Top-k chunks (optionally only from `sources`), best first.
        expand=True turns the hits into complete passages (see _expand)."""
        if self.index is None or self.index.ntotal == 0:
            return []
        hits = self._rank(query, self._rows(sources), k, extra_queries=extra_queries)
        return self._expand(hits, word_budget) if expand else hits

    def search_per_document(self, query: str, k_per_doc: int = config.DEFAULT_TOP_K,
                            sources: Optional[List[str]] = None, expand: bool = False,
                            word_budget: int = config.CONTEXT_WORD_BUDGET,
                            extra_queries: Optional[List[str]] = None) -> List[Dict]:
        """
        COMPARISON retrieval: the best k chunks from EACH document, so every document
        gets a voice in the context (plain top-k can return chunks from only one file).
        The context budget is shared equally between the documents.
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        wanted = [s for s in (sources or self.sources) if s in self.documents]
        results = []
        for source in wanted:
            hits = self._rank(query, self._rows([source]), k_per_doc, compare=True,
                              extra_queries=extra_queries)
            results += self._expand(hits, word_budget // max(1, len(wanted))) if expand else hits
        return results  # grouped by document

    def _expand(self, hits: List[Dict], word_budget: int) -> List[Dict]:
        """
        Neighbour expansion: add the chunk before and after every hit (same document)
        and stitch touching chunks into one passage, so a policy's heading, its rules
        and its table travel together. Passages are kept in relevance order until the
        word budget is used (the best passage is always kept).
        """
        window = config.NEIGHBOR_WINDOW
        rows = set()
        best_hit = {}
        for rank, h in enumerate(hits):
            for r in range(h["row"] - window, h["row"] + window + 1):
                if 0 <= r < len(self.records) and self.records[r]["source"] == h["source"]:
                    rows.add(r)
                    best_hit.setdefault(r, rank)
        groups, current = [], []
        for r in sorted(rows):
            if current and (r != current[-1] + 1 or self.records[r]["source"] != self.records[current[-1]]["source"]):
                groups.append(current)
                current = []
            current.append(r)
        if current:
            groups.append(current)

        by_row = {h["row"]: h for h in hits}
        passages = []
        for g in groups:
            first = self.records[g[0]]
            text = " ".join([first["text"]] + [self.records[r]["body"] for r in g[1:]])
            matched = [by_row[r] for r in g if r in by_row]
            passages.append({
                "source": first["source"],
                "chunk_no": first["chunk_no"],
                "chunk_nos": [self.records[r]["chunk_no"] for r in g],
                "matched_chunks": [m["chunk_no"] for m in matched],
                "section": first["section"] or next((self.records[r]["section"] for r in g if self.records[r]["section"]), ""),
                "page": first["page"],
                "page_end": self.records[g[-1]]["page_end"],
                "text": text,
                "score": max(m["score"] for m in matched),
                "keyword_score": max(m["keyword_score"] for m in matched),
                "rank": min(best_hit[r] for r in g),
            })
        passages.sort(key=lambda p: p["rank"])
        kept, used = [], 0
        for p in passages:
            n = len(p["text"].split())
            if kept and used + n > word_budget:
                # Too big with its neighbours: keep just the matched chunk(s) if they fit,
                # so an important hit is never dropped completely.
                core_rows = [h["row"] for h in hits if h["source"] == p["source"] and h["chunk_no"] in p["matched_chunks"]]
                core = " ".join(self.records[r]["text"] for r in core_rows)
                if core and used + len(core.split()) <= word_budget:
                    kept.append({**p, "text": core, "chunk_nos": p["matched_chunks"],
                                 "page": self.records[core_rows[0]]["page"],
                                 "page_end": self.records[core_rows[-1]]["page_end"]})
                    used += len(core.split())
                continue
            kept.append(p)
            used += n
        return kept


# ============ Deciding HOW to retrieve (normal vs comparison) ============
COMPARE_PATTERN = re.compile(
    r"\b(compare|comparing|comparison|contrast|differ|differs|difference|differences|different|"
    r"similar|similarity|similarities|vs|versus|both|which (one|document|file)|all (the )?(documents|files|pdfs))\b",
    re.IGNORECASE,
)
# "Which university is cheaper?", "Which one has the later deadline?" ...
SUPERLATIVE_PATTERN = re.compile(
    r"\bwhich\b.*\b(higher|lower|more|less|better|worse|cheaper|costlier|larger|smaller|bigger|"
    r"earlier|later|longer|shorter|best|worst|most|least|\w+est)\b|\bbetween\b.*\band\b",
    re.IGNORECASE,
)


def is_comparison_question(question: str) -> bool:
    return bool(COMPARE_PATTERN.search(question) or SUPERLATIVE_PATTERN.search(question))


def focus_query(question: str, sources: Optional[List[str]] = None) -> str:
    """
    Remove words that identify WHICH document (acronyms like 'IBA', proper names like
    'Suffa University', file names) and comparison words, keeping the TOPIC
    ('attendance policies'). Used when searching inside each document separately,
    where the names only attract cover pages, addresses and contact lists.
    """
    text = COMPARE_PATTERN.sub(" ", question)
    for s in sources or []:
        stem = s.rsplit(".", 1)[0]
        text = re.sub(re.escape(stem), " ", text, flags=re.IGNORECASE)
    kept = []
    for i, word in enumerate(text.split()):
        core = word.strip(",.?!:;()'\"“”’")
        if re.fullmatch(r"[A-Z][A-Z0-9&\-]+", core):   # acronym: IBA, KSBL, DHA
            continue
        if i > 0 and core[:1].isupper():               # proper noun: Suffa, University
            continue
        kept.append(word)
    focused = " ".join(kept).strip(" ,.;:")
    return focused if tokenize(focused) else question


QUERY_PROMPT = (
    "You write search queries for a document search engine. Given a user's question, write "
    "{n} short search queries (3-8 words each) that together cover the different aspects a "
    "document might use to answer it - for example the main requirement or rule, numbers and "
    "tables, limits, penalties or consequences, exceptions, procedures and deadlines, and "
    "alternatives. Use generic wording that would appear in such documents. Do NOT include "
    "names of institutions, organisations or people, and do NOT answer the question. "
    "Return exactly one query per line and nothing else."
)


def expand_queries(question: str, api_key: str, model: str = config.QUERY_MODEL,
                   n: int = config.MAX_SUBQUERIES) -> List[str]:
    """
    Multi-query retrieval: ask a small LLM for a few sub-queries that describe the aspects
    of the question. They only steer the SEARCH - the answer still comes only from the
    documents. Returns [] on any problem, so retrieval simply falls back to one query.
    """
    if not api_key or n <= 0:
        return []
    try:
        from groq import Groq

        client = Groq(api_key=api_key, timeout=30, max_retries=0)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": QUERY_PROMPT.format(n=n)},
                      {"role": "user", "content": question}],
            temperature=0.2,
        )
        text = re.sub(r"<think>.*?</think>", "", response.choices[0].message.content or "", flags=re.DOTALL)
    except Exception:
        return []
    queries = []
    for line in text.splitlines():
        q = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"')
        q = focus_query(q) if q else q
        if 2 <= len(q.split()) <= 12 and q.lower() not in {x.lower() for x in queries}:
            queries.append(q)
    return queries[:n]


def _normalise(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "


def mentioned_sources(question: str, sources: List[str]) -> List[str]:
    """Files whose name (without extension) appears in the question, e.g. 'riverside_admissions'."""
    q = _normalise(question)
    found = []
    for s in sources:
        stem = _normalise(s.rsplit(".", 1)[0])
        if stem.strip() and stem in q:
            found.append(s)
    return found


def plan_retrieval(question: str, sources: List[str], mode: str = "auto"):
    """
    Returns (compare: bool, targets: list of filenames to search).
    mode: "auto" (detect comparison questions), "all" (normal top-k), "compare" (always per-document).
    """
    mentioned = mentioned_sources(question, sources)
    targets = mentioned or list(sources)
    if mode == "compare":
        compare = len(targets) >= 2
    elif mode == "all":
        compare = False
    else:
        compare = len(targets) >= 2 and (len(mentioned) >= 2 or is_comparison_question(question))
    return compare, targets


# ======================= Stage 5 - GENERATE =======================
def _label(h: Dict) -> str:
    pages = h.get("page")
    if pages and h.get("page_end") and h["page_end"] != pages:
        pages = f"{pages}-{h['page_end']}"
    parts = [h["source"]]
    if pages:
        parts.append(f"page {pages}")
    if h.get("section"):
        parts.append(f"section: {h['section']}")
    return " | ".join(parts)


def build_context(hits: List[Dict], compare: bool = False) -> str:
    if not compare:
        return "\n\n".join(f"[Source {n}: {_label(h)}]\n{h['text']}" for n, h in enumerate(hits, start=1))
    blocks, current = [], None
    for h in hits:  # hits are already grouped by document
        if h["source"] != current:
            current = h["source"]
            n_passages = sum(1 for x in hits if x["source"] == current)
            blocks.append(f"===== DOCUMENT: {current} ({n_passages} passages - use all of them) =====")
        blocks.append(f"[{_label(h)}]\n{h['text']}")
    return "\n\n".join(blocks)


EXTRACTION_RULES = (
    "Rules for reading the context:\n"
    "- Report every specific requirement exactly as written: percentages, numbers of allowed "
    "absences, deadlines, time windows, grades/penalties, exceptions and alternatives (e.g. withdrawal).\n"
    "- For time windows, say WHO must act and WITHIN WHAT TIME, e.g. 'the student must request "
    "within X hours; the office responds within Y hours'.\n"
    "- If the context contains a table, reproduce its rows (as a table) instead of summarising them.\n"
    "- A document may have SEVERAL passages from different pages (e.g. the regulations section "
    "AND an FAQ). Read and combine ALL passages of a document before writing about it; rules "
    "about withdrawal, exceptions or deadlines are often in a different passage than the main rule.\n"
    "- Keep conditions attached to the rule they belong to: if a penalty applies to missing an "
    "exam, do not present it as the penalty for low attendance.\n"
    "- Only say a document does not specify something if it truly does not appear in that "
    "document's passages - check all passages of that document first.\n"
    "- Cite the file and page for each fact, e.g. (file.pdf, p. 19).\n"
    "- Text extracted from PDFs can be slightly garbled (merged words, numbers run together); "
    "interpret it carefully and mention any uncertainty."
)


FORMAT_RULES = (
    "Format the answer in clean Markdown:\n"
    "- Start with a short '**Summary:**' line (1-2 sentences).\n"
    "- Use '### ' headings (one per document when comparing).\n"
    "- Use bullet points for rules; put key numbers, percentages, deadlines and grades in **bold**.\n"
    "- Use Markdown tables for comparisons and tables from the documents. Keep each table cell "
    "short and on ONE line: no line breaks, no HTML tags such as <br>, no code blocks inside tables.\n"
    "- End with a '### Sources' section listing the files and pages used."
)


def format_answer(text: str) -> str:
    """Tidy the model's Markdown so it renders well in Streamlit."""
    text = re.sub(r"<br\s*/?>", " · ", text, flags=re.IGNORECASE)   # HTML line breaks in tables
    text = re.sub(r"</?(p|div|span)[^>]*>", "", text, flags=re.IGNORECASE)
    text = text.replace("\u202f", " ").replace("\u00a0", " ")       # odd spaces from the model
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_messages(question: str, hits: List[Dict], compare: bool = False) -> List[Dict]:
    if compare:
        n_docs = len({h["source"] for h in hits})
        instructions = (
            f"The context below comes from {n_docs} different documents and is grouped by document. "
            "Answer the question by COMPARING them, using ONLY the context:\n"
            "1. For each document (name the file), list everything it says about the question.\n"
            "2. Then give a comparison table of the key points and summarise the similarities and differences.\n"
            "If a document does not contain the needed information, say so for that document. "
            "Do not use outside knowledge.\n\n" + EXTRACTION_RULES
        )
    else:
        instructions = ("Answer the question using ONLY the context below. If the answer is not in "
                        "the context, say that you don't know.\n\n" + EXTRACTION_RULES)
    instructions += "\n\n" + FORMAT_RULES
    user_prompt = (f"{instructions}\n\nContext:\n{build_context(hits, compare)}\n\n"
                   f"Question: {question}\nAnswer:")
    return [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _friendly_groq_error(exc: Exception, model: str) -> RAGError:
    """Convert Groq SDK exceptions into messages a normal user understands.
    The API key itself is never included."""
    import groq

    msg = str(exc).lower()
    if isinstance(exc, groq.AuthenticationError):
        return RAGError("Invalid Groq API key. Re-copy it from console.groq.com (no spaces or quotes).")
    if isinstance(exc, groq.RateLimitError):
        return RAGError("Groq rate limit reached (free tier). Wait about a minute and try again, or pick a smaller model.")
    if isinstance(exc, groq.NotFoundError) or "decommissioned" in msg or "does not exist" in msg:
        return RAGError(f"The model '{model}' is not available on Groq any more. Choose another model in the sidebar.")
    if isinstance(exc, groq.PermissionDeniedError):
        return RAGError("Your Groq account is not allowed to use this model. Choose another model.")
    if isinstance(exc, (groq.APIConnectionError, groq.APITimeoutError)):
        return RAGError("Could not reach Groq. Check your internet connection and try again.")
    if isinstance(exc, groq.APIStatusError):
        if exc.status_code == 413 or "too large" in msg or "context" in msg:
            return RAGError("The request is too large for Groq's free tier (tokens per minute). Lower k, untick some documents, or wait a minute and try again.")
        return RAGError(f"Groq returned an error (HTTP {exc.status_code}). Please try again.")
    return RAGError(f"Unexpected error while generating the answer ({type(exc).__name__}).")


def generate_answer(question: str, hits: List[Dict], api_key: str,
                    model: str = config.DEFAULT_MODEL, compare: bool = False) -> str:
    """Send question + retrieved chunks to Groq and return the grounded answer."""
    from groq import Groq

    if not api_key:
        raise RAGError("Missing Groq API key. Enter it in the sidebar.")
    if not hits:
        # Never fall back to the model's general knowledge.
        raise RAGError("No relevant chunks were retrieved, so there is nothing to answer from.")

    import groq

    client = Groq(api_key=api_key, timeout=90, max_retries=1)
    # If Groq says the request is too large (free-tier tokens-per-minute cap), shorten every
    # passage and try again instead of failing.
    for attempt, keep in enumerate((1.0, 0.6, 0.35)):
        trimmed = [{**h, "text": " ".join(h["text"].split()[:max(60, int(len(h["text"].split()) * keep))])}
                   for h in hits]
        try:
            response = client.chat.completions.create(
                model=model,
                messages=build_messages(question, trimmed, compare),
                temperature=config.TEMPERATURE,
            )
            break
        except groq.APIStatusError as exc:
            too_large = exc.status_code == 413 or "too large" in str(exc).lower() or "reduce" in str(exc).lower()
            if too_large and attempt < 2:
                continue
            raise _friendly_groq_error(exc, model) from None
        except Exception as exc:
            raise _friendly_groq_error(exc, model) from None

    answer = response.choices[0].message.content or ""
    # Some reasoning models wrap their thinking in <think> tags - hide it.
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    return format_answer(answer) or "The model returned an empty answer. Please try again."
