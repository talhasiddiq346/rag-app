"""
End-to-end tests for the RAG pipeline (no browser needed).

Run from the rag-app folder:
    python tests/test_pipeline.py
Tests that call Groq are skipped if GROQ_API_KEY is not set.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from document_processor import extract_text  # noqa: E402
from rag_engine import RAGError, VectorStore, chunk_text, generate_answer, load_embedder  # noqa: E402

FILES = ROOT / "tests" / "test_files"
SAMPLES = ROOT / "sample_documents"
API_KEY = config.get_api_key_from_environment()
results = []


def check(name, condition, detail=""):
    results.append((name, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))


def load(path):
    return extract_text(path.name, path.read_bytes())


print("Loading embedding model...")
embedder = load_embedder()

# ---------------- chunking ----------------
words = " ".join(f"w{i}" for i in range(400))
chunks = chunk_text(words, 150, 30)
# windows start at word 0, 120, 240, 360 -> 4 chunks; together they cover every word
covered = {w for c in chunks for w in c.split()}
check("Chunking: 400 words -> 4 chunks, 30-word overlap, all words covered",
      len(chunks) == 4 and chunks[1].split()[0] == "w120" and len(covered) == 400,
      f"{len(chunks)} chunks, starts {[c.split()[0] for c in chunks]}")
check("Chunking: empty text -> no chunks", chunk_text("   ") == [])
check("Chunking: exactly 150 words -> 1 chunk (no empty/duplicate tail)",
      len(chunk_text(" ".join(["x"] * 150))) == 1)
try:
    chunk_text("a b c", 10, 10)
    check("Chunking: overlap >= size rejected", False)
except ValueError:
    check("Chunking: overlap >= size rejected", True)

# ---------------- TEST 1-3: TXT, PDF, DOCX ----------------
txt, pdf, docx_ = load(SAMPLES / "sample.txt"), load(FILES / "robotics_report.pdf"), load(FILES / "cafeteria_policy.docx")
for label, doc in [("TEST 1 TXT", txt), ("TEST 2 PDF", pdf), ("TEST 3 DOCX", docx_)]:
    check(f"{label}: text extracted", doc["error"] is None and len(doc["text"]) > 100,
          f"{len(doc['text'])} chars from {doc['filename']}")
check("TEST 3 DOCX: table text extracted", "Bilal Ahmed" in docx_["text"])

store = VectorStore(embedder)
n = store.build([txt, pdf, docx_], 150, 30)
check("TEST 1-3: knowledge base built from 3 files", n > 3 and store.index.ntotal == n, f"{n} chunks")
check("Index rows map 1:1 to records", store.index.ntotal == len(store.records))

# ---------------- edge cases ----------------
check("Empty TXT -> friendly error", load(FILES / "empty.txt")["error"] is not None, load(FILES / "empty.txt")["error"])
scan = load(FILES / "scanned_image_only.pdf")
check("Scanned PDF -> OCR-not-supported message", scan["error"] and "OCR" in scan["error"], scan["error"])
check("Corrupted PDF -> no crash", load(FILES / "corrupted.pdf")["error"] is not None, load(FILES / "corrupted.pdf")["error"])
check("Unsupported .csv -> rejected", "Unsupported" in (load(FILES / "notes.csv")["error"] or ""))
try:
    VectorStore(embedder).build([], 150, 30)
    check("No documents -> RAGError", False)
except RAGError as e:
    check("No documents -> RAGError", True, str(e))
check("Search on empty store -> []", VectorStore(embedder).search("anything") == [])

# ---------------- TEST 6: retrieval relevance + filenames ----------------
hits = store.search("How many solar panels were installed on the library?", 3)
check("TEST 6: top hit is from sample.txt and mentions panels",
      hits[0]["source"] == "sample.txt" and "panels" in hits[0]["text"], f"score {hits[0]['score']:.3f}")
hits = store.search("Who is the president of the robotics club?", 3)
check("TEST 6: robotics question retrieves the PDF", hits[0]["source"] == "robotics_report.pdf",
      f"{hits[0]['source']} score {hits[0]['score']:.3f}")
hits = store.search("What time does the cafeteria open?", 3)
check("TEST 6: cafeteria question retrieves the DOCX", hits[0]["source"] == "cafeteria_policy.docx",
      f"{hits[0]['source']} score {hits[0]['score']:.3f}")
check("Scores sorted best-first", all(hits[i]["score"] >= hits[i + 1]["score"] for i in range(len(hits) - 1)))

# ---------------- TEST 7: changing k ----------------
for k in (1, 3, 5):
    check(f"TEST 7: k={k} returns {k} chunks", len(store.search("solar budget", k)) == k)
check("TEST 7: k larger than index is capped", len(VectorStore(embedder).search("x", 5)) == 0)

# ---------------- TEST 9: missing key / empty hits ----------------
try:
    generate_answer("q", hits, "", config.DEFAULT_MODEL)
    check("TEST 9: missing API key -> friendly error", False)
except RAGError as e:
    check("TEST 9: missing API key -> friendly error", True, str(e))
try:
    generate_answer("q", [], "dummy", config.DEFAULT_MODEL)
    check("Empty retrieval -> no LLM call", False)
except RAGError as e:
    check("Empty retrieval -> no LLM call", True, str(e))

# ---------------- MULTI-DOCUMENT: adding in rounds + comparison ----------------
from rag_engine import mentioned_sources, plan_retrieval  # noqa: E402
north, river, robo = load(FILES / "northfield_admissions.pdf"), load(FILES / "riverside_admissions.pdf"), pdf
multi = VectorStore(embedder)
multi.add_documents([north], 150, 30)
multi.add_documents([river], 150, 30)          # second upload round
rep = multi.add_documents([robo, north], 150, 30)  # third round, north re-uploaded
check("MULTI: 3 PDFs added in separate rounds are all kept",
      multi.sources == ["northfield_admissions.pdf", "riverside_admissions.pdf", "robotics_report.pdf"], multi.sources)
check("MULTI: re-uploaded unchanged file is skipped (not embedded twice)",
      rep["unchanged"] == ["northfield_admissions.pdf"] and multi.index.ntotal == len(multi.records))
check("MULTI: rows still map 1:1 to records", multi.index.ntotal == len(multi.records) == len(multi.vectors))
before = len(multi.records)
multi.add_documents([], 100, 20)                # chunk settings changed -> re-chunk everything
check("MULTI: changing chunk settings re-chunks all stored docs",
      multi.settings == (100, 20) and len(multi.sources) == 3 and len(multi.records) != before,
      f"{before} -> {len(multi.records)} chunks")
multi.add_documents([], 150, 30)
q = "Compare the tuition fees of Northfield and Riverside"
plain = {h["source"] for h in multi.search(q, 3)}
both = multi.search_per_document(q, 3, ["northfield_admissions.pdf", "riverside_admissions.pdf"])
check("MULTI: per-document retrieval covers BOTH files (plain top-k may not)",
      {h["source"] for h in both} == {"northfield_admissions.pdf", "riverside_admissions.pdf"},
      f"plain top-3 sources={sorted(plain)}")
check("MULTI: per-document returns k chunks per file, grouped",
      len(both) == 6 and [h["source"] for h in both][:3] == ["northfield_admissions.pdf"] * 3)
only = multi.search("tuition fee", 3, ["riverside_admissions.pdf"])
check("MULTI: document filter restricts results", {h["source"] for h in only} == {"riverside_admissions.pdf"})
check("MULTI: filename mentioned in question is detected",
      mentioned_sources("What does riverside admissions say about hostels?", multi.sources) == ["riverside_admissions.pdf"])
check("MULTI: 'compare' question -> comparison mode", plan_retrieval(q, multi.sources)[0] is True)
check("MULTI: 'which ... cheaper' -> comparison mode",
      plan_retrieval("Which university is cheaper?", multi.sources)[0] is True)
check("MULTI: normal question -> normal mode", plan_retrieval("What is the hostel fee?", multi.sources)[0] is False)
check("MULTI: forced 'all' mode never compares", plan_retrieval(q, multi.sources, "all")[0] is False)
multi.remove_document("robotics_report.pdf")
check("MULTI: removing a document drops its chunks",
      "robotics_report.pdf" not in {r["source"] for r in multi.records} and multi.index.ntotal == len(multi.records))

# ---------------- Groq tests ----------------
if not API_KEY:
    print("\nGROQ_API_KEY not set - skipping TEST 4, 5 and Groq error tests.")
else:
    q = "How many solar panels were installed and what is their total capacity?"
    ans = generate_answer(q, store.search(q, 3), API_KEY)
    check("TEST 4: answer found in document (120 panels, 66 kW)", "120" in ans and "66" in ans, ans[:200])

    q = "Who designed the PDF file format?"
    ans = generate_answer(q, store.search(q, 3), API_KEY)
    lowered = ans.lower()
    check("TEST 5: unknown answer -> model admits it doesn't know",
          any(p in lowered for p in ["don't know", "do not know", "not in the", "not provided", "no information", "not contain", "not mention"])
          and "adobe" not in lowered, ans[:200])

    try:
        generate_answer("hi", hits, "gsk_invalid_key_for_testing_000000", config.DEFAULT_MODEL)
        check("TEST 9: invalid API key -> friendly error", False)
    except RAGError as e:
        check("TEST 9: invalid API key -> friendly error", "Invalid Groq API key" in str(e), str(e))
    try:
        generate_answer("hi", hits, API_KEY, "llama-3.3-70b-versatile")
        check("Retired model -> friendly error", False)
    except RAGError as e:
        check("Retired model -> friendly error", "not available" in str(e) or "not allowed" in str(e), str(e))
    q = "Compare the tuition fees of Northfield and Riverside"
    compare, targets = plan_retrieval(q, multi.sources)
    ans = generate_answer(q, multi.search_per_document(q, 3, targets), API_KEY, compare=compare)
    check("MULTI+GROQ: comparison answer contains BOTH tuition fees", "250,000" in ans and "310,000" in ans,
          ans[:300].replace("\n", " "))
    q2 = "Which university needs a higher entry test score, and which has the later application deadline?"
    compare, targets = plan_retrieval(q2, multi.sources)
    ans = generate_answer(q2, multi.search_per_document(q2, 3, targets), API_KEY, compare=compare)
    ans = " ".join(ans.split())  # LLMs sometimes use non-breaking spaces ("15\u202fJuly")
    check("MULTI+GROQ: 'which university' comparison (70% Riverside, 15 July Northfield)",
          compare and "70" in ans and "15 July" in ans, ans[:300].replace("\n", " "))
    for m in config.GROQ_MODELS:
        q = "What time does the cafeteria open?"
        ans = generate_answer(q, store.search(q, 3), API_KEY, m)
        check(f"Model {m} answers from DOCX (8:00 am)", "8" in ans, ans[:120].replace("\n", " "))

passed = sum(ok for _, ok in results)
print(f"\n{passed}/{len(results)} checks passed")
sys.exit(0 if passed == len(results) else 1)
