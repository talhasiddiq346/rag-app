"""
UI tests that run app.py headlessly with Streamlit's AppTest (simulated clicks).

Run from the rag-app folder:
    python tests/test_ui.py
Answer tests are skipped if GROQ_API_KEY is not set.
"""
import os
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
FILES, SAMPLES = ROOT / "tests" / "test_files", ROOT / "sample_documents"
API_KEY = os.environ.pop("GROQ_API_KEY", "")  # hide from the app; we type it into the sidebar instead
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))


def texts(at, kind):
    return [e.value for e in getattr(at, kind)]


def button(at, label_part):
    return next(b for b in at.button if label_part in b.label)


def new_app():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    return at


def upload(at, paths):
    at.file_uploader[0].set_value([(p.name, p.read_bytes(), "application/octet-stream") for p in paths]).run()


at = new_app()
check("App renders without exceptions", not at.exception, at.exception)
check("Title shown", any("Open-Source RAG Application" in t for t in texts(at, "title")))
check("Sidebar has password key input, model select, k slider",
      at.sidebar.text_input[0].label == "Groq API key" and len(at.sidebar.selectbox) == 1 and len(at.sidebar.slider) == 1)

# ---- No documents uploaded ----
button(at, "Add to knowledge base").click().run()
check("Build with no files -> warning", any("upload at least one" in w for w in texts(at, "warning")))

# ---- Missing API key / empty question / KB not built ----
button(at, "Answer").click().run()
check("Empty question -> warning", any("type a question" in w for w in texts(at, "warning")))
at.text_input(key="question").set_value("What is this?").run()
button(at, "Answer").click().run()
check("TEST 9: missing API key -> error", any("API key" in e for e in texts(at, "error")))
at.sidebar.text_input[0].set_value("gsk_dummy_value").run()
button(at, "Answer").click().run()
check("KB not built -> error", any("Add to knowledge base" in e for e in texts(at, "error")))

# ---- TEST 1-3: upload TXT + PDF + DOCX (+ bad files) and build ----
upload(at, [SAMPLES / "sample.txt", FILES / "robotics_report.pdf", FILES / "cafeteria_policy.docx",
            FILES / "scanned_image_only.pdf", FILES / "empty.txt"])
button(at, "Add to knowledge base").click().run()
check("TEST 1-3: KB built from TXT+PDF+DOCX", any("now has 3 file(s)" in s for s in texts(at, "success")),
      texts(at, "success"))
warns = texts(at, "warning")
check("Scanned PDF skipped with OCR message", any("scanned_image_only.pdf" in w and "OCR" in w for w in warns))
check("Empty TXT skipped with message", any("empty.txt" in w for w in warns))
check("No traceback shown", not at.exception)

# ---- Chunk settings ----
at.sidebar.number_input[1].set_value(200).run()  # overlap 200 >= size 150
check("Overlap >= size -> sidebar error", any("Overlap must be smaller" in e for e in texts(at.sidebar, "error")))
at.sidebar.number_input[1].set_value(30).run()

if API_KEY:
    at.sidebar.text_input[0].set_value(API_KEY).run()
    # ---- TEST 4 + 6 ----
    at.text_input(key="question").set_value("Who is the president of the robotics club?").run()
    button(at, "Answer").click().run()
    md = " ".join(texts(at, "markdown"))
    check("TEST 4: answer from document (Ayesha Khan)", "Ayesha Khan" in md)
    check("TEST 6: Sources expander lists filename + score",
          "robotics_report.pdf" in md and "similarity score" in md)
    check("TEST 7: k=3 -> 3 chunks shown", md.count("**Chunk ") == 3)
    # ---- TEST 7: change k ----
    at.sidebar.slider[0].set_value(5).run()
    button(at, "Answer").click().run()
    md = " ".join(texts(at, "markdown"))
    check("TEST 7: k=5 -> 5 chunks shown", md.count("**Chunk ") == 5)
    # ---- TEST 5 ----
    at.text_input(key="question").set_value("What is the capital city of Australia?").run()
    button(at, "Answer").click().run()
    md = " ".join(texts(at, "markdown")).lower()
    check("TEST 5: unrelated question -> 'don't know' (no Canberra)",
          ("know" in md or "not" in md) and "canberra" not in md)
    # ---- Invalid key ----
    at.sidebar.text_input[0].set_value("gsk_invalid_for_testing").run()
    button(at, "Answer").click().run()
    check("TEST 9: invalid key -> friendly error", any("Invalid Groq API key" in e for e in texts(at, "error")))
else:
    print("GROQ_API_KEY not set - skipping answer tests.")


# ---- MULTI-DOCUMENT: 3 PDFs uploaded in 3 separate rounds, then compare ----
button(at, "Clear knowledge base").click().run()
for name in ["northfield_admissions.pdf", "riverside_admissions.pdf", "robotics_report.pdf"]:
    upload(at, [FILES / name])
    button(at, "Add to knowledge base").click().run()
check("MULTI: 3 PDFs uploaded one at a time are all in the KB",
      at.session_state["store"].sources == ["northfield_admissions.pdf", "riverside_admissions.pdf", "robotics_report.pdf"],
      texts(at, "success"))
check("MULTI: uploader emptied after each round", not at.file_uploader[0].value)
upload(at, [FILES / "riverside_admissions.pdf"])
button(at, "Add to knowledge base").click().run()
check("MULTI: re-uploading the same PDF is skipped", any("Already in the knowledge base" in i for i in texts(at, "info")))
check("MULTI: document filter offers all 3 PDFs", len(at.multiselect[0].value) == 3)
if API_KEY:
    at.sidebar.text_input[0].set_value(API_KEY).run()
    at.text_input(key="question").set_value("Compare the tuition fees and hostel fees of the two universities").run()
    button(at, "Answer").click().run()
    md = " ".join(texts(at, "markdown"))
    check("MULTI: comparison answer has both universities' fees",
          "250,000" in md and "310,000" in md and "northfield_admissions.pdf" in md and "riverside_admissions.pdf" in md)
    check("MULTI: caption says documents were compared", any("compared" in c for c in texts(at, "caption")))
    at.text_input(key="question").set_value("Who is the president of the robotics club?").run()
    button(at, "Answer").click().run()
    check("MULTI: single-document question still answered from the 3rd PDF",
          "Ayesha Khan" in " ".join(texts(at, "markdown")))
    at.multiselect[0].set_value(["riverside_admissions.pdf"]).run()
    at.text_input(key="question").set_value("What is the tuition fee?").run()
    button(at, "Answer").click().run()
    md = " ".join(texts(at, "markdown"))
    check("MULTI: filtering to one document uses only that document", "310,000" in md and "250,000" not in md)
next(b for b in at.button if b.label == "Remove").click().run()
check("MULTI: Remove button deletes a document", len(at.session_state["store"].sources) == 2)

# ---- TEST 8: clear knowledge base ----
button(at, "Clear knowledge base").click().run()
check("TEST 8: store removed from session", at.session_state["store"] is None)
check("TEST 8: last answer removed", at.session_state["last_result"] is None)
check("TEST 8: file uploader reset", not at.file_uploader[0].value)
check("TEST 8: UI says KB empty", any("Knowledge base is empty" in c for c in texts(at, "caption")))
at.text_input(key="question").set_value("Who is the president?").run()
at.sidebar.text_input[0].set_value("gsk_dummy_value").run()
button(at, "Answer").click().run()
check("TEST 8: asking after clear -> 'build first' error", any("Add to knowledge base" in e for e in texts(at, "error")))

print(f"\n{sum(results)}/{len(results)} UI checks passed")
sys.exit(0 if all(results) else 1)
