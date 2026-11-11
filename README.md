# 📚 Open-Source RAG Application

**Talha Siddiq** (b04-0923-0038) · **Ajwad Muhammad Waseem** (b04-0923-000004)

**CS4106 – Foundations of Generative AI · Session 9 Activity**
Built with **Groq · Sentence-Transformers · FAISS · Streamlit**

Upload your own documents (PDF, DOCX or TXT), ask a question in plain English, and the app
answers **from your documents**, not from the model's memory. If the answer isn't in the
documents, it says it doesn't know. Everything here is free: open-source libraries and Groq's free API tier.

---

## 1. What is RAG?

**Retrieval-Augmented Generation (RAG)** = *find the relevant passages first, then let the LLM
answer using only those passages.*

A plain LLM answers from what it memorised during training: it can be out of date and it can
make things up ("hallucinate"). A RAG app gives the LLM the right pages from **your** documents
and tells it to answer only from them.

```
                 ┌──────────── BUILD KNOWLEDGE BASE (once) ────────────┐
 PDF/DOCX/TXT ─► │ extract text ─► 1.CHUNK ─► 2.EMBED ─► 3.INDEX (FAISS)│
                 └──────────────────────────────────────────────────────┘
                                                             │
                 ┌──────────────── ANSWER (each question) ───▼─────────┐
   Question ───► │ 2.EMBED question ─► 4.RETRIEVE top-k ─► 5.GENERATE   │ ─► Answer + Sources
                 └──────────────────────────────────────────(Groq LLM)──┘
```

## 2. The five stages (and where they are in the code)

| # | Stage | What happens | Code |
|---|-------|--------------|------|
| 1 | **Chunk** | Text is split into passages of **150 words** with **30 words of overlap**, so a sentence cut at a boundary still appears whole in the neighbouring chunk. Chunk 1 = words 0–149, chunk 2 = words 120–269, … | `rag_engine.chunk_text()` |
| 2 | **Embed** | `all-MiniLM-L6-v2` turns each chunk into a list of **384 numbers** (a vector). Texts with similar meaning get vectors pointing in a similar direction. | `rag_engine.load_embedder()`, `embed()` |
| 3 | **Index** | Vectors are **normalised** (length = 1) and stored in a FAISS `IndexFlatIP`. For unit vectors, inner product = **cosine similarity**. A parallel list remembers each chunk's text and **source filename**. | `VectorStore.build()` |
| 4 | **Retrieve** | The question is embedded the same way, and FAISS returns the **top-k** most similar chunks (k = 3 by default, adjustable 1–5) with their similarity scores. | `VectorStore.search()` |
| 4b | **Compare** | For comparison questions ("compare…", "difference…", "which … cheaper") the top-k chunks are taken from **each** document, so every file is represented. | `VectorStore.search_per_document()`, `plan_retrieval()` |
| 5 | **Generate** | The retrieved chunks + the question are sent to a **Groq** LLM with a strict system prompt: *answer only from the context, otherwise say you don't know, never invent facts*. | `rag_engine.generate_answer()` |

**Similarity score:** ranges from −1 to 1. Around 0.5+ usually means a strong match; around 0.1 means unrelated.

## 3. Technologies

| Tool | Role | Cost |
|------|------|------|
| Python 3.12 / 3.13 | Language | Free |
| Streamlit | Web interface | Free, open source |
| Groq API | Runs open-weight LLMs very fast | Free tier, no credit card |
| Sentence-Transformers (`all-MiniLM-L6-v2`) | Embeddings, runs locally | Free, open source |
| FAISS (`faiss-cpu`) | Vector search index | Free, open source (Meta) |
| NumPy | Vector maths / normalisation | Free |
| PyPDF2 / python-docx | Read PDF and DOCX files | Free |

### ⚠️ Difference from the PDF: the Groq model name
The handout uses `llama-3.3-70b-versatile` and `llama-3.1-8b-instant`. **Groq has retired both**
(this is exactly the PDF's "model_decommissioned" watch-out). The app uses models that were live
on Groq's free tier when it was built (November 2026):

- `openai/gpt-oss-120b` (default: OpenAI's open-weight model, best quality)
- `openai/gpt-oss-20b` (faster)
- `qwen/qwen3.8-27b` (alternative open-weight model)

If one stops working, open <https://console.groq.com/docs/models>, copy a current chat model
name and put it in `GROQ_MODELS` in `config.py`. That's the only place you need to change it.

## 4. Project structure

```
rag-app/
├── app.py                  # Streamlit UI (sidebar, upload, ask, answer, sources)
├── rag_engine.py           # The 5 RAG stages: chunk, embed, index, retrieve, generate
├── document_processor.py   # Reads PDF / DOCX / TXT safely, keeps the filename
├── config.py               # All settings: models, chunk size, k, system prompt, API-key lookup
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .env.example            # Template for your local API key file
├── .gitignore              # Keeps .env, secrets and .venv out of git
├── .streamlit/
│   ├── config.toml               # Streamlit settings (upload limit, no telemetry)
│   └── secrets.toml.example      # Optional alternative to .env
├── sample_documents/
│   ├── sample.txt          # Fictional "Indus Green Campus Initiative" handbook
│   ├── sample.pdf          # Fictional robotics club report
│   ├── sample.docx         # Fictional cafeteria policy (includes a table)
│   ├── northfield_admissions.pdf  # Fictional university A  ┐ use these two to
│   └── riverside_admissions.pdf   # Fictional university B  ┘ test comparisons
└── tests/
    ├── make_test_files.py  # Generates the PDF/DOCX/edge-case test files
    ├── test_pipeline.py    # Tests the RAG engine directly (48 checks)
    ├── test_ui.py          # Simulates clicks in the Streamlit app (32 checks)
    └── test_files/         # PDF, DOCX, scanned PDF, empty, corrupted, unsupported files
```

The sample documents are about **fictional** organisations on purpose: the LLM can't know these
facts from training, so a correct answer proves it came from your document.

## 5. Installation on Windows

Open the `rag-app` folder in VS Code / Antigravity, then open a terminal (**Terminal → New Terminal**).
The commands below are for **PowerShell** (the default terminal in VS Code on Windows, already
built into Windows, so there's nothing to download).

### 5.1 Check Python
```powershell
python --version        # should print Python 3.12 or 3.13
```

### 5.2 Create and activate a virtual environment
A virtual environment keeps this project's libraries separate from the rest of your computer.
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
You should now see `(.venv)` at the start of the prompt.

> **"running scripts is disabled on this system"?** Run this once, then activate again:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
> Using Command Prompt (cmd) instead? Activate with `.venv\Scripts\activate.bat`.

### 5.3 Install the dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```
This takes a few minutes the first time (PyTorch is large). The embedding model (~90 MB) is
downloaded automatically the first time you build a knowledge base.

## 6. Groq API key setup

1. Go to <https://console.groq.com> → sign up (free) → **API Keys** → **Create API Key**.
2. Choose **one** way to give it to the app. **Never** write the key inside a `.py` file.

**Option A: type it in the app (simplest).** Paste it into the password box in the sidebar. It's
masked and kept only in memory for that browser session.

**Option B: a `.env` file (so you don't retype it).**
```powershell
Copy-Item .env.example .env
notepad .env
```
Replace `your_groq_api_key_here` with your key and save. `.env` is in `.gitignore`, so it's never
uploaded to GitHub. The sidebar will say *"Using GROQ_API_KEY from environment / secrets"*.

> 🔒 Treat the key like a password. Don't paste it into chats, screenshots or shared notebooks.
> If it's ever exposed, delete it in the Groq console and create a new one.

## 7. Run the app

```powershell
streamlit run app.py
```
Your browser opens at **http://localhost:8501**. Stop the app with **Ctrl + C** in the terminal.

## 8. How to use it

1. **Sidebar:** enter your Groq key (unless you used `.env`), pick a model, choose *k* (how many chunks to retrieve).
2. **Upload documents:** drag in one or more PDF / DOCX / TXT files (try the ones in `sample_documents/`).
3. Click **➕ Add to knowledge base**. You can repeat steps 2–3 to **add more files later**; earlier
   files are kept. The **📂 Knowledge base** panel lists every file with its chunk count and a **Remove** button.
4. **Ask a question**, e.g. *"How many solar panels were installed on the library?"*, then click **💬 Answer**.
5. Open **Sources** to see which chunks were used, which **file** each came from, and its **similarity score**.
6. Try a question that is **not** in the documents (e.g. *"What is the capital of Australia?"*). The app should say it doesn't know.

### Homework improvements included
- ✅ **Source file name** shown for every retrieved chunk
- ✅ **🗑️ Clear knowledge base** button (resets the index, chunks, answer and the uploader)
- ✅ **Chunk size and overlap** configurable in the sidebar (click *Add to knowledge base* to re-chunk)
- ✅ **Multi-document support**: add files in several rounds, remove single files, filter which files to search
- ✅ **Document comparison**: compare information across 2+ files (see section 8.1)

### 8.1 Working with several documents and comparing them

**Why plain top-k fails at comparisons:** with *k = 3*, the 3 most similar chunks often all come from
**one** file. Asked *"Compare the tuition fees of Northfield and Riverside"*, plain search returned 3
Riverside chunks and no Northfield chunk, so the LLM could not compare.

**What the app does instead (`Search mode` in the sidebar):**
| Mode | Behaviour |
|------|-----------|
| **Auto** (default) | Detects comparison questions ("compare", "difference", "vs", "both", "which … cheaper/higher/later", or two file names) and switches to per-document retrieval. Other questions use normal top-k. |
| Search all documents together | Always normal top-k over all selected files. |
| Compare documents side by side | Always takes the top-*k* chunks **from each** selected file. |

In comparison mode the context sent to Groq is **grouped by document**, and the model is asked to state
what **each file** says, then summarise similarities and differences (often as a table), and to say when
a file doesn't cover the topic.

**Tips:** use the **Documents to search** box to pick exactly which files to compare; mentioning a file
name in the question (e.g. *"riverside_admissions"*) also focuses on that file.

**Efficiency:** each file is embedded only once. Re-uploading an unchanged file is skipped, a changed
file with the same name replaces the old version, and a comparison runs **one** FAISS search for all files.

Good questions for the sample files:

| File | Question | Expected |
|------|----------|----------|
| sample.txt | What was the total budget for phase one and where did the money come from? | 18 million PKR; 60 % university, 25 % alumni, 15 % grant |
| sample.pdf | Who is the president of the robotics club? | Ayesha Khan |
| sample.docx | Who manages the cafeteria? | Bilal Ahmed (from the table) |
| northfield + riverside | Compare the tuition fees and hostel fees of the two universities | 250,000 vs 310,000 PKR/semester; hostel 40,000 (with meals) vs 35,000 (without) |
| northfield + riverside | Which university needs a higher entry test score? | Riverside (70 % vs 60 %) |
| any | Who founded Groq? | "I don't know" (not in the documents) |

## 9. Run the automated tests (optional)

```powershell
python tests\test_pipeline.py      # RAG engine: chunking, extraction, retrieval, Groq answers, errors
python tests\test_ui.py            # Simulated clicks through the Streamlit UI
```
The tests that call Groq run only when `GROQ_API_KEY` is set (e.g. via `.env`); the others still run without it.

## 10. Common errors and fixes

| Symptom | Cause / fix |
|---------|-------------|
| `streamlit` / `python` is not recognized | The venv isn't active. Run `.\.venv\Scripts\Activate.ps1`. If `python` itself isn't found, reinstall Python with **"Add python.exe to PATH"** ticked. |
| "running scripts is disabled" | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again. |
| `pip install` fails on faiss-cpu or torch | Use Python 3.12 or 3.13 (64-bit). Upgrade pip first: `python -m pip install --upgrade pip`. |
| "Invalid Groq API key" | Re-copy the key from console.groq.com; no spaces or quotes. |
| "The model … is not available on Groq any more" | Pick another model in the sidebar or update `GROQ_MODELS` in `config.py`. |
| "Groq rate limit reached" | Free-tier limit. Wait a minute, or use the smaller `gpt-oss-20b`. |
| "Could not reach Groq" | Check your internet / firewall / VPN. |
| "No extractable text found … OCR is not currently supported" | It's a scanned/image PDF. Use a text-based PDF, DOCX or TXT. |
| "Upload a document and click 'Add to knowledge base' first" | You asked before adding files, or you clicked *Clear knowledge base*. |
| Answer says "I don't know" but the info *is* in the file | Retrieval missed it. Increase *k*, rephrase using words from the document, untick unrelated files in *Documents to search*, or try a smaller chunk size and re-add. |
| A comparison only talks about one file | Choose *Compare documents side by side* in the sidebar, and select the files in *Documents to search*. |
| Only my last uploaded file is used | Fixed: files are now **added** to the knowledge base. Check the 📂 panel to see which files are loaded. |
| "Overlap must be smaller than the chunk size" | Lower the overlap in the sidebar. |
| Port 8501 already in use | Close the other Streamlit window, or `streamlit run app.py --server.port 8502`. |
| First "Add to knowledge base" is slow | The embedding model downloads once (~90 MB); later adds are fast. |

## 11. Deployment (homework item 1)

The handout says to deploy to a Hugging Face Space with the Streamlit SDK. Two facts matter here:
- Hugging Face **deprecated its built-in Streamlit SDK in April 2025**. Streamlit apps there now
  have to use the Docker SDK.
- This project instead targets **Streamlit Community Cloud**, a free host made for Streamlit apps.
  It needs no code changes.

**Steps (you do these yourself, with your own accounts):**
1. Push the `rag-app` folder to a GitHub repository. `.gitignore` already keeps `.env` and
   `.venv` out. Double-check that `.env` is **not** in the repo before pushing.
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **Create app**, choose your
   repo and branch, and set **Main file path** to `app.py`.
3. Open **Advanced settings**, choose Python **3.12**, and paste into **Secrets**:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
4. Click **Deploy**. The first build installs PyTorch and takes a few minutes. The app reads the key
   through `st.secrets` (see `config.get_api_key_from_environment()`), so the sidebar shows
   *"Using GROQ_API_KEY from environment / secrets"*.

> Note: on a public deployment, everyone who opens the link uses **your** Groq key (and your
> free-tier rate limit). For a class demo that's fine. Otherwise, leave the secret out and let
> each user type their own key in the sidebar.

## 12. Known limitations

- **No OCR**: scanned/image-only PDFs can't be read (the app tells the user).
- **Session-only memory**: the knowledge base lives in the browser session. Refreshing the page
  or clicking *Clear* removes it; nothing is saved to disk.
- **Word-based chunking** can still split a table or list awkwardly; overlap reduces but doesn't remove this.
- `all-MiniLM-L6-v2` is optimised for **English**; other languages retrieve less accurately.
- Groq's **free tier has rate limits**; heavy use may need a short wait.
- Groq model names change over time (see section 3).
