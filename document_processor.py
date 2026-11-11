"""
document_processor.py - Turns uploaded files (PDF, DOCX, TXT) into plain text.

Every file produces a result dict so one bad file never crashes the app:
    {"filename": str, "text": str, "pages": [str, ...], "error": str | None}
"pages" keeps the PDF page boundaries so answers can cite page numbers
(DOCX/TXT have no pages, so they are returned as a single "page").
"""

import io
import re
import unicodedata
from typing import Dict, List, Optional

import docx

# pypdf is the maintained successor of PyPDF2 (same API, same authors). It extracts
# tables and multi-column pages more accurately (e.g. "28" and "5" are not glued into
# "285"). If it is not installed we fall back to PyPDF2, the library named in the handout.
try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

from config import SUPPORTED_EXTENSIONS


def _extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def clean_text(text: str) -> str:
    """Fix common PDF extraction artefacts without changing the meaning."""
    text = unicodedata.normalize("NFKC", text)           # ligatures: "ﬁ" -> "fi"
    text = re.sub(r"/([a-z])\.([a-z])\2", r"\1\2", text)   # glyph names: "Su/f.ffa" -> "Suffa"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # "environ-\nment" -> "environment"
    text = re.sub(r"[ \t\u00a0]+", " ", text)              # collapse runs of spaces
    return text.strip()


def _read_pdf_pages(data: bytes) -> List[str]:
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        try:
            reader.decrypt("")  # many "encrypted" PDFs have an empty password
        except Exception:
            raise ValueError("PDF is password-protected.")
    return [clean_text(page.extract_text() or "") for page in reader.pages]


def _read_docx(data: bytes) -> str:
    document = docx.Document(io.BytesIO(data))
    parts = [p.text for p in document.paragraphs]
    # Text inside tables is not part of .paragraphs, so collect it too.
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def _read_txt(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "ignore")


def extract_text(filename: str, data: bytes) -> Dict[str, Optional[str]]:
    """Extract text from one file. Never raises - errors are returned in 'error'."""
    ext = _extension(filename)
    result = {"filename": filename, "text": "", "pages": [], "error": None}

    if ext not in SUPPORTED_EXTENSIONS:
        result["error"] = f"Unsupported file type '.{ext}'. Use PDF, DOCX or TXT."
        return result
    if not data:
        result["error"] = "The file is empty."
        return result

    try:
        if ext == "pdf":
            pages = _read_pdf_pages(data)
        elif ext == "docx":
            pages = [clean_text(_read_docx(data))]
        else:
            pages = [clean_text(_read_txt(data))]
    except Exception as exc:  # corrupted / renamed / protected files
        result["error"] = f"Could not read this {ext.upper()} file ({type(exc).__name__}: {exc})."
        return result

    text = "\n".join(pages)
    text = text.strip()
    if not text:
        if ext == "pdf":
            result["error"] = (
                "No extractable text found. This looks like a scanned/image PDF - "
                "OCR is not currently supported. Please upload a text-based PDF."
            )
        else:
            result["error"] = "The file contains no readable text."
        return result

    result["text"] = text
    result["pages"] = pages
    return result


def process_uploaded_files(uploaded_files) -> List[Dict[str, Optional[str]]]:
    """Process Streamlit UploadedFile objects (anything with .name and .getvalue())."""
    return [extract_text(f.name, f.getvalue()) for f in uploaded_files]
