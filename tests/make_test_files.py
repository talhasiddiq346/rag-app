"""Creates the PDF / DOCX / edge-case files used by the tests (run once)."""
from pathlib import Path
import docx
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import textwrap

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "test_files"
OUT.mkdir(exist_ok=True)

PDF_TEXT = """Karachi Robotics Club - Annual Report 2025 (fictional test document).
The Karachi Robotics Club has 75 members and meets every Wednesday at 4 pm in Lab 3.
In 2025 the club built a line-following robot named Falcon-7, which won second place
at the National Student Robotics Challenge held in Lahore. The club's annual budget
is 900,000 rupees and its president is Ayesha Khan. The club plans to build an
underwater drone in 2026."""

DOCX_TEXT = """Campus Cafeteria Policy (fictional test document).
The cafeteria opens at 8:00 am and closes at 7:30 pm from Monday to Saturday.
It is closed on Sundays. Students receive a 15 percent discount when they show
their student ID card. The most popular dish is chicken biryani, which costs 450 rupees.
Single-use plastic straws were banned in the cafeteria in August 2025."""

# 1. Text-based PDF
c = canvas.Canvas(str(OUT / "robotics_report.pdf"), pagesize=A4)
y = 800
for line in PDF_TEXT.splitlines():
    for part in textwrap.wrap(line, 95) or [""]:
        c.drawString(40, y, part); y -= 16
c.save()

# 2. "Scanned" PDF: a page with only a drawn rectangle, no text layer
c = canvas.Canvas(str(OUT / "scanned_image_only.pdf"), pagesize=A4)
c.rect(100, 400, 300, 200, fill=1); c.save()

# 3. DOCX (paragraphs + a table)
d = docx.Document()
for line in DOCX_TEXT.splitlines():
    d.add_paragraph(line)
t = d.add_table(rows=2, cols=2)
t.cell(0, 0).text = "Manager"; t.cell(0, 1).text = "Bilal Ahmed"
t.cell(1, 0).text = "Phone extension"; t.cell(1, 1).text = "2214"
d.save(OUT / "cafeteria_policy.docx")

# 4. Empty TXT, 5. corrupted PDF, 6. unsupported type
(OUT / "empty.txt").write_text("")
(OUT / "corrupted.pdf").write_bytes(b"this is not really a pdf")
(OUT / "notes.csv").write_text("a,b\n1,2\n")
print("Created:", sorted(p.name for p in OUT.iterdir()))

# 7-8. Two similar PDFs for comparison tests (fictional universities, several chunks each)
def long_pdf(path, title, facts, filler_topic):
    filler = (f"The {filler_topic} is open to all students and staff and is described here only to make "
              "the document longer, so that it is split into several chunks like a real report. ") * 6
    paras = [title] + facts[:3] + [filler] + facts[3:] + [filler.replace("open to", "available for")]
    c = canvas.Canvas(str(path), pagesize=A4); y = 800
    for p in paras:
        for part in textwrap.wrap(p, 95):
            if y < 60: c.showPage(); y = 800
            c.drawString(40, y, part); y -= 15
        y -= 10
    c.save()

long_pdf(OUT / "northfield_admissions.pdf", "Northfield Institute of Technology - Admissions Guide 2026 (fictional).", [
    "Tuition fee at Northfield is 250,000 rupees per semester for all undergraduate programmes.",
    "The application deadline for the fall intake is 15 July 2026.",
    "Applicants need a minimum entry test score of 60 percent to be considered.",
    "Northfield offers a merit scholarship covering 50 percent of tuition for the top 10 percent of applicants.",
    "The hostel fee is 40,000 rupees per month including meals, and the campus has 6,000 students.",
], "sports complex at Northfield")
long_pdf(OUT / "riverside_admissions.pdf", "Riverside University - Admissions Guide 2026 (fictional).", [
    "Tuition fee at Riverside is 310,000 rupees per semester for all undergraduate programmes.",
    "The application deadline for the fall intake is 30 June 2026.",
    "Applicants need a minimum entry test score of 70 percent to be considered.",
    "Riverside offers a need-based scholarship covering 100 percent of tuition for students from low-income families.",
    "The hostel fee is 35,000 rupees per month without meals, and the campus has 11,000 students.",
], "central library at Riverside")
print("Created comparison PDFs")
