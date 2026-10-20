import zipfile
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

import pytest
from pypdf import PdfWriter

SAMPLE_RESUME_LINES = [
    "SKILLS",
    "Python, SQL, Pandas, scikit-learn, Power BI",
    "EXPERIENCE",
    "Data Analyst, Example Corp   Jan 2019 - Dec 2021",
    "Built weekly reporting dashboards in Power BI",
    "Senior Analyst, Sample Ltd   Jan 2022 - Present",
    "EDUCATION",
    "B.Tech in Computer Science, Example Institute, 2018",
    "CERTIFICATIONS",
    "Example Cloud Practitioner",
    "PROJECTS",
    "Churn prediction model using scikit-learn",
]

SAMPLE_JOB = """Data Analyst

About the role
We build analytics for clients.

Requirements
- 3+ years of experience in analytics
- Strong Python and SQL skills
- Experience with Tableau
- Bachelor's degree in a quantitative field
- Data Engineering certification

Preferred
- Knowledge of Spark
- Master's degree
"""

NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _pdf_bytes(lines: list[str]) -> bytes:
    def esc(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    content = ["BT", "/F1 11 Tf", "14 TL", "50 750 Td"]
    for line in lines:
        content += [f"({esc(line)}) Tj", "T*"]
    content.append("ET")
    stream = "\n".join(content).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_start = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF\n"
    ).encode()
    return bytes(out)


@pytest.fixture
def today() -> date:
    return date(2026, 1, 1)


@pytest.fixture
def sample_resume_lines() -> list[str]:
    return list(SAMPLE_RESUME_LINES)


@pytest.fixture
def sample_resume_text() -> str:
    return "\n".join(SAMPLE_RESUME_LINES)


@pytest.fixture
def sample_job() -> str:
    return SAMPLE_JOB


@pytest.fixture
def make_docx():
    def _make(path: Path, paragraphs: list[str]) -> Path:
        body = "".join(f"<w:p><w:r><w:t>{escape(p)}</w:t></w:r></w:p>" for p in paragraphs)
        document = f'<w:document xmlns:w="{NAMESPACE}"><w:body>{body}</w:body></w:document>'
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("word/document.xml", document)
        return path

    return _make


@pytest.fixture
def make_pdf():
    def _make(path: Path, lines: list[str]) -> Path:
        path.write_bytes(_pdf_bytes(lines))
        return path

    return _make


@pytest.fixture
def make_blank_pdf():
    def _make(path: Path) -> Path:
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        with path.open("wb") as handle:
            writer.write(handle)
        return path

    return _make