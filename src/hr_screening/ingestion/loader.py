"""Validated loading of resume files (PDF and DOCX) into plain text."""

import logging
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from pypdf import PdfReader

from hr_screening.errors import (
    EmptyFileError,
    NoExtractableTextError,
    ScreeningError,
    UnreadableFileError,
    UnsupportedFileTypeError,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".pdf", ".docx")
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_PDF_PAGES = 30
MAX_TEXT_CHARS = 200_000
_MAX_DOCX_XML_BYTES = 20 * 1024 * 1024
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def load_resume_text(
    path: str | Path,
    *,
    base_dir: str | Path | None = None,
    max_bytes: int = MAX_FILE_BYTES,
) -> str:
    """Validate a resume file and return its text.

    If ``base_dir`` is given, the file must live inside it.
    Never logs the path or the content.
    """
    file_path = _validate(Path(path), base_dir, max_bytes)
    suffix = file_path.suffix.lower()
    raw = _read_pdf(file_path) if suffix == ".pdf" else _read_docx(file_path)
    text = _clean(raw)
    if not text:
        if suffix == ".pdf":
            raise NoExtractableTextError(
                "The PDF has no extractable text. It is probably a scanned image, "
                "so OCR would be required. Provide a text based PDF or a DOCX."
            )
        raise NoExtractableTextError("The DOCX contains no text.")
    logger.info("Extracted %d characters from a %s resume", len(text), suffix)
    return text[:MAX_TEXT_CHARS]


def _validate(path: Path, base_dir: str | Path | None, max_bytes: int) -> Path:
    suffix = path.suffix.lower()
    if suffix == ".doc":
        raise UnsupportedFileTypeError(
            "Legacy .doc files are not supported. Save the resume as .docx or .pdf."
        )
    if suffix not in SUPPORTED_EXTENSIONS:
        shown = suffix or "none"
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{shown}'. Supported types: .pdf, .docx."
        )
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise UnreadableFileError("The resume file was not found or is not accessible.") from exc
    if base_dir is not None and not resolved.is_relative_to(Path(base_dir).resolve()):
        raise UnreadableFileError("The resume path is outside the allowed directory.")
    if not resolved.is_file():
        raise UnreadableFileError("The resume path is not a regular file.")
    size = resolved.stat().st_size
    if size == 0:
        raise EmptyFileError("The resume file is empty (0 bytes).")
    if size > max_bytes:
        raise UnreadableFileError(f"The resume file is larger than the {max_bytes} byte limit.")
    if suffix == ".pdf":
        with resolved.open("rb") as handle:
            looks_valid = handle.read(5) == b"%PDF-"
    else:
        looks_valid = zipfile.is_zipfile(resolved)
    if not looks_valid:
        raise UnreadableFileError(f"The file content does not look like a valid {suffix} file.")
    return resolved


def _read_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise UnreadableFileError("The PDF is password protected. Provide an unprotected copy.")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise UnreadableFileError(f"The PDF has more than {MAX_PDF_PAGES} pages.")
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except ScreeningError:
        raise
    except Exception as exc:
        raise UnreadableFileError("The PDF could not be read. It may be corrupted.") from exc


def _read_docx(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            try:
                info = archive.getinfo("word/document.xml")
            except KeyError as exc:
                raise UnreadableFileError("The file is not a valid Word document.") from exc
            if info.file_size > _MAX_DOCX_XML_BYTES:
                raise UnreadableFileError("The Word document body is unexpectedly large.")
            xml_bytes = archive.read(info)
        root = ET.fromstring(xml_bytes)
    except ScreeningError:
        raise
    except (zipfile.BadZipFile, ET.ParseError, OSError) as exc:
        raise UnreadableFileError("The DOCX could not be read. It may be corrupted.") from exc

    paragraphs = []
    for paragraph in root.iter(f"{_W}p"):
        parts = []
        for node in paragraph.iter():
            if node.tag == f"{_W}t" and node.text:
                parts.append(node.text)
            elif node.tag == f"{_W}tab":
                parts.append(" ")
            elif node.tag == f"{_W}br":
                parts.append("\n")
        paragraphs.append("".join(parts))
    return "\n".join(paragraphs)


def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()