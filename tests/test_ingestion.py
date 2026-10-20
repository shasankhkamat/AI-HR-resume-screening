import zipfile

import pytest

from hr_screening.errors import (
    EmptyFileError,
    NoExtractableTextError,
    UnreadableFileError,
    UnsupportedFileTypeError,
)
from hr_screening.ingestion.loader import load_resume_text


def test_docx_text_is_extracted(tmp_path, make_docx, sample_resume_lines):
    path = make_docx(tmp_path / "cv.docx", sample_resume_lines)
    text = load_resume_text(path)
    assert "Python" in text
    assert "Jan 2019" in text


def test_pdf_text_is_extracted(tmp_path, make_pdf, sample_resume_lines):
    path = make_pdf(tmp_path / "cv.pdf", sample_resume_lines)
    assert "Python" in load_resume_text(path)


def test_unsupported_extension(tmp_path):
    with pytest.raises(UnsupportedFileTypeError, match="Unsupported"):
        load_resume_text(tmp_path / "cv.txt")


def test_legacy_doc_is_rejected_with_advice(tmp_path):
    with pytest.raises(UnsupportedFileTypeError, match="docx"):
        load_resume_text(tmp_path / "cv.doc")


def test_missing_file(tmp_path):
    with pytest.raises(UnreadableFileError, match="not found"):
        load_resume_text(tmp_path / "missing.pdf")


def test_directory_is_not_a_file(tmp_path):
    (tmp_path / "folder.pdf").mkdir()
    with pytest.raises(UnreadableFileError, match="regular file"):
        load_resume_text(tmp_path / "folder.pdf")


def test_empty_file(tmp_path):
    path = tmp_path / "cv.pdf"
    path.write_bytes(b"")
    with pytest.raises(EmptyFileError):
        load_resume_text(path)


def test_content_must_match_extension(tmp_path):
    path = tmp_path / "cv.pdf"
    path.write_bytes(b"this is not a pdf")
    with pytest.raises(UnreadableFileError, match="does not look like"):
        load_resume_text(path)


def test_corrupt_docx(tmp_path):
    path = tmp_path / "cv.docx"
    path.write_bytes(b"not a zip file")
    with pytest.raises(UnreadableFileError):
        load_resume_text(path)


def test_zip_without_word_document(tmp_path):
    path = tmp_path / "cv.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("other.txt", "hello")
    with pytest.raises(UnreadableFileError, match="not a valid Word"):
        load_resume_text(path)


def test_pdf_without_text_mentions_ocr(tmp_path, make_blank_pdf):
    with pytest.raises(NoExtractableTextError, match="OCR"):
        load_resume_text(make_blank_pdf(tmp_path / "scan.pdf"))


def test_docx_without_text(tmp_path, make_docx):
    with pytest.raises(NoExtractableTextError):
        load_resume_text(make_docx(tmp_path / "cv.docx", []))


def test_size_limit(tmp_path, make_pdf):
    path = make_pdf(tmp_path / "cv.pdf", ["Python"])
    with pytest.raises(UnreadableFileError, match="larger"):
        load_resume_text(path, max_bytes=10)


def test_base_dir_blocks_files_outside(tmp_path, make_docx):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = make_docx(tmp_path / "cv.docx", ["Python"])
    with pytest.raises(UnreadableFileError, match="outside"):
        load_resume_text(outside, base_dir=allowed)


def test_base_dir_allows_files_inside(tmp_path, make_docx):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    inside = make_docx(allowed / "cv.docx", ["Python"])
    assert "Python" in load_resume_text(inside, base_dir=allowed)