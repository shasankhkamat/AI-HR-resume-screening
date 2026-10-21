"""Command line entry point.

Example: python -m hr_screening.main --resume data/incoming/cv.docx --job-file job.txt
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from hr_screening.config import Settings, get_settings
from hr_screening.errors import InvalidJobDescriptionError, LLMConfigError, ScreeningError
from hr_screening.evaluation.formatting import format_report_text
from hr_screening.llm.base import LLMClient
from hr_screening.llm.factory import build_llm_client
from hr_screening.llm.ollama_client import base_url_is_local
from hr_screening.logging_config import configure_logging
from hr_screening.models.report import ScreeningReport
from hr_screening.pipeline import screen_resume
from hr_screening.rag.embeddings import Embedder
from hr_screening.rag.factory import build_embedder

MAX_JOB_FILE_BYTES = 200_000


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hr-screening",
        description="Screen one resume against one job description. Decision support only.",
    )
    parser.add_argument("--resume", required=True, help="Path to a .pdf or .docx resume")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--job-file", help="Path to a text file with the job description")
    source.add_argument("--job-text", help="Job description as text")
    parser.add_argument("--resume-dir", help="If set, the resume must be inside this folder")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    parser.add_argument(
        "--llm",
        action="store_true",
        help=(
            "Opt in to extraction with a local Ollama model (see OLLAMA_BASE_URL and LLM_MODEL). "
            "Resume text is sent to that server. If the model fails, rules are used instead."
        ),
    )
    parser.add_argument(
        "--rag",
        choices=["lexical", "semantic"],
        help=(
            "Opt in to listing resume passages for each requirement, for human review. "
            "'lexical' is a local keyword search and uses no network. 'semantic' ranks "
            "passages with an Ollama embedding model (see EMBEDDING_MODEL) and sends resume "
            "text to that server. Passages do not change any status or the score."
        ),
    )
    return parser


def _report_json(report: ScreeningReport) -> str:
    """The report as JSON. Without retrieval the output has no retrieval key, as before."""
    exclude = None if report.retrieval else {"retrieval"}
    return report.model_dump_json(indent=2, exclude=exclude)


def _invalid_settings_message(exc: ValidationError) -> str:
    """Name the invalid settings. The invalid values themselves are never shown."""
    names = sorted({str(item["loc"][0]).upper() for item in exc.errors() if item.get("loc")})
    return f"Invalid value for: {', '.join(names)}. Check your environment variables or .env."


def _settings() -> Settings:
    """The application settings. An invalid value raises LLMConfigError naming the setting."""
    try:
        return get_settings()
    except ValidationError as exc:
        raise LLMConfigError(_invalid_settings_message(exc)) from None


def _warn_if_remote(settings: Settings) -> None:
    if not base_url_is_local(settings.ollama_base_url):
        print(
            "Warning: OLLAMA_BASE_URL is not a local address. "
            "Resume text will be sent to that server.",
            file=sys.stderr,
        )


def _build_client() -> LLMClient:
    """Build the LLM client from settings. Configuration problems raise LLMConfigError."""
    settings = _settings()
    client = build_llm_client(settings)
    _warn_if_remote(settings)
    return client


def _build_embedder() -> Embedder:
    """Build the embedder from settings. Configuration problems raise LLMConfigError."""
    settings = _settings()
    embedder = build_embedder(settings)
    _warn_if_remote(settings)
    return embedder


def _read_job_file(path: str) -> str:
    file_path = Path(path)
    try:
        if file_path.stat().st_size > MAX_JOB_FILE_BYTES:
            raise InvalidJobDescriptionError("The job description file is too large.")
        return file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise InvalidJobDescriptionError("The job description file could not be read.") from exc


def _use_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _use_utf8_output()
    try:
        configure_logging()
    except ValidationError as exc:
        print(f"Error: {_invalid_settings_message(exc)}", file=sys.stderr)
        return 2
    try:
        job_text = args.job_text if args.job_text is not None else _read_job_file(args.job_file)
        client = _build_client() if args.llm else None
        embedder = _build_embedder() if args.rag == "semantic" else None
        report = screen_resume(
            args.resume,
            job_text,
            client=client,
            base_dir=args.resume_dir,
            rag_mode=args.rag,
            embedder=embedder,
        )
    except ScreeningError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(_report_json(report) if args.json else format_report_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())