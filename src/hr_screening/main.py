"""Command line entry point.

Examples:
    python -m hr_screening.main --resume data/incoming/cv.docx --job-file job.txt
    python -m hr_screening.main --batch-dir data/incoming --job-file job.txt --output-dir out

Exit codes: 0 success. 2 an error, or (batch) nothing could be screened. 1 (batch only) some
files were screened and some failed.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from hr_screening.batch import screen_batch, write_batch_outputs
from hr_screening.config import Settings, get_settings
from hr_screening.errors import (
    InvalidJobDescriptionError,
    LLMConfigError,
    OutputWriteError,
    ScreeningError,
)
from hr_screening.evaluation.batch_formatting import (
    batch_json,
    batch_text,
    resume_json,
    resume_text,
)
from hr_screening.evaluation.formatting import format_report_text
from hr_screening.llm.base import LLMClient
from hr_screening.llm.factory import build_llm_client
from hr_screening.llm.ollama_client import base_url_is_local
from hr_screening.logging_config import configure_logging
from hr_screening.models.batch import BatchSummary
from hr_screening.models.report import ScreeningReport
from hr_screening.pipeline import screen_resume
from hr_screening.rag.embeddings import Embedder
from hr_screening.rag.factory import build_embedder

MAX_JOB_FILE_BYTES = 200_000


class _StoreOnce(argparse.Action):
    """Store a value, and refuse a repeat. argparse would silently keep only the last one."""

    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"{option_string} was given more than once. Give it once per run.")
        setattr(namespace, self.dest, values)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hr-screening",
        description=(
            "Screen one resume, or every resume in a folder, against one job description. "
            "Decision support only: a person makes every hiring decision."
        ),
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--resume", action=_StoreOnce, help="Path to one .pdf or .docx resume")
    target.add_argument(
        "--batch-dir",
        action=_StoreOnce,
        help=(
            "Screen every file directly inside this folder (sub folders are not searched). "
            "Files are read in file name order. A failed file is reported and does not stop "
            "the batch. The result is a summary ordered for human review."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--job-file", action=_StoreOnce, help="Path to a text file with the job description"
    )
    source.add_argument("--job-text", action=_StoreOnce, help="Job description as text")
    parser.add_argument(
        "--resume-dir",
        action=_StoreOnce,
        help="With --resume: the resume must be inside this folder",
    )
    parser.add_argument(
        "--output-dir",
        action=_StoreOnce,
        help=(
            "With --batch-dir: also save batch_summary.json, batch_summary.txt and one report "
            "per resume (in a reports folder) here. Must be a different folder from --batch-dir. "
            "Without it, nothing is written to disk and the summary is only printed."
        ),
    )
    parser.add_argument(
        "--extra-skill",
        action="append",
        metavar="SKILL",
        help=(
            "Add a required skill that the built in vocabulary does not know, for example "
            "--extra-skill Terraform. Can be given more than once. It is matched as a whole "
            "word in the resume, for this run only."
        ),
    )
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
        action=_StoreOnce,
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


def _check_output_dir(batch_dir: str, output_dir: str) -> None:
    """Stop before screening if the output folder would clash with the resume folder."""
    try:
        target = Path(output_dir).resolve()
        source = Path(batch_dir).resolve()
    except OSError as exc:
        raise OutputWriteError("The output folder could not be used.") from exc
    if target == source:
        raise OutputWriteError("--output-dir must be a different folder from --batch-dir.")
    if target.exists() and not target.is_dir():
        raise OutputWriteError("--output-dir is a file, not a folder.")


def _batch_outputs(summary: BatchSummary) -> dict[str, str]:
    files = {
        "batch_summary.json": batch_json(summary),
        "batch_summary.txt": batch_text(summary),
    }
    for result in summary.results:
        files[f"reports/{result.source_file}.report.json"] = resume_json(result)
        files[f"reports/{result.source_file}.report.txt"] = resume_text(result)
    return files


def _batch_exit_code(summary: BatchSummary) -> int:
    if summary.counts.failed == 0:
        return 0
    return 1 if summary.counts.screened else 2


def _run_batch(args: argparse.Namespace, job_text: str) -> int:
    if args.output_dir is not None:
        _check_output_dir(args.batch_dir, args.output_dir)
    client = _build_client() if args.llm else None
    embedder = _build_embedder() if args.rag == "semantic" else None
    summary = screen_batch(
        args.batch_dir,
        job_text,
        client=client,
        rag_mode=args.rag,
        embedder=embedder,
        extra_skills=args.extra_skill or (),
        exclude=[args.job_file] if args.job_file else (),
    )
    print(batch_json(summary) if args.json else batch_text(summary))
    if args.output_dir is not None:
        written = write_batch_outputs(args.output_dir, _batch_outputs(summary))
        print(f"Saved {len(written)} file(s) to {args.output_dir}", file=sys.stderr)
    return _batch_exit_code(summary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.batch_dir is not None and args.resume_dir is not None:
        parser.error("--resume-dir is for --resume. A batch only reads its own folder.")
    if args.batch_dir is None and args.output_dir is not None:
        parser.error("--output-dir is only for --batch-dir.")
    _use_utf8_output()
    try:
        configure_logging()
    except ValidationError as exc:
        print(f"Error: {_invalid_settings_message(exc)}", file=sys.stderr)
        return 2
    try:
        job_text = args.job_text if args.job_text is not None else _read_job_file(args.job_file)
        if args.batch_dir is not None:
            return _run_batch(args, job_text)
        client = _build_client() if args.llm else None
        embedder = _build_embedder() if args.rag == "semantic" else None
        report = screen_resume(
            args.resume,
            job_text,
            client=client,
            base_dir=args.resume_dir,
            rag_mode=args.rag,
            embedder=embedder,
            extra_skills=args.extra_skill or (),
        )
    except ScreeningError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(_report_json(report) if args.json else format_report_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
