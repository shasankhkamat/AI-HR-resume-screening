# AI-Powered HR Resume Screening System

An AI-powered HR resume screening and candidate evaluation system planned around a Retrieval-Augmented Generation (RAG) architecture.

## Current Status

**Milestone 1: Project scaffolding.**

Resume parsing, embeddings, ChromaDB indexing, candidate retrieval, LLM evaluation, and API endpoints are planned but not implemented yet.

## Planned Architecture

1. Ingest PDF and DOCX resumes from a monitored folder.
2. Extract resume text and preserve candidate metadata.
3. Chunk text, generate embeddings, and index it in persistent ChromaDB.
4. Convert recruiter requirements into filters and semantic queries.
5. Retrieve relevant candidates and generate evidence-grounded evaluations.
6. Expose workflows through FastAPI and produce Markdown reports.

## Technology Stack

- Python 3.11+
- FastAPI
- Pydantic v2
- LangChain
- ChromaDB
- PyMuPDF
- OpenAI-compatible LLM and embedding APIs
- pytest and Ruff

Application dependencies will be introduced incrementally.

## Repository Structure

```text
ai-hr-resume-screening/
├── .github/workflows/
├── data/
│   ├── incoming/
│   ├── processed/
│   └── chroma/
├── docs/
├── logs/
├── src/hr_screening/
├── tests/
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

## Prerequisites

- Python 3.11 or newer
- Git

## Setup — Windows PowerShell

Run from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Setup — macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Verification

```bash
python -c "from hr_screening import __version__; print(__version__)"
python -m pytest
python -m ruff check src tests
```

## Configuration

Copy `.env.example` to `.env` when local configuration is needed. Configuration loading has not been implemented yet. Never commit API keys or `.env`.

## Security and Privacy

- Never commit real resumes, extracted personal information, or logs containing personal data.
- Keep credentials in local environment files and never commit secrets.
- Keep local databases and candidate datasets out of version control.
- Use synthetic candidate data in automated tests.
- Treat model-generated assessments as decision-support information, with human oversight for hiring decisions.

## Development Roadmap

1. Project scaffolding and Git setup.
2. Configuration and settings management.
3. Resume parsing.
4. Text chunking and metadata.
5. Embeddings and ChromaDB indexing.
6. Automated resume ingestion.
7. Recruiter query decomposition.
8. Hybrid candidate retrieval.
9. Evidence-grounded LLM evaluation.
10. Structured scorecards and citation validation.
11. FastAPI endpoints.
12. Markdown reports.
13. Testing and quality evaluation.
14. Security, privacy, and deployment hardening.

## License

No license has been selected yet.
