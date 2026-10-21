# AI HR Resume Screening

An explainable resume screening tool that compares resumes against a job description and produces evidence-based reports. It supports single-resume screening and batch processing, identifies duplicate resumes, and organizes results for human review.

**Responsible use:** This tool provides decision support only. It does not accept, reject, or recommend candidates. Every hiring decision must be made by a human reviewer. A coverage score measures evidence found for listed requirements, not a candidate's overall ability or suitability. No fairness, bias, or compliance testing has been performed.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Screening Multiple Resumes](#screening-multiple-resumes)
- [Understanding the Report](#understanding-the-report)
- [Requirements and Matching Rules](#requirements-and-matching-rules)
- [Optional LLM Extraction](#optional-llm-extraction)
- [Retrieved Passages](#retrieved-passages)
- [Testing and Verification](#testing-and-verification)
- [Privacy and Data Handling](#privacy-and-data-handling)
- [Limitations](#limitations)
- [Project Structure](#project-structure)

## Features

- **Resume ingestion:** Reads text-based PDF and DOCX files, with file validation and extraction checks.
- **Requirement extraction:** Identifies skills, experience, education, and certifications from a job description.
- **Evidence-based assessment:** Shows the exact resume text supporting each assessment.
- **Transparent scoring:** Calculates a weighted requirement coverage score.
- **Skill matching:** Supports aliases, version suffixes, related skills, and basic detection of negated or aspirational skill mentions.
- **Experience and education checks:** Applies rule-based matching, including checks for stated experience durations and degree completion.
- **Optional local LLM extraction:** Uses a locally configured Ollama model, with validation and a rule-based fallback for supported failures.
- **Optional retrieval:** Adds relevant resume passages using lexical or semantic search without changing the assessment or score.
- **Batch screening:** Processes a folder of resumes, isolates file-level failures, and produces a summary for human review.
- **Duplicate detection:** Flags identical resume text after whitespace normalization.
- **Automated quality checks:** Includes pytest tests, Ruff checks, and a synthetic retrieval evaluation.

## Requirements

- Python 3.11 or newer
- Windows, macOS, or Linux
- The project dependencies installed in the steps below

Optional features:

| Feature | Additional requirement |
|---|---|
| Default screening | No model or network access required |
| Lexical retrieval | No model or network access required |
| LLM extraction (`--llm`) | Local Ollama server and an installed chat model |
| Semantic retrieval (`--rag semantic`) | Local Ollama server and an installed embedding model |

## Installation

Open a terminal in the project directory.

### 1. Create and activate a virtual environment

PowerShell on Windows:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install the project

```bash
python -m pip install -e ".[dev]"
```

This installs the application, its dependencies, and the development tools used for testing and linting.

### 3. Check the command-line interface

```powershell
python -m hr_screening.main --help
```

The project also provides the `hr-screening` command when installed.

## Quick Start

The default mode uses rule-based extraction and does not require Ollama.

### 1. Create a sample resume

```powershell
python docs\make_sample_resume.py
```

This creates a synthetic sample resume at `data\incoming\sample_resume.docx`.

### 2. Screen one resume

```powershell
python -m hr_screening.main --resume data\incoming\sample_resume.docx --job-text "Requirements`n- 3+ years of experience`n- Python and SQL`n- Bachelor's degree"
```

### 3. Use a job description file

Create a job description:

```powershell
@"
Job Title: Data Analyst

Required Skills:
- Python
- SQL

Required Education:
- Bachelor's degree

Experience:
- 3+ years of experience
"@ | Set-Content -Encoding utf8 data\incoming\job.txt
```

Run the screening:

```powershell
python -m hr_screening.main --resume data\incoming\sample_resume.docx --job-file data\incoming\job.txt --json
```

The report includes each assessed requirement, its status, supporting evidence, explanation, coverage score, and limitations.

### Common options

| Option | Purpose |
|---|---|
| `--resume PATH` | Screen one resume |
| `--job-file PATH` | Read the job description from a file |
| `--job-text TEXT` | Supply the job description directly |
| `--json` | Print structured JSON output |
| `--llm` | Enable optional local LLM extraction |
| `--rag lexical` | Add passages found by lexical search |
| `--rag semantic` | Add passages found by semantic search |
| `--extra-skill NAME` | Add a skill to assess for this run |
| `--batch-dir FOLDER` | Screen resumes in a folder |
| `--output-dir FOLDER` | Save batch summaries and individual reports |

Options that accept a single value cannot be repeated. For example, repeating `--resume` is rejected instead of silently selecting only the last value.

## Screening Multiple Resumes

Batch mode screens every eligible file in a folder using the same job description and produces an ordered summary for human review.

### Example

```powershell
python -m hr_screening.main --batch-dir "data\incoming\my resumes" --job-file "data\incoming\job.txt"
```

To save the results:

```powershell
python -m hr_screening.main --batch-dir "data\incoming\my resumes" --job-file "data\incoming\job.txt" --output-dir "data\outputs\run1" --json
```

To assess an additional skill:

```powershell
python -m hr_screening.main --batch-dir "data\incoming\my resumes" --job-file "data\incoming\job.txt" --output-dir "data\outputs\run1" --extra-skill Terraform
```

Quote paths containing spaces.

### How batch processing works

- The job description is parsed once per run.
- Resumes are processed in a deterministic filename order.
- A corrupt or unsupported file is reported as a failure without stopping the remaining files.
- Duplicate resumes are flagged but are never silently removed.
- With LLM extraction or semantic retrieval enabled, the relevant client is initialized once and shared across the batch.
- Results are ordered for human review, not to make hiring decisions.

Only files directly inside the selected folder are screened. Subfolders are not searched. Hidden files, Word lock files, `Thumbs.db`, `desktop.ini`, and the job description file itself are excluded where applicable.

### Duplicate detection

Duplicate detection uses a SHA-256 fingerprint of extracted resume text after Unicode NFC normalization and whitespace collapsing.

- Case, punctuation, and numbers are preserved.
- Resumes with matching normalized text are grouped.
- The first filename in deterministic order is the reference copy.
- Other copies are marked with `duplicate_of`.
- All duplicate files remain in the results and are screened.

A resume with even a small textual change may not be identified as a duplicate. This is text-based duplicate detection, not identity matching.

### Batch output files

When `--output-dir` is supplied, the application saves:

| File | Contents |
|---|---|
| `batch_summary.json` | Complete batch summary and reports |
| `batch_summary.txt` | Human-readable batch summary |
| `reports/<resume name>.report.json` | Structured report for an individual resume |
| `reports/<resume name>.report.txt` | Human-readable report for an individual resume |

Existing output files with matching names are replaced. Keep output folders outside version control because individual reports contain resume evidence and may contain personal data.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Successful single run or all batch files screened |
| `1` | Batch partially completed; some files failed |
| `2` | Invalid arguments or another fatal error, including a batch where no file could be screened |

A failed file receives no score or rank. Review every failure manually because an unreadable resume has not been assessed.

## Understanding the Report

### Requirement statuses

| Status | Meaning |
|---|---|
| `supported` | Supporting text was found under the tool's matching rules |
| `partial` | Related evidence was found, but it does not directly establish the requirement |
| `not_found` | No usable supporting evidence was found |
| `unverified` | Evidence is ambiguous, contradictory, or insufficient to confirm the requirement |
| `unassessed` | A job-description requirement could not be interpreted and scored |

**Not found does not mean absent.** A resume may omit a skill, or text extraction and matching rules may fail to identify relevant evidence. Unverified requirements also count as zero in the score, so the score may understate a candidate.

### Coverage score

The score is calculated as:

`100 × sum(weight × credit) / sum(weight)`

Credits:

- Supported: `1.0`
- Partial: `0.5`
- Not found: `0`
- Unverified: `0`

Weights:

| Requirement type | Weight |
|---|---:|
| Required skill | 3 |
| Experience | 3 |
| Education | 2 |
| Certification | 2 |
| Preferred skill | 1 |

Repeated requirements are counted once. A skill listed as both required and preferred retains the required classification.

The score summarizes evidence coverage for the requirements that were assessed. It is not a measure of candidate quality, job performance, or hiring success.

### Complete and incomplete scores

A score is marked incomplete when some job-description requirements could not be assessed.

- An incomplete score excludes unassessed lines from the calculation.
- A score of 100.0 with unassessed requirements does not mean every job requirement was met.
- Unverified and partial items are flagged for manual review.
- Incomplete scores are not automatically demoted in the review order.

Always read the original job description alongside the report to identify requirements the parser may have missed.

### Order for human review

Batch results are ordered by:

1. Higher coverage score.
2. More supported required items, if scores tie.
3. Filename in case-insensitive order, followed by exact filename.

A resume without a score comes last. Preferred skills do not count toward the supported required-item tie-breaker. Retrieved passages do not affect statuses, scores, or ordering.

This ordering is a review aid, not a recommendation or decision about any candidate.

## Requirements and Matching Rules

### Job-description parsing

The parser recognizes common headings such as `Required Skills`, `Required Education`, `Experience`, and `Qualifications`. It skips metadata labels such as `Job Title` and `Location`.

Requirements that cannot be interpreted are reported as unassessed rather than silently treated as satisfied or missing.

The built-in skill vocabulary covers approximately 55 skills and includes aliases. Arbitrary skill names are not automatically recognized.

### Adding extra skills

Use `--extra-skill` to assess a skill that the built-in vocabulary does not recognize:

```powershell
python -m hr_screening.main --resume data\incoming\cv.docx --job-file data\incoming\job.txt --extra-skill Terraform --extra-skill Rust
```

Each added skill is treated as a required skill for that run and receives weight 3. Extra skills are not permanently added to the built-in vocabulary.

Very short skill names may match unrelated text. Review the evidence carefully.

### Skill matching

Skill matching uses whole-word rules, aliases, and basic detection of negated or aspirational mentions.

Examples:

- `Python` can match `Python3`.
- `Java` does not match `JavaScript`.
- A skill mentioned only as something the candidate is learning is not counted as an affirmative mention.
- A recognized denial of a skill is not treated as supporting evidence.
- A skill found only under hobbies or interests is unverified.
- A project-based mention can be supported, but it does not establish professional experience.
- Related technologies may receive partial credit.

Matching rules are intentionally limited to reduce false positives. They can still misinterpret unusual wording or equate technologies that differ in practice.

### Experience and education

- A stated duration can be used when the text clearly identifies the relevant field or role.
- Months are not converted into years.
- Multiple stated year figures are not automatically added together.
- Future work periods are not counted as completed experience.
- An in-progress or expected degree is assessed differently from a completed degree.
- A degree's field of study is not automatically treated as a skill.
- Experience relevance is determined by limited text rules, not by a full understanding of the candidate's work.

For data-analysis experience, the exact title `Data Analyst` is recognized as relevant. The generic word `Analyst` does not automatically establish data-analysis experience. For example, `Business Analyst` and `Financial Analyst` may remain unverified for a data-analysis requirement.

These rules help keep the assessment traceable, but a human reviewer must resolve ambiguous cases.

## Optional LLM Extraction

The default rule-based mode does not require a model. Optional LLM extraction uses a local Ollama server and an installed chat model.

### Configure Ollama

Install Ollama from the official site:

https://ollama.com/download

Download a model and check the installed model names:

```powershell
ollama pull qwen2.5:3b
ollama list
```

Create `.env` from `.env.example` and configure the installed model:

```text
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:3b
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
```

Use the exact model names shown by `ollama list`. These are example settings; the model names are not built into the application.

Run extraction with:

```powershell
python -m hr_screening.main --resume data\incoming\sample_resume.docx --job-file data\incoming\job.txt --llm
```

### Fallback behavior

If the model server is unreachable, times out, or returns an unusable response, the application can fall back to rule-based extraction and explain the fallback in the report.

Invalid model configuration stops the run with an error rather than silently proceeding.

Model-proposed evidence is validated against the resume text. This confirms that the text exists; it does not guarantee that the interpretation is correct.

## Retrieved Passages

Retrieval is optional and is disabled by default. It adds passages related to each requirement so that reviewers can inspect the surrounding resume text.

### Lexical retrieval

```powershell
python -m hr_screening.main --resume data\incoming\sample_resume.docx --job-file data\incoming\job.txt --rag lexical
```

Lexical retrieval uses keyword-based BM25 search. It is deterministic and requires no model or network access.

### Semantic retrieval

Semantic retrieval uses embeddings from an Ollama embedding model:

```powershell
python -m hr_screening.main --resume data\incoming\sample_resume.docx --job-file data\incoming\job.txt --rag semantic
```

Ensure the embedding model is installed and `EMBEDDING_MODEL` is configured in `.env`.

Semantic retrieval can find related wording that keyword search misses, but it can also return unrelated passages. A nearest passage is not proof that a requirement is met.

### Retrieval guarantees and limitations

- Passages are exact slices of the extracted resume text.
- Passage text is checked against the original extracted text.
- At most three passages are returned for each requirement.
- Experience requirements are not searched with a generic keyword query because that can return misleading text.
- Retrieval does not change requirement statuses, evidence assessments, coverage scores, or batch ordering.

### Synthetic retrieval evaluation

The synthetic retrieval evaluation checks answerable queries, absent skills, and negated or learning mentions.

The recorded lexical baseline found a correct passage at rank 1 for 28 of 31 answerable queries, returned no passages for all 9 absent skills, and correctly labelled all 4 negated or learning mentions. Three paraphrases remain known lexical-search limitations.

These results are a small sanity check, not a benchmark of real-world retrieval quality.

Run the evaluation with:

```powershell
python evals\run_retrieval_eval.py
```

## Testing and Verification

Run the automated checks from the project root:

```powershell
python -m pytest -q
ruff check .
python evals\run_retrieval_eval.py
python -m hr_screening.main --help
```

The automated suite uses fake model clients and embedders for many integration tests. It checks application behavior without requiring a running Ollama server.

### Recorded verification results

| Check | Recorded result | Notes |
|---|---|---|
| Full pytest suite | 1,440 passed, 1 skipped | Linux completion run, 2026-10-21 |
| Earlier Windows pytest run | 1,388 passed, 2 skipped | Earlier baseline; rerun after final changes to confirm the current Windows result |
| Ruff | All checks passed | Linux completion run |
| Lexical retrieval evaluation | 28/31 answerable queries at rank 1; 9/9 absent skills returned no passages; 4/4 mention labels correct | Synthetic evaluation |
| Final synthetic Windows batch | 2 screened, 0 failed | Output saved to `test_data\output_final` |
| Live Ollama extraction | Not confirmed by the results recorded here | Requires a running local server and installed chat model |
| Live semantic retrieval | Not confirmed by the results recorded here | Requires a running local server and installed embedding model |

The Linux and Windows test counts refer to separate runs and must not be combined. Rerun the tests on the final Windows working copy before recording its final verification result.

Automated tests do not prove that real resumes will be assessed accurately. Live Ollama integration, scanned PDFs, complex document layouts, and real-world screening quality require separate checks.

## Privacy and Data Handling

Resumes contain personal information. Use synthetic resumes for development and testing.

- Do not commit real resumes or generated reports containing personal information.
- Keep `.env` out of version control.
- The default rule-based and lexical modes do not send resume text to an external service.
- `--llm` sends resume text to the configured Ollama server.
- `--rag semantic` sends resume text chunks and requirement query text to the configured Ollama server.
- Confirm that Ollama is running locally and that `OLLAMA_BASE_URL` points to a trusted endpoint before using candidate data.
- Reports contain quoted resume evidence and must be treated as sensitive.
- Logs are designed to avoid recording resume text and filenames.

Generated output folders should remain outside the repository or be excluded through `.gitignore`.

## Limitations

This project is an explainable screening aid, not an autonomous hiring system.

Known limitations include:

- No OCR for scanned PDFs.
- Multi-column PDFs may produce incorrectly ordered text.
- Rule-based parsing depends on recognized headings and wording.
- The built-in skill vocabulary is limited; unfamiliar skills may need `--extra-skill`.
- Skill mentions do not establish proficiency, depth, recency, or professional experience.
- Experience-field matching uses limited text rules and may not understand the actual responsibilities of a role.
- Degree and certification wording can be ambiguous.
- Lexical retrieval can miss paraphrases.
- Semantic retrieval can return unrelated passages and has not been comprehensively benchmarked.
- No fairness, bias, legal compliance, or hiring-outcome guarantees have been established.
- No comprehensive accuracy evaluation on real candidate resumes has been performed.

Review the original job description, each report's evidence, all unverified or unassessed requirements, and every file failure before drawing conclusions.

## Project Structure

```text
ai-hr-resume-screening/
├── src/
│   └── hr_screening/
│       ├── extraction/       # Job and resume parsing
│       ├── evaluation/       # Requirement assessment and scoring
│       ├── ingestion/        # Resume file reading and validation
│       ├── llm/              # Optional local model integration
│       ├── rag/              # Lexical and semantic retrieval
│       └── main.py           # Command-line entry point
├── tests/                    # Automated tests
├── evals/                    # Synthetic retrieval evaluation
├── docs/                     # Documentation and sample generation
├── test_data/                # Synthetic test inputs
├── pyproject.toml
├── README.md
└── .gitignore
```

## Project Goal

The goal is to reduce repetitive resume-review work by organizing job requirements, extracting traceable evidence, identifying files that need attention, and producing consistent reports.

The final responsibility remains with the human reviewer.
