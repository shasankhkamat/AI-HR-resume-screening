﻿# AI HR Resume Screening

An explainable screening tool. It takes one PDF or DOCX resume and one job description and
prints a report: each requirement is marked supported, partial, not found or unverified,
with the exact resume line that supports it. It supports human review. It does not make
hiring decisions.

**Responsible use.** This is decision support, not an autonomous hiring system. The score counts
how many listed requirements have supporting text. It does not measure a person. Every report
says that a human must decide, and a reviewer should read the evidence lines and the
"not found or unverified" list before relying on any result. No fairness, bias or compliance
testing has been done. A resume is personal data: test with synthetic resumes only, and keep
real ones out of the repository.

## Setup

Python 3.11 or newer.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

This installs the package with its three dependencies (pydantic, pydantic-settings and pypdf),
the dev tools (pytest and ruff) and a `hr-screening` command. The examples below use
`python -m hr_screening.main`, which does the same thing. If PowerShell refuses to run
`Activate.ps1`, allow scripts for this window only with
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`, then activate again.

What each mode needs:

| Mode | Needs |
|---|---|
| Default run and `--rag lexical` | Python 3.11 or newer and the three packages above. No model, no network |
| `--llm` | A local Ollama server and a chat model. The project owner's setup is `qwen2.5:3b` |
| `--rag semantic` | A local Ollama server and an embedding model. The project owner's setup is `nomic-embed-text` |

Those two model names are one working setup, not built in. Any installed model can be used by
changing `.env`. To download them: `ollama pull qwen2.5:3b` and `ollama pull nomic-embed-text`,
then check the exact names with `ollama list`.

## Status

Implemented and tested:

- Resume ingestion for text based PDF and DOCX files (size, type and page checks)
- Rule based parsing of the resume and the job description
- Pydantic models for candidate, job requirements and report
- Keyword skill matching with aliases, version suffixes, negation and learning detection
- Evidence validation: every evidence snippet is exact resume text and is checked against it
- Weighted requirement coverage score
- Optional extraction with a local Ollama model (opt in with `--llm`), with verification of
  the model output and a rule based fallback
- Optional list of resume passages for each requirement, for a reviewer to read: keyword
  search (`--rag lexical`) or embedding search with a local Ollama model (`--rag semantic`)
- A small synthetic check of the retrieval (`evals/`)
- Batch screening of a folder of resumes (`--batch-dir`): every file is screened with the same
  pipeline, duplicates are flagged, failures are listed, and the summary is ordered for human
  review
- A pipeline, a command line tool, pytest tests and Ruff

Not implemented (future work):

- Hosted LLM providers. Only a local Ollama server is supported.
- A vector database. Embeddings are computed for each run and kept in memory.
- Searching a large pool of resumes, a database of candidates, or any web interface. A batch
  reads one folder per run.
- OCR for scanned PDFs
- Any fairness, bias or compliance guarantee. No such testing has been done.

## Architecture

```text
resume file -> ingestion (validate, extract text)
job text    -> extraction.job_parser -> JobRequirements
resume text -> extraction.resume_parser (rules)  or  llm.extractor (optional, verified)
                -> CandidateProfile
evaluation.evaluator -> ScreeningReport (status + evidence + score + limitations)

with --rag, after the report is built:
resume text -> rag.chunking -> chunks (exact slices of the text, split at section headings)
            -> rag.retrieval (keyword, BM25)   or   rag.semantic (embeddings, cosine)
            -> report.retrieval: passages for each requirement
```

With `--llm`, `llm.factory` builds an `OllamaClient` from the settings. The client implements
the small `LLMClient` interface (`complete(system_prompt, user_prompt) -> str`) and only
returns the model's text. Parsing, validation and the evidence checks stay in `llm.extractor`.

With `--rag semantic`, `rag.factory` builds an `OllamaEmbedder` that implements the small
`Embedder` interface (`embed(texts) -> vectors`). It shares its HTTP code and retry rules with
the chat client. Retrieval runs after the report is built and only adds the `retrieval` field
and one line in the limitations. It cannot change a status or the score.

## Run it

Rule based mode (the default). It makes no network request and needs no model.

```powershell
python docs\make_sample_resume.py
python -m hr_screening.main --resume data\incoming\sample_resume.docx --job-text "Requirements`n- 3+ years of experience`n- Python and SQL`n- Bachelor's degree"
```

LLM mode. It needs Ollama and a downloaded model (see the section on local LLM mode).

```powershell
python -m hr_screening.main --resume data\incoming\sample_resume.docx --job-text "Requirements`n- 3+ years of experience`n- Python and SQL`n- Bachelor's degree" --llm
```

With retrieved passages. `lexical` needs nothing else. `semantic` needs an embedding model
(see the next section).

```powershell
python -m hr_screening.main --resume data\incoming\sample_resume.docx --job-text "Requirements`n- 3+ years of experience`n- Python and SQL`n- Bachelor's degree" --rag lexical
```

With a job description file and JSON output. The file can be written from PowerShell like this:

```powershell
@"
Requirements
- 3+ years of experience
- Python and SQL
- Bachelor's degree
"@ | Set-Content -Encoding utf8 data\incoming\job.txt
python -m hr_screening.main --resume data\incoming\sample_resume.docx --job-file data\incoming\job.txt --json
```

Options: `--job-file path.txt`, `--json`, `--resume-dir folder`, `--llm`,
`--rag lexical|semantic`, `--extra-skill NAME`, `--batch-dir folder`, `--output-dir folder`.
Every option except `--extra-skill` can be given once. `--resume` screens one resume. To screen
a folder, see "Screening a folder of resumes". `--extra-skill` can be repeated (see
"Requirements the tool cannot assess"). A
repeated option, such as two `--resume` flags, is refused with exit code 2 instead of silently
screening only the last file. The sample resume is synthetic. Never commit real resumes.

## Screening a folder of resumes (batch)

One job description and one folder of resumes. Every file is screened, and the result is a
summary ordered for human review, plus one report per resume.

```powershell
python -m hr_screening.main --batch-dir "data\incoming\my resumes" --job-file "data\incoming\job.txt"
python -m hr_screening.main --batch-dir "data\incoming\my resumes" --job-file "data\incoming\job.txt" --output-dir "data\outputs\run1" --extra-skill Terraform
python -m hr_screening.main --batch-dir "data\incoming\my resumes" --job-file "data\incoming\job.txt" --json
```

Quote a path that contains spaces. `--batch-dir` replaces `--resume`, and the two cannot be used
together. `--resume-dir` belongs to `--resume` and is refused with `--batch-dir`. `--llm`,
`--rag` and `--extra-skill` work as they do for one resume. Nothing asks a question during a
batch.

**What is screened.** The files directly inside the folder, in file name order (case
insensitive, then the exact name). Sub folders are not searched. Hidden files, Word lock files
(`~$name.docx`), `Thumbs.db` and `desktop.ini` are left out and listed under LEFT OUT. If the
job description file is inside the folder, it is left out and listed too. Any other file is
screened, so an unsupported file such as `notes.txt` appears as a failure and does not vanish.
The job description is parsed once. With `--llm` or `--rag semantic`, the model client or the
embedder is built once and shared by every resume. Without them, none is built. Each resume is
screened by the same code as a single run, so its report is the same as `--resume` gives.

**Failures.** A corrupt, empty, scanned, password protected, oversized or unsupported file is
listed under FAILED FILES with the reason. It gets no score and no rank. The batch continues. A
model server that is down is not a failure: each resume falls back to the rule based parser and
says so in its limitations, as in a single run. An invalid model setting stops the whole batch
with exit code 2, because every file would fail the same way.

**Exit codes.**

| Code | Meaning |
|---|---|
| 0 | Single run: success. Batch: every file was screened |
| 1 | Batch only: some files were screened and some failed. The summary is still printed and saved |
| 2 | An error. Single run: any failure. Batch: bad arguments, a missing or empty folder, a bad job description or model setting, a failed output write, or no file could be screened |

**Duplicates.** Each resume's text is normalised and hashed (SHA 256, the first 16 characters
are shown). Normalisation is Unicode NFC and then every run of whitespace becomes one space.
Letter case, punctuation and numbers are kept, so two different resumes are not merged by
accident, and a resume that differs by one word or date is not a duplicate. Resumes with the
same fingerprint form a group. The first file name in the order above is the reference, and the
others are flagged `duplicate_of` that name. Every duplicate is still screened and listed. A
failed file has no text, so it is never in a group.

**Order for human review.**

1. Higher coverage score first.
2. If scores tie, more supported required items first. Required items are required skills,
   experience, education and certifications. Preferred skills are not counted.
3. If that ties, the file name in case insensitive order, then the exact file name.

A resume with no score (the job had nothing to score) comes last. The order depends only on the
reports. It does not depend on the order the files were found, and `--rag` cannot change it. The
list is an order for review. It does not accept, reject or recommend anyone.

**Incomplete scores.** A score is marked INCOMPLETE when parts of the job description were not
assessed (see "Requirements the tool cannot assess"). The score leaves those parts out, so an
incomplete 100.0 is not the same as a complete 100.0. The order does not push incomplete scores
down, because there is no fair way to price a requirement nobody assessed. Instead, every
incomplete score is flagged next to the number, counted in the summary, and listed under NEEDS
HUMAN REVIEW with the unassessed text. The same section lists every `unverified` and `partial`
requirement. Unverified requirements count as zero, so a score can understate a candidate, and
the unverified count is shown beside every score.

**Where the output goes.** Without `--output-dir`, the summary is printed (text, or JSON with
`--json`) and nothing is written to disk. With `--output-dir folder`, the same summary is
printed and these files are saved. Outputs from an earlier run with the same names are
replaced.

| File | Content |
|---|---|
| `batch_summary.json` | The whole summary as JSON, including every report |
| `batch_summary.txt` | The text summary. It holds no resume text |
| `reports\<resume name>.report.json` | The source file name, rank, counts, review items and duplicate flags, and under `report` the usual report |
| `reports\<resume name>.report.txt` | The usual text report, with a `SOURCE FILE` line on top |

`--output-dir` must be a different folder from `--batch-dir`. A folder inside the resume folder
is fine, because sub folders are not searched. Output names end in `.json` or `.txt`, so a PDF
or DOCX resume cannot be overwritten, and the resumes are only read.

**The summary shows**: files found, screened, failed and left out; complete and incomplete
scores; duplicate files and groups; the ordered list with, for each resume, the score, the
completeness flag, the supported required items, and the counts of supported, partial, not found,
unverified and not assessed requirements; the review items; duplicate groups; failed files with
the reason; the extra skills used; notes from the job parser; the limitations; and a human
review notice. JSON keys are fixed and their order is stable. `counts.requirements` counts every
assessed requirement, preferred skills included.

Example (shortened, synthetic files):

```text
BATCH SCREENING SUMMARY
Decision support only. This list is ordered for human review. It does not accept, reject or recommend any candidate, and a person must make every hiring decision.

FILES
Found: 5   Screened: 3   Failed: 2   Left out: 1
Complete scores: 0   INCOMPLETE scores: 3
Duplicate files: 1 in 1 group(s)

ORDERED FOR HUMAN REVIEW
1. Bo.docx
    score 100.0 (INCOMPLETE SCORE); required items supported 4 of 4
    supported 4, partial 0, not found 0, UNVERIFIED 0, NOT ASSESSED 1
2. anna copy.docx
    score 72.7 (INCOMPLETE SCORE); required items supported 3 of 4
    reference copy for duplicate group 1
3. Anna one.docx
    score 72.7 (INCOMPLETE SCORE); required items supported 3 of 4
    DUPLICATE of anna copy.docx (group 1)

FAILED FILES (not screened, no score)
- broken.docx: The file content does not look like a valid .docx file.
- notes.txt: Unsupported file type '.txt'. Supported types: .pdf, .docx.
```

How to read it: `Bo.docx` has the top score, but the score is INCOMPLETE (one job line was not
assessed) and the line is listed for review, so it is not a finished comparison. The two Anna
files have the same text, so they have the same score. Both are kept. Read each failed file
yourself, because a failed resume has not been assessed at all.

Privacy in a batch: resumes are read and never copied or changed. Logs hold counts and error
class names only, never a file name or resume text. The text summary and the error messages hold
no resume text. The JSON files include each report, and a report quotes resume lines as
evidence, so treat them as personal data. File names are shown as the file system gives them,
with control characters replaced, and a file name can itself contain a person's name. Nothing is
sent anywhere unless you use `--llm` or `--rag semantic`, which send resume text to the
configured Ollama server for every resume.

## Requirements the tool cannot assess

The job parser reads skills only from a built in list of about 55 names. It does not treat
arbitrary words as skills, because words such as Go, R or Excel would match ordinary text.
When a line under a required heading gives no skill from the list, no years of experience, no
degree and no certification, nothing on it is scored. The report says so in three places:

- Text report: a `SCORE INCOMPLETE` line under the score, and a `NOT ASSESSED` section that
  lists each line.
- JSON: `"score_complete": false` and an `unassessed_requirements` list. Each entry has `text`
  and `reason`. Both keys are always present. `score_complete` is `true` and the list is empty
  when every line was assessed.
- `score_explanation` and the limitations also say the score is incomplete.

**Unassessed is not the same as not found.** A requirement that was assessed and not found has a
status and counts as zero in the score. An unassessed line has no status and no weight, so the
score does not include it. A score of 100 with a `NOT ASSESSED` section means every assessed
requirement was met, not that the candidate meets the job.

To assess such a skill, name it for this run:

```powershell
python -m hr_screening.main --resume data\incoming\cv.docx --job-file data\incoming\job.txt --extra-skill Terraform --extra-skill Rust
```

An extra skill is a required skill with weight 3. It is matched as a whole word in the resume
text with the same negation and learning rules as any skill, and it is listed in the limitations
so the report shows who added it. It is not added to the built in list, and it applies to this
run only. A skill the job lists as preferred becomes required if you name it. A job line that
names an extra skill is no longer listed as unassessed. Very short names (one or two characters)
can match unrelated text, and the report warns about them. A bad value (empty, longer than 60
characters, more than 20 names) stops the run with exit code 2.

A line where something was recognised is checked too. The tool removes the skills, the years
and the degree it understood, and looks at what is left. A leftover piece that looks like the
name of a skill is listed on its own, so "Python and Terraform" lists `Terraform` and
"Rust and Go, Bachelor's degree" lists `Rust` and `Go`. A piece looks like a name when it is
capitalised (`Terraform`), upper case (`CI`) or has a digit, a plus or a hash (`Helm3`).

Limits. A skill written in plain lower case ("python and terraform") cannot be told from an
ordinary word and is not listed. A capitalised word that is not a skill, such as a second field
of study, is listed anyway. Lines under a heading the parser does not recognise, or a job with
no headings, can list many lines that are not requirements (a title, a sentence about the
company), because the tool cannot tell them apart. Always read the original job description, and
add a skill with `--extra-skill` whenever you see one the report does not list.

## Headings and labelled sections

Job descriptions: a line such as `Required Skills:`, `Required Education:`, `Experience:` or
`Qualifications:` is a heading, not a requirement. Text on the same line (`Required Education:
Bachelor's degree in Statistics.`) is kept as the requirement. Labels such as `Job Title:` or
`Location:` are metadata and are skipped. A colon at the end of any other line does not make it a
heading, and a line that cannot be classified is listed under NOT ASSESSED.

A field of study after a degree ("Bachelor's degree in Computer Science, Statistics") is part of
the education requirement. It is not read as a skill.

Resumes: `Skills:`, `Experience:`, `Education:` and `Certifications:` are recognised when the text
follows on the same line. A single line in the experience section that starts with a years figure
(`2 years as a Data Analyst`) is read as stated years. With two or more such lines nothing is
guessed. Months are never converted to years. A skill word that appears only in the education
section is `unverified`, not supported, because it may be a field of study.

**Experience in a named field.** A job line such as "1 year of data analysis experience" is an
experience requirement. Its field is not also listed as a required skill. Under a `Required
Skills` heading, `Data Analysis` is still a skill. Tools are not affected: "experience with
Python" still lists Python.

A resume meets a field requirement when the stated or dated experience text names the field's
words, or, for data analysis only, the exact title "Data Analyst" (`FIELD_TITLES` in
`extraction/experience_field.py`). The bare word "analyst" never counts, so "Business Analyst"
and "Financial Analyst" stay unverified. Durations are unchanged: "2 years as a Data Analyst"
meets a one year minimum, "6 months" does not, and months are never converted into years (the
report says the experience section states months). Two or more stated year figures are not
added up or guessed, and the result stays unverified.

## Retrieved passages (`--rag`)

The report tells you whether a requirement has supporting text. A reviewer still has to read
the resume to judge it, so `--rag` adds a section that lists the passages most related to each
requirement, with their position in the resume. It is off by default and it changes nothing
else: statuses, evidence and the score are the same with or without it, and tests compare them.

Output of the `--rag lexical` command above (shortened):

```text
RETRIEVED PASSAGES (for review)
Found by keyword search. A passage is not evidence that a requirement is met, and it does not change any status or the score. Read each one yourself.
[required skill] python
    Passage 1 (skills; chars 0-50; relevance 1.51; skill mentioned as: affirmative)
        | SKILLS
        | Python, SQL, Pandas, scikit-learn, Power BI
[experience] 3+ years of experience
    Not searched. The dated roles above are the evidence for years of experience, and a keyword search would match any sentence that contains the word experience.
[education] Bachelor level education or higher
    Passage 1 (education; chars 205-266; relevance 3.18)
        | EDUCATION
        | B.Tech in Computer Science, Example Institute, 2018
```

The sample resume is only 359 characters, so each search has one or two passages to choose
from. The section is more useful on a resume of normal length.

### What a passage is

- **An exact slice of the resume.** `text == resume[start:end]`, with the offsets shown. Before
  the report is returned, every passage is checked against the resume text. If a check fails,
  the passages are dropped and the report says so.
- **Not evidence.** A passage is returned because it relates to the requirement, so it can say
  "no hands on experience with Kubernetes" or "currently learning Spark". For skills the report
  labels how the skill is mentioned (affirmative, negated or aspirational) with the same rules
  the evaluator uses. The label does not change the status.
- **Ranked by a score that depends on the mode.** Keyword search shows it as `relevance`
  (BM25). Semantic search shows it as `similarity` (cosine). The two cannot be compared.
- **Not a verdict when missing.** "No matching passage found" does not show that the
  candidate lacks the skill.

At most three passages are listed for each requirement. How each kind is searched:

| Requirement | Query |
|---|---|
| Required or preferred skill | The skill and its aliases. Keyword search also uses the related skills from the built in list, at lower weight |
| Certification | The words of the requirement |
| Education | A fixed set of degree words plus the requested level |
| Years of experience | Not searched. A keyword query for "experience" would match any sentence with that word, including negated ones, and the evaluator already cites the dated roles |

With `--json`, the report gets a `retrieval` list. Each entry has the requirement, its passages
and `method` (`lexical` or `semantic`). Without retrieval the key is absent, so the default
JSON is the same as before.

### Why there are two modes

Keyword search is the dependable default. It is deterministic, it needs no model and no
network, and it can say why a passage was chosen (the matched words are in the JSON). Its
weakness is that it only sees words. In the synthetic check it found nothing for `ci/cd` on a
resume that says "automated build and release pipelines".

Semantic search embeds the passages and the requirement and ranks by cosine similarity, so it
can connect different words. That is the reason it is there. The costs: it needs an Ollama
embedding model, it sends the resume text and the requirement queries to the Ollama server,
and it always has a nearest passage, so it can show text that is unrelated and give low scores
to a skill that is not in the resume. Whether it ranks better than keyword search on real
resumes has not been measured. Keyword search stays the default and the baseline to compare
against.

### Keyword search (`--rag lexical`)

A small BM25 index over the chunks, written with the standard library only. Words are lower
cased and plural s is dropped. A version number after a word is dropped too, so `Python3`
matches `python`, as it does in the evaluator. Skill queries use the same alias and related
skill lists as the evaluator. It makes no network request and reads no settings.

### Semantic search (`--rag semantic`)

The chunks of the resume are embedded in batches of 8 with Ollama's `/api/embed`, and each
requirement is embedded with one more call. Passages are ranked by cosine similarity and ties
keep the earlier passage first. Only passages with a similarity above 0 are listed. Vectors are
kept in memory for one run and are never written to disk. A resume with no text sends nothing.

### Set up semantic search

1. Install Ollama as described in the next section.
2. Download an embedding model. No model name is built into this project. Pick one from
   https://ollama.com/library, then look at the exact name:

```powershell
   ollama pull <embedding-model>
   ollama list
```

3. Set `EMBEDDING_MODEL=<embedding-model>` in `.env`. The server address, timeout and retries
   are the same settings as for `--llm` (`OLLAMA_BASE_URL`, `LLM_TIMEOUT_SECONDS`,
   `LLM_MAX_RETRIES`). No chat model and no `LLM_PROVIDER` are needed for embeddings.
   Models that expect a task label in front of each text, such as `nomic-embed-text`, can use
   the optional `EMBEDDING_QUERY_PREFIX` and `EMBEDDING_DOCUMENT_PREFIX` settings. They are
   empty by default, and whether they help has not been measured.
4. Run with `--rag semantic`. Names ending in `-cloud` or `:cloud` are refused, as for chat
   models.

| Situation | Result |
|---|---|
| `EMBEDDING_MODEL` missing, a cloud model name, a bad URL or an invalid setting | Message on screen and exit code 2. No report and no silent fallback |
| Ollama not running, timeout, model not installed, server error, unusable vectors | Keyword search is used instead. The report says why, for example "Semantic search failed (the model server could not be reached), so keyword search was used instead." |
| Keyword search also fails | The screening result is returned without passages, with a note |

## Retrieval check on synthetic cases

`evals/run_retrieval_eval.py` runs 40 queries over four invented resumes and prints what was
found. Keyword search found a correct passage at rank 1 for 28 of 31 answerable queries,
returned nothing for all 9 absent skills, and labelled all 4 negated and learning mentions
correctly. It missed three paraphrases (`a/b testing`, `ci/cd`, `monitoring`), which are
limits of keyword search, not bugs. A fourth, `forecasting`, was a word form problem and is
fixed. The cases are tiny and were written by the same
assistant that wrote the code, so this is a sanity check and not a benchmark. The project
owner ran the first 22 queries with the Ollama model `nomic-embed-text` on Windows: semantic
search found all 3 paraphrases in the top 3, but returned passages for all 3 absent skills,
because the minimum similarity is 0 and the score ranges overlap. The report labels such
passages as related text. `docs/retrieval_evaluation.md` has the full output, what the check
found (one bug, now fixed) and how to run it for semantic search, including `--scores`.

```powershell
python evals\run_retrieval_eval.py
python evals\run_retrieval_eval.py --semantic
```

## Local LLM mode with Ollama

### What is sent

With `--llm`, the resume text and a fixed set of extraction instructions are sent to the
Ollama server at `OLLAMA_BASE_URL`. The job description is not sent. Nothing is sent to Ollama
unless you use `--llm` or `--rag semantic` (see the section on retrieved passages for what that
sends).

### 1. Install Ollama

Install Ollama from the official site, https://ollama.com/download. On Windows the installer
runs Ollama in the background and adds the `ollama` command to your terminal. Check it:

```powershell
ollama --version
```

The Windows guide is at https://docs.ollama.com/windows.

### 2. Download a model

No model name is built into this project. Choose a small model that fits your memory from
https://ollama.com/library, download it, and look at the exact name that Ollama shows:

```powershell
ollama pull <model-name>
ollama list
```

Model downloads are large (often several GB) and need disk space. Do not choose a name that
ends in `-cloud`. Cloud models run on Ollama's remote servers, so resume text would leave your
computer. The project refuses names ending in `-cloud` or `:cloud` as a safety check. That
check is best effort. Confirm with `ollama list` that your model is downloaded to this machine.

### 3. Configure

Copy `.env.example` to `.env` in the project folder and set at least these two lines. Use the
exact name from `ollama list`. `.env` is git ignored and holds no secret for this provider. The
project owner's setup is:

```text
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:3b
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
```

A setting in the environment overrides `.env` for that PowerShell window only, for example
`$env:LLM_TIMEOUT_SECONDS = "120"`, so `.env` does not need to change for a slow first call.

| Setting | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | none | Must be `ollama` |
| `LLM_MODEL` | none | Name of an installed model. No default is provided |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Address of the Ollama server |
| `LLM_TIMEOUT_SECONDS` | 30 | Wait per attempt. The first call loads the model, so use 120 or more on slow computers |
| `LLM_MAX_RETRIES` | 2 | Extra attempts for timeouts, lost connections and server errors. 0 to 5 |
| `LLM_MAX_OUTPUT_TOKENS` | 2048 | Limit on the reply length. The JSON for one resume is short. Models that think first may need more |

No API key is needed. `LLM_API_KEY` is not used by this provider. The longest a run can wait
is about (retries + 1) x timeout.

### 4. What happens when something goes wrong

| Situation | Result |
|---|---|
| `LLM_PROVIDER` or `LLM_MODEL` missing, unsupported provider, bad URL, a setting out of range, or a cloud model name | Message on screen and exit code 2. No report and no silent fallback |
| Ollama not running, timeout, model not installed, server error, empty reply, reply cut off at the token limit, invalid JSON | The rule based parser is used. The report states the reason, for example "LLM extraction failed (the model server could not be reached), so the rule based parser was used instead." |
| Model proposes items that are not in the resume, or skills the resume denies | Those items are discarded and the report says how many |

Transient failures are retried a limited number of times with a short pause. A server that is
not running, a missing model and bad requests are not retried.

### 5. Manual smoke test with a synthetic resume

Use only the synthetic sample resume. Never use a real candidate resume.

1. Create it: `python docs\make_sample_resume.py`
2. Run rule based mode and note the score and the statuses.
3. Make sure Ollama is running and `ollama list` shows your model.
4. Run the same command with `--llm`.
5. Look for the line "A language model performed the extraction", or for a fallback reason.
6. Compare the two reports. The score and some statuses can differ. Decide for yourself
   whether each difference is right by reading the resume lines shown as evidence.

### Privacy

- The default address is on this computer. The privacy benefit holds only if the server is
  trusted and `OLLAMA_BASE_URL` still points at it. Do not point it at a remote service. The
  command line prints a warning when the address is not localhost, 127.0.0.1 or ::1.
- Requests ignore proxy settings and never follow redirects, so resume text goes only to the
  configured address. Only `http` and `https` addresses are accepted.
- Prompts, resume text, model replies and response bodies are never logged or placed in error
  messages. Error messages are fixed text.
- `--rag semantic` follows the same rules and sends two things to the same server: chunks of
  the resume text, and short query texts built from the requirements (for example `python`,
  `scikit-learn, sklearn, scikit learn` or `bachelor degree education university college
  institute`).
- The report itself contains resume lines as evidence. Treat the output as sensitive.
- Local inference has no per request API charge. It does need local compute, disk space and
  model downloads, and it can be slow without a GPU.

### Accuracy

A model that runs and passes the tests is not a model that is accurate. The automated tests
use fake servers and prove only that the plumbing and the error handling work. No test in this
project calls a real Ollama server, so a working `--llm` or `--rag semantic` run on your
machine depends on your Ollama install. Check it with the smoke test above. The evidence
check proves that extracted text exists in the resume. It does not prove the text means what
the model says. Treat every LLM result as a suggestion for a human reviewer.

## Tests

```powershell
pytest
ruff check .
hr-screening --help
python evals\run_retrieval_eval.py
```

No test needs Ollama, a downloaded model, an API key or the internet. The tests use fake
transports and a fake embedder. A few tests start a tiny HTTP server on 127.0.0.1 inside the
test process to exercise the real HTTP code, and a few tests start Python subprocesses (to
check that results do not depend on hash randomization, and to run the real command line on
generated files). Nothing leaves your computer.

`tests/test_phase2_e2e.py` is the end to end validation. Each test generates a synthetic DOCX
or PDF and runs it through the real pipeline or the command line: strong, partial and missing
matches, negated and learning skills, hobby only mentions, unfinished degrees, certification
wording, undated roles, job description parsing, bad files and bad arguments, model output and
fallback, and retrieval in every mode. Expected scores are worked out by hand from the weights.
Tested only with fakes: the model (`FakeLLMClient`) and the embedder (`FakeEmbedder`). Not tested
by any automated test: a real Ollama server, a real embedding model, scanned or multi column
PDFs, and real resumes. Those need a manual check (see "Local LLM mode with Ollama").

What the tests prove: the plumbing, the validation of model output and of vectors, the
fallbacks and error messages, that passages are exact slices of the resume, and that
`--rag` never changes a status or the score. What they do not prove: that Ollama's chat or
`/api/embed` endpoint works on your machine, or that semantic search ranks passages well. The
fake embedder hashes word counts. It is not a language model and its results mean nothing about
real embeddings. `tests/test_retrieval_eval.py` also pins the numbers reported in
`docs/retrieval_evaluation.md`, so the document and the code cannot drift apart unnoticed.
`tests/test_phase3_cli.py` checks that a repeated option such as two `--resume` flags is refused,
`tests/test_negative_evidence.py` covers how a denied skill is reported, `tests/test_today_is_used.py`
checks that the supplied date reaches every date calculation, and
`tests/test_unassessed_requirements.py` covers unassessed lines, `--extra-skill` and the
incomplete score flag.

## Verification status

Automated checks use fake models and a fake embedder. Live checks need a running Ollama server.
They are kept apart here on purpose.

| Check | Result | Where and how |
|---|---|---|
| `pytest` | 1440 passed, 1 skipped | Linux, 2026-10-21, fakes only |
| `ruff check .` | All checks passed | Linux, 2026-10-21 |
| `python evals/run_retrieval_eval.py` | 28 of 31 answerable queries at rank 1, 9 of 9 absent skills empty, 4 of 4 mention labels | Linux, deterministic |
| Windows `pytest` and Ruff | Reported passing by the project owner. Record your own numbers with the commands above | Windows, owner |
| Command line on four synthetic resumes (strong, partial, poor, ambiguous), DOCX and PDF | 57 of 57 checks passed: score equals the formula, evidence is in the resume, `--rag` changes nothing | Linux, real command line |
| `--llm` and `--rag semantic` with the Ollama server unreachable | Exit code 0, rule based parser and keyword search used, and the report says why | Linux, real command line, no Ollama |
| `--llm` with `qwen2.5:3b` on a running server | **Not run** | Needs Ollama |
| `--rag semantic` with `nomic-embed-text` on a running server | **Not run** for the command line. The first 22 evaluation queries were run once by the project owner (see `docs/retrieval_evaluation.md`) | Needs Ollama |
| Batch command line on synthetic DOCX files (valid, duplicate, corrupt, unsupported, temporary) | Ran once by hand: summary, exit code 1 and the saved files were as described above | Linux, real command line |
| Batch on Windows, with paths that contain spaces | Tests use a folder and files with spaces and pass on Linux. **Not run on Windows** by the assistant | Needs Windows |
| `--batch-dir` with `--llm` or `--rag semantic` on a running server | **Not run.** Tests use fakes and check that the client and embedder are built once | Needs Ollama |
| Real or scanned PDFs, multi column layouts, real resumes | **Not tested** | |

## How skill matching works

Skills come from a built in list of about 55 skills, each with a few aliases (for example
`js` for JavaScript, `sklearn` for scikit-learn). A skill matches as a whole word, ignoring
case. This is why `Java` does not match `JavaScript` and `SQL` does not match `PostgreSQL`
or `NoSQL`. A version number may follow directly, so `Python3` and `C++17` match.

Each mention is then labelled:

- **Affirmative**: counts as a mention of the skill.
- **Negated**: "no experience with Python", "never used Python", "Python: none". Not counted.
- **Aspirational**: "currently learning Python", "plan to learn Python". Not counted.

If a skill has both an affirmative and a negated mention, the result is `unverified` and
both lines are shown, because the resume contradicts itself.

A skill found only under a PROJECTS heading is `supported`, but the report labels it as
project based evidence. It is a mention. It is not proof of professional experience.

A skill found only under an INTERESTS, HOBBIES or REFERENCES heading is `unverified`. A hobby is
not skill evidence, so a reviewer must read the line. If a skill is listed and also described as
being learned elsewhere ("Python" under SKILLS, "currently learning Python" in a summary), it
stays `supported` and both lines are shown, with a note, so the learning line is not hidden.
"I no longer use Python" is read as negated.

A related skill (for example Power BI for a Tableau requirement) gives `partial` credit, unless
the resume denies the skill itself. If the resume says "I have no experience with Kubernetes" and
has no line that names Kubernetes affirmatively, a Docker line earns nothing: Kubernetes is
`not_found` and the denial line is shown as the evidence. If the resume both names the skill and
denies it, the result is `unverified`, as before. A skill that is only being learned is not a
denial, so a related skill still earns `partial` credit there.

Limitations of matching:

- The negation and learning rules are small and can miss unusual wording. Wording they do not
  recognize is treated as a normal mention.
- A keyword does not show depth, recency, proficiency or years of use. The tool never infers
  any of these.
- `regression` also matches "regression testing" in QA. `js` also matches `Node.js`.
- Only skills in the built in list (about 55) are read from a job description. A skill outside
  the list, such as Terraform, Excel, R, Rust or Go, is not matched, not scored and not found
  automatically. The report lists the job line as NOT ASSESSED, and the person running the
  screen can name such a skill with `--extra-skill` (see "Requirements the tool cannot
  assess").

## How evidence is validated

- Evidence is copied from the resume. Snippets are exact slices of the original text.
  Case folding and whitespace collapsing are used only to find a match. The folded text is
  never stored or shown.
- Before a report is returned, every snippet is checked to exist in the resume text. A
  supported or partial result left with no traceable evidence becomes `unverified`.
- Model output is untrusted. Each item the model proposes must be found in the resume as a
  whole word match. Kept items are stored as the resume wrote them, not as the model did.
  Model skills need an affirmative mention. Model dates must sit on the role's line or the
  line after it. Years of experience are never taken from the model.
- Certifications need all key terms on one line. The line must sit under a CERTIFICATIONS
  heading, or a word such as "certified" or "certificate" must stand next to the key terms in
  the same sentence and clause. "Managed AWS infrastructure. Certified Scrum Master." is
  therefore not an AWS certification. Negated, in progress, planned or expired certifications
  are `unverified`.
- Degrees that are in progress or expected are `partial`, not completed. Dropped or
  ambiguous degrees are `unverified`.
- Date ranges in the future are not counted as work done. Impossible ranges are ignored.

**This is a text check, not a meaning check.** It shows the words exist in the resume. It
does not prove the resume says what the requirement needs, so a human must still review.

## Score

`100 x sum(weight x credit) / sum(weight)`, rounded to one decimal, ties half up.
Credit: supported 1.0, partial 0.5, not found 0, unverified 0.
Weights: required skill 3, experience 3, education 2, certification 2, preferred skill 1.
Repeated requirements are counted once, and a skill listed as both required and preferred
keeps the required classification.

The score summarizes how many listed requirements have supporting text. It is not a measure
of a candidate's quality, job performance or hiring success. Weights and credits are
judgment calls, not statistics.

## Why not found and unverified do not mean "lacks the skill"

Both statuses count as zero, so the score can understate a candidate. They mean the tool
found no usable evidence. Text extraction can miss content, resumes leave things out, and
the rules can misread wording. They are not proof of absence. Treat the
"NOT FOUND OR UNVERIFIED" list as a checklist for a person to read.

## Why human review is required

The tool uses keyword rules, simple pattern matching and optionally a small local model. It
can be wrong in both directions. Every report says that a human reviewer must make any hiring
decision, and the report model does not allow this to be turned off.

## Known limitations

- Scanned PDFs need OCR, which is not included.
- Multi column PDFs can scramble text order.
- Rule based parsing expects common headings such as SKILLS and EXPERIENCE.
- Degrees written as `BA` or `BS` are not recognized. `Bachelor of Arts` is.
- The evaluator reads skills from the raw resume text, so `CandidateProfile.skills` does not
  affect skill results. The profile does affect experience, education and the certification
  fallback.
- Job requirements are extracted with simple rules. A skill that a line marks as optional right
  next to its name ("Python is a plus", "Python (optional)", "Python is not required", "No
  Python experience needed") is scored as preferred, not required, and the report says so.
  Other wording ("Python, though not essential", "a plus: Python") is still read as required.
  A preferred line that names a degree or years of experience is not scored, and the report
  says so.
- An experience role is cut off at a later line only when that line has a year but no date
  range, or reads as a job title plus a company ("Data Analysis Intern, Foo Corp") with a title
  word in it. A bare title line ("Data Analyst") or an undated role with no title word cannot be
  told from the description of the role above it, so years may still be credited to the wrong
  role. A role whose description merely mentions the field ("reported to the data analysis team")
  still counts as being in that field.
- A line that offers a certification as an alternative to a degree is scored as the degree. A
  skill named on that line (for example AWS) is still scored as required. A line that combines
  a degree and a certification with "and" becomes one certification requirement that a resume
  rarely matches, and the report says so.
- A degree line counts as completed unless it carries wording the rules know: "expected", "in
  progress", "coursework toward", "credits towards", "working towards", "partial", "some
  college", "dropped out", "did not complete" and similar. Other ways of saying a degree is
  unfinished ("Bachelor of Science, 2 years") are read as completed.
- Keyword retrieval only matches words, plus the aliases and related skills in the built in
  list. In the synthetic check it misses three paraphrases: `a/b testing` against "a controlled
  test", `ci/cd` against "automated build and release pipelines", and `monitoring` against
  "dashboards and paging alerts". `monitoring` is not in the skill list, so no job description
  can ask for it. It also treats a word and its other forms as different words ("test" and
  "testing"), except for a one entry table that lets `forecasting` find "forecast". A general
  rule for word endings was tried and rejected, because it made `scikit-learn` match "learning"
  and `a/b testing` match any "test". Semantic retrieval is the intended route for paraphrases,
  and it has not been run against these three cases with a real model.
- Semantic retrieval was run once against a real embedding model (`nomic-embed-text`, by the
  project owner, on the first 22 synthetic queries). It always returns the nearest passages,
  even for a skill that is not in the resume, and the ranking depends on the model. Queries
  are sent without the prefixes that some embedding models expect, unless
  `EMBEDDING_QUERY_PREFIX` and `EMBEDDING_DOCUMENT_PREFIX` are set.
- Passages are chunks of up to 500 characters split at section headings, so a passage can
  include lines that have nothing to do with the requirement. Only the first 200 chunks of a
  resume are searched, and the report says when a resume was cut.
- A denial of a skill is found by small rules ("no experience with X"). Wording they miss, such
  as "X is not my area", is read as a normal mention. A resume that denies a skill in unusual
  words can still be marked `supported` or earn related skill credit. This was fixed for
  recognised denials only: they now block related skill credit and show the denial line
  (`tests/test_negative_evidence.py`).
- A role that ends in "Present" is counted up to the day you run the tool, so the years in a
  report grow over time. The date is taken once per run. Code that calls `screen_resume` can
  pass `today`, and every date calculation then uses it.
- `CHROMA_PERSIST_DIR` and `INCOMING_RESUME_DIR` are read into the settings but no code uses
  them. There is no vector database.
- An invalid value in `.env` for any setting (for example `LLM_TIMEOUT_SECONDS=abc`) stops
  the program with a clear message, even without `--llm`.
- Matching is by exact words, so it can miss experience or skills that are described in other
  words. A "Data Analyst" role does not match "experience in analytics", and a role that
  describes ETL pipelines does not match "data engineering". Those requirements are
  `unverified` for a reviewer to read, not `not_found`. No broad synonym list is used, because
  it would also create false matches.
- Aliases can equate things that differ. `js` also matches `Node.js` and `Vue.js`, and
  `regression` also matches "regression testing" in QA. Check such matches in the evidence.
- Related technologies earn partial credit (half the weight) and are not proof of
  proficiency. Docker alone (with no denial of Kubernetes) gives Kubernetes `partial`, Hadoop alone gives Spark `partial`,
  and MySQL alone gives SQL `partial`. The explanation says "may be transferable", but the
  credit still moves the score.
- Experience relevance is a word check on each role's own text, not an understanding of the
  work. It cannot prove that a role was in the field. A requirement that names no field
  ("5+ years of relevant experience", "in a related field") counts all dated years, and the
  report says that relevance was not checked. A short role line that has a year but no date
  range (for example "summer 2022") ends the role above it. If that line names the field, the
  requirement is `unverified`.
- Only one experience year requirement is assessed. If the job lists several, the report says
  that only the first was assessed. A range such as "3-5 years" uses its lower bound.
- A requirement line that offers a certification as an alternative to a degree ("Bachelor's
  degree or relevant certification") is scored as an education requirement only, and the report
  says so. A line that joins a degree and a certification with "and" still becomes one
  certification requirement made of the whole line, which no resume can meet, so check it by
  hand.
- A job heading that is not on the recognised list (for example "Technical Must Haves") is not
  read as a requirements heading. If a job then has no required lines at all, the report names
  the skills it found there and says they were not scored. Preferred labels inside a line need a
  colon or a dash ("Nice to have: Spark").

## About this project

what its authors intended. They do not show that it is right on real resumes.