# AI HR Resume Screening

An explainable screening MVP. It takes one PDF or DOCX resume and one job description and
prints a report: each requirement is marked supported, partial, not found or unverified,
with the exact resume line that supports it. It supports human review. It does not make
hiring decisions.

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
- A pipeline, a command line tool, pytest tests and Ruff

Not implemented (future work):

- Hosted LLM providers. Only a local Ollama server is supported.
- RAG, embeddings or semantic retrieval. One resume and one job description fit in one prompt.
  RAG becomes useful for searching a pool of many resumes.
- OCR for scanned PDFs
- Any fairness, bias or compliance guarantee. No such testing has been done.

## Architecture

```text
resume file -> ingestion (validate, extract text)
job text    -> extraction.job_parser -> JobRequirements
resume text -> extraction.resume_parser (rules)  or  llm.extractor (optional, verified)
                -> CandidateProfile
evaluation.evaluator -> ScreeningReport (status + evidence + score + limitations)
```

With `--llm`, `llm.factory` builds an `OllamaClient` from the settings. The client implements
the small `LLMClient` interface (`complete(system_prompt, user_prompt) -> str`) and only
returns the model's text. Parsing, validation and the evidence checks stay in `llm.extractor`.

## Run it

Rule based mode (the default). It makes no network request and needs no model.

```powershell
python docs\make_sample_resume.py
python -m hr_screening.main --resume data\incoming\sample_resume.docx --job-text "Requirements`n- 3+ years of experience`n- Python and SQL`n- Bachelor's degree"
```

LLM mode. It needs Ollama and a downloaded model (see the next section).

```powershell
python -m hr_screening.main --resume data\incoming\sample_resume.docx --job-text "Requirements`n- 3+ years of experience`n- Python and SQL`n- Bachelor's degree" --llm
```

Options: `--job-file path.txt`, `--json`, `--resume-dir folder`, `--llm`.
The sample resume is synthetic. Never commit real resumes.

## Local LLM mode with Ollama

### What is sent

With `--llm`, the resume text and a fixed set of extraction instructions are sent to the
Ollama server at `OLLAMA_BASE_URL`. The job description is not sent. Nothing is sent without
`--llm`.

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

Copy `.env.example` to `.env` and set at least these two lines. Use the exact name from
`ollama list`.

```text
LLM_PROVIDER=ollama
LLM_MODEL=<model-name>
```

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
- The report itself contains resume lines as evidence. Treat the output as sensitive.
- Local inference has no per request API charge. It does need local compute, disk space and
  model downloads, and it can be slow without a GPU.

### Accuracy

A model that runs and passes the tests is not a model that is accurate. The automated tests
use fake servers and prove only that the plumbing and the error handling work. The evidence
check proves that extracted text exists in the resume. It does not prove the text means what
the model says. Treat every LLM result as a suggestion for a human reviewer.

## Tests

```powershell
pytest
ruff check .
```

No test needs Ollama, a downloaded model, an API key or the internet. The tests use fake
transports. A few tests start a tiny HTTP server on 127.0.0.1 inside the test process to
exercise the real HTTP code. Nothing leaves your computer.

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

A related skill (for example Power BI for a Tableau requirement) gives `partial` credit.

Limitations of matching:

- The negation and learning rules are small and can miss unusual wording. Wording they do not
  recognize is treated as a normal mention.
- A keyword does not show depth, recency, proficiency or years of use. The tool never infers
  any of these.
- `regression` also matches "regression testing" in QA. `js` also matches `Node.js`.
- Skills outside the built in list are matched only when they are named in the job
  description, as plain whole words.

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
- Certifications need all key terms on one line that sits under a CERTIFICATIONS heading or
  names a certificate. Negated, in progress, planned or expired certifications are
  `unverified`.
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
- Job requirements are extracted with simple rules. A job line such as "Python not required"
  is still read as requiring Python.
- An invalid value in `.env` for any setting (for example `LLM_TIMEOUT_SECONDS=abc`) stops
  the program with a clear message, even without `--llm`.