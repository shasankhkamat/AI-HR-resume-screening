# AI HR Resume Screening

An explainable screening tool. It takes one PDF or DOCX resume and one job description and
prints a report: each requirement is marked supported, partial, not found or unverified,
with the exact resume line that supports it. It supports human review. It does not make
hiring decisions.

## Setup

Python 3.11 or newer.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

This installs the package with its three dependencies (pydantic, pydantic-settings and pypdf),
the dev tools (pytest and ruff) and a `hr-screening` command. The examples below use
`python -m hr_screening.main`, which does the same thing.

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
- A pipeline, a command line tool, pytest tests and Ruff

Not implemented (future work):

- Hosted LLM providers. Only a local Ollama server is supported.
- A vector database. Embeddings are computed for each run and kept in memory.
- Searching a pool of many resumes. The tool handles one resume at a time.
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

Options: `--job-file path.txt`, `--json`, `--resume-dir folder`, `--llm`,
`--rag lexical|semantic`. The sample resume is synthetic. Never commit real resumes.

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
weakness is that it only sees words. In the synthetic check it found nothing for `forecasting`
on a resume that says "Predicted monthly demand with ARIMA models".

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
4. Run with `--rag semantic`. Names ending in `-cloud` or `:cloud` are refused, as for chat
   models.

| Situation | Result |
|---|---|
| `EMBEDDING_MODEL` missing, a cloud model name, a bad URL or an invalid setting | Message on screen and exit code 2. No report and no silent fallback |
| Ollama not running, timeout, model not installed, server error, unusable vectors | Keyword search is used instead. The report says why, for example "Semantic search failed (the model server could not be reached), so keyword search was used instead." |
| Keyword search also fails | The screening result is returned without passages, with a note |

## Retrieval check on synthetic cases

`evals/run_retrieval_eval.py` runs 22 queries over three invented resumes and prints what was
found. Keyword search found a correct passage at rank 1 for 17 of 19 answerable queries,
returned nothing for all 3 absent skills, and labelled both a negated and a learning mention
correctly. It missed two paraphrases. The cases are tiny and were written by the same
assistant that wrote the code, so this is a sanity check and not a benchmark. Semantic search
has not been measured. `docs/retrieval_evaluation.md` has the full output, what the check
found (one bug, now fixed) and how to run it for semantic search.

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
```

No test needs Ollama, a downloaded model, an API key or the internet. The tests use fake
transports and a fake embedder. A few tests start a tiny HTTP server on 127.0.0.1 inside the
test process to exercise the real HTTP code, and one test starts Python subprocesses to check
that results do not depend on hash randomization. Nothing leaves your computer.

What the tests prove: the plumbing, the validation of model output and of vectors, the
fallbacks and error messages, that passages are exact slices of the resume, and that
`--rag` never changes a status or the score. What they do not prove: that Ollama's chat or
`/api/embed` endpoint works on your machine, or that semantic search ranks passages well. The
fake embedder hashes word counts. It is not a language model and its results mean nothing about
real embeddings. `tests/test_retrieval_eval.py` also pins the numbers reported in
`docs/retrieval_evaluation.md`, so the document and the code cannot drift apart unnoticed.

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
- Keyword retrieval only matches words, plus the aliases and related skills in the built in
  list. In the synthetic check it missed `forecasting` against "Predicted monthly demand with
  ARIMA models" and `a/b testing` against "a controlled test".
- Semantic retrieval has not been run against a real embedding model here. It always returns
  the nearest passages, even for a skill that is not in the resume, and the ranking depends on
  the model. Queries are sent without the prefixes that some embedding models expect.
- Passages are chunks of up to 500 characters split at section headings, so a passage can
  include lines that have nothing to do with the requirement. Only the first 200 chunks of a
  resume are searched, and the report says when a resume was cut.
- The evaluator gives partial credit for a related skill even when the resume says the skill
  itself was not used. In the synthetic test resume, "no hands on experience with Kubernetes"
  plus a Docker line gives Kubernetes a partial status. The retrieved passage shows the
  negated line next to it. The rule was left as it is. It is one to revisit.
- `CHROMA_PERSIST_DIR` and `INCOMING_RESUME_DIR` are read into the settings but no code uses
  them. There is no vector database.
- An invalid value in `.env` for any setting (for example `LLM_TIMEOUT_SECONDS=abc`) stops
  the program with a clear message, even without `--llm`.

## About this project

This project was built with AI assistance (Claude), in small batches. Each batch added code
and tests, and the project owner applied, ran and committed them. The tests and the synthetic
retrieval cases were also written with AI assistance. That means they check that the code does
what its authors intended. They do not show that it is right on real resumes.