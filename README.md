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
- An `LLMClient` interface, a fake client, and verification of model output
- A pipeline with rule based fallback, a command line tool, pytest tests and Ruff

Not implemented (future work):

- A real LLM provider adapter. Live inference does not exist yet.
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

## Run it

```powershell
python docs\make_sample_resume.py
python -m hr_screening.main --resume data\incoming\sample_resume.docx --job-text "Requirements`n- 3+ years of experience`n- Python and SQL`n- Bachelor's degree"
```

Options: `--job-file path.txt`, `--json`, `--resume-dir folder`.
The sample resume is synthetic. Never commit real resumes.

## Tests

```powershell
pytest
ruff check .
```

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

The tool uses keyword rules and simple pattern matching. It can be wrong in both directions.
Every report says that a human reviewer must make any hiring decision, and the report model
does not allow this to be turned off.

## LLM status

The `LLMClient` interface, output validation and a fake client exist and are tested.
No real provider adapter is connected, so live inference is not implemented.
If a client is supplied and it fails or returns something invalid, the rule based parser is
used instead and the report says so. Logs name only the error class, never resume text.
Live use will need a chosen provider plus `LLM_PROVIDER`, `LLM_API_KEY` and `LLM_MODEL` set
in `.env`.

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