# Retrieval check on synthetic cases

This is a small sanity check of the passage retrieval behind `--rag`. It is not a benchmark.

## What was tested

Three short synthetic resumes (a data analyst, a software engineer, a marketing analyst) and
22 queries. A query is a requirement as the pipeline would pass it to the retriever, for
example the skill `kubernetes` or the education requirement `Bachelor level education or
higher`. Each query lists one or more quotes from the resume that a good answer should
contain. Absent skills list none, and the correct answer there is to return nothing.

A query is a hit when one of the top three passages contains a listed quote. The check goes
through the same `retrieve_for_assessments` function the pipeline uses, so it measures the
real behavior. The cases are in `evals/retrieval_cases.py` and the runner is
`evals/run_retrieval_eval.py`. The case types:

| Type | What it tests | Queries |
|---|---|---|
| exact | The skill is written as in the requirement | 5 |
| alias | The resume uses another name (`JS`, `sklearn`) | 2 |
| version | The resume writes `Python3` | 1 |
| related | Only a related skill is present (Power BI for Tableau) | 1 |
| paraphrase | The resume describes the skill without naming it | 3 |
| negated | The resume says it has not used the skill | 1 |
| aspirational | The resume says it is learning the skill | 1 |
| text | Education and certification requirements | 5 |
| absent | The resume has nothing relevant | 3 |

## Results: keyword search (`--rag lexical`)

```text
Retrieval check: keyword search (lexical). Synthetic cases, 22 queries.

analyst-python            exact         hit at rank 1
analyst-sql               exact         hit at rank 1
analyst-git               exact         hit at rank 1
analyst-tableau           related       hit at rank 1
analyst-forecasting       paraphrase    miss (0 passages returned)
analyst-docker            absent        correct, nothing returned
analyst-degree            text          hit at rank 1
analyst-cert              text          hit at rank 1
engineer-python           version       hit at rank 1
engineer-javascript       alias         hit at rank 1
engineer-sklearn          alias         hit at rank 1
engineer-airflow          exact         hit at rank 1
engineer-kubernetes       negated       hit at rank 1, mention label ok
engineer-spark            aspirational  hit at rank 1, mention label ok
engineer-snowflake        absent        correct, nothing returned
engineer-degree           text          hit at rank 1
marketing-looker          exact         hit at rank 1
marketing-visualization   paraphrase    hit at rank 1
marketing-abtesting       paraphrase    miss (0 passages returned)
marketing-python          absent        correct, nothing returned
marketing-degree          text          hit at rank 1
marketing-cert            text          hit at rank 1

Answerable queries (a correct passage exists in the resume)
type           queries  rank 1   top 3
exact                5       5       5
alias                2       2       2
version              1       1       1
related              1       1       1
paraphrase           3       1       1
negated              1       1       1
aspirational         1       1       1
text                 5       5       5
all                 19      17      17

Absent skills: nothing returned for 3 of 3.
Mention labels correct: 2 of 2.
```

In short: 17 of 19 answerable queries found a correct passage at rank 1 (and so in the top
three). The three absent skills returned nothing. Both skill mention labels (negated,
aspirational) were correct.

### The two misses

- `analyst-forecasting`: the resume says "Predicted monthly demand with ARIMA models". There is
  no shared word with the requirement `forecasting`, so keyword search returns nothing.
- `marketing-abtesting`: the resume says "Compared two email subject lines with a controlled
  test". Keyword search does not connect "testing" to "test" (it only strips plural s), and
  the single letters in "a/b" are dropped from the query.

These are the cases semantic search is meant to help with. That is untested, see below.

### One hit that is less good than it looks

`marketing-visualization` (`data visualization`) found the right passage, but not by
understanding it. The match came from the built in related skills list, which links data
visualization to Looker, and from the generic word "data". A resume with a different charting
tool would probably have missed. Treat it as luck, not as a paraphrase success.

### A bug this check found

On the first run, `engineer-python` missed: 16 of 19 instead of 17. The resume writes
`Python3`, the evaluator accepts a version number after a skill name, but the keyword index
treated `python3` as a different word. The evaluator would have called Python supported while
retrieval showed no passage for it. The tokenizer now drops a trailing version number from
words such as `python3` and `html5` (`k8s`, `s3` and `x86` are left alone). The cases were not
changed between the two runs. `tests/test_rag_versions.py` covers the fix.

## Results: semantic search (`--rag semantic`)

Not measured. No embedding model was run for this project. The automated tests use a fake
embedder that hashes word counts, so a result from it says nothing about real embeddings. The
script's `--semantic` path was run once against the fake embedder to confirm that it works
end to end, and those numbers are not reported.

To measure it, set up an Ollama embedding model (see the README) and run:

```powershell
python evals/run_retrieval_eval.py --semantic
```

The output has the same layout as above. If the server is unreachable or `EMBEDDING_MODEL` is
not set, the script stops with a message and does not fall back to keyword search. Add the
output to this file when you have it.

## What these results suggest

- Keyword search is reliable when the resume uses the requirement's words, a known alias or a
  known related skill, and it labels negated and learning mentions correctly.
- It fails quietly when the resume paraphrases the skill. The report then says "No matching
  passage found", which does not mean the candidate lacks the skill.
- It returns nothing for absent skills, where semantic search is likely to return its nearest
  passages anyway.

## What they do not prove

- The cases are small (22 queries) and invented. The same AI assistant wrote the retrieval
  code and the cases, before seeing results, but it knew how the code works, so the cases are
  probably easier than real resumes.
- Real resumes are longer, messier and come from PDFs with scrambled text. None were used.
- A hit means a passage containing a quote was in the top three. It does not mean the passage
  shows the candidate has the skill.
- Nothing here measures semantic search, speed or any effect on a hiring decision.