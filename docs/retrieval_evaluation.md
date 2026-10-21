# Retrieval check on synthetic cases

This is a small sanity check of the passage retrieval behind `--rag`. It is not a benchmark.

## What was tested

Four short synthetic resumes (a data analyst, a software engineer, a marketing analyst, a cloud
platform engineer) and 40 queries. The first 22 queries were written before any result was
seen. The other 18 were added after the first real semantic run, to test absent skills that
sit next to skills the resume does have, so they are not independent of what went wrong.

A query is a requirement as the pipeline would pass it to the retriever, for example the skill
`kubernetes` or the education requirement `Bachelor level education or higher`. Each query
lists one or more quotes from the resume that a good answer should contain. Absent skills list
none, and the correct answer there is to return nothing.

A query is a hit when one of the top three passages contains a listed quote. The check goes
through the same `retrieve_for_assessments` function the pipeline uses, so it measures the
real behavior. The cases are in `evals/retrieval_cases.py` and the runner is
`evals/run_retrieval_eval.py`. The case types:

| Type | What it tests | Queries |
|---|---|---|
| exact | The skill is written as in the requirement | 9 |
| alias | The resume uses another name (`JS`, `sklearn`) | 2 |
| version | The resume writes `Python3` | 1 |
| related | Only a related skill is present (Power BI for Tableau) | 3 |
| paraphrase | The resume describes the skill without naming it | 5 |
| negated | The resume says it has not used the skill | 2 |
| aspirational | The resume says it is learning the skill | 2 |
| text | Education and certification requirements | 7 |
| absent | The resume has nothing relevant | 9 |

## Results: keyword search (`--rag lexical`)

```text
Retrieval check: keyword search (lexical). Synthetic cases, 40 queries.

analyst-python            exact         hit at rank 1
analyst-sql               exact         hit at rank 1
analyst-git               exact         hit at rank 1
analyst-tableau           related       hit at rank 1
analyst-forecasting       paraphrase    hit at rank 1
analyst-docker            absent        correct, nothing returned
analyst-tensorflow        absent        correct, nothing returned
analyst-degree            text          hit at rank 1
analyst-cert              text          hit at rank 1
engineer-python           version       hit at rank 1
engineer-javascript       alias         hit at rank 1
engineer-sklearn          alias         hit at rank 1
engineer-airflow          exact         hit at rank 1
engineer-kubernetes       negated       hit at rank 1, mention label ok
engineer-spark            aspirational  hit at rank 1, mention label ok
engineer-snowflake        absent        correct, nothing returned
engineer-pytorch          absent        correct, nothing returned
engineer-degree           text          hit at rank 1
marketing-looker          exact         hit at rank 1
marketing-visualization   paraphrase    hit at rank 1
marketing-abtesting       paraphrase    miss (0 passages returned)
marketing-python          absent        correct, nothing returned
marketing-tableau         related       hit at rank 1
marketing-degree          text          hit at rank 1
marketing-cert            text          hit at rank 1
cloud-linux               exact         hit at rank 1
cloud-git                 exact         hit at rank 1
cloud-docker              exact         hit at rank 1
cloud-aws                 exact         hit at rank 1
cloud-gcp                 related       hit at rank 1
cloud-cicd                paraphrase    miss (0 passages returned)
cloud-monitoring          paraphrase    miss (0 passages returned)
cloud-kubernetes          aspirational  hit at rank 1, mention label ok
cloud-azure               negated       hit at rank 1, mention label ok
cloud-terraform           absent        correct, nothing returned
cloud-java                absent        correct, nothing returned
cloud-spark               absent        correct, nothing returned
cloud-sql                 absent        correct, nothing returned
cloud-degree              text          hit at rank 1
cloud-cert                text          hit at rank 1

Answerable queries (a correct passage exists in the resume)
type           queries  rank 1   top 3
exact                9       9       9
alias                2       2       2
version              1       1       1
related              3       3       3
paraphrase           5       2       2
negated              2       2       2
aspirational         2       2       2
text                 7       7       7
all                 31      28      28

Absent skills: nothing returned for 9 of 9.
Mention labels correct: 4 of 4.
```

In short: 28 of 31 answerable queries found a correct passage at rank 1 (and so in the top
three). All 9 absent skills returned nothing. All 4 skill mention labels (negated,
aspirational) were correct. Before the Phase 3 change below the result was 27 of 31, with
`analyst-forecasting` missed. The semantic comparison table further down was measured before
that change, so its keyword column still says 17 of 19 and 1 of 3 paraphrases. That column
would now read 18 of 19 and 2 of 3 if `analyst-forecasting` is one of the first 22 queries,
which was not checked against the original run.

### The four paraphrase cases (Phase 3 investigation)

Each case was traced by printing the query terms, the terms of the passage that holds the
answer, and what overlaps. The tokenizer lower cases, drops stop words and strips a plural s.
Nothing else is folded.

| Case | Query terms | Overlap with the answer passage | Cause | Outcome |
|---|---|---|---|---|
| `analyst-forecasting` | `forecasting` | none, because the passage has `forecast` | Word form. The line is "Predicted monthly demand with ARIMA models and reported forecast error to the planning team", so this was never a pure paraphrase | Fixed. Now a hit at rank 1 |
| `marketing-abtesting` | `ab`, `ab testing`, `testing` | none | The resume never says A/B or testing. "controlled test" is a real paraphrase | Not fixed. Keyword search limit |
| `cloud-cicd` | `ci`, `cd`, `ci cd` | none | "automated build and release pipelines" describes CI/CD without naming it | Not fixed. Keyword search limit |
| `cloud-monitoring` | `monitoring` | none | "dashboards and paging alerts" shares no word. Also `monitoring` is not in the skill list, so a job description can never produce this query | Not fixed. Keyword search limit, and a query the command line cannot issue |

The fix for `forecasting` is a one entry table in `src/hr_screening/rag/lexical.py`
(`_WORD_FORMS`). It adds `forecast` and `forecasted` to the query at the lower related weight.
Only keyword search uses it. The evaluator's skill matching does not, so no status or score
changes: on the same resume the evaluator still reports `forecasting` as not found, and the
report shows the passage for a person to read. Across all 55 skills and the four resumes
(220 pairs), this changed the result of exactly one pair.

What was tried and rejected: a general rule that strips `ing`. It also fixed `forecasting`,
but it changed 9 of the 220 pairs. `scikit-learn` began to match "Currently learning
Kubernetes" (through the alias `scikit learn`), and `a/b testing` began to match any "test" or
"testing", such as "idle test server". It would also have turned `marketing-abtesting` into a
hit only because of the generic word "test". That would have improved the number without
improving the retrieval, so it was not kept. `tests/test_phase3_retrieval.py` fails if such a
rule is added.

No synonym was added for the other three. `pipeline` as an alias for `ci/cd` would match data
and ML pipelines. `controlled test` as an alias for `a/b testing` would match QA wording. These
are the cases semantic search is meant to help with. That is untested here: no Ollama server
was available in Phase 3, so semantic retrieval was not run on these four cases.

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

The owner of the project ran the first 22 queries on a Windows computer with the Ollama model
`nomic-embed-text` and reported these numbers. The assistant that wrote the code did not run
that model and has not seen the passages it returned. The 18 later queries have not been run
with a real embedding model.

| First 22 queries, `nomic-embed-text`, no task prefixes | Keyword | Semantic |
|---|---:|---:|
| Correct passage at rank 1 (of 19 answerable) | 17 | 18 |
| Correct passage in the top 3 (of 19) | 17 | 19 |
| Paraphrases at rank 1 (of 3) | 1 | 2 |
| Paraphrases in the top 3 (of 3) | 1 | 3 |
| Absent skills correctly empty (of 3) | 3 | 0 |
| Negated and aspirational labels correct (of 2) | 2 | 2 |

Semantic search found all three paraphrases in the top 3 (two at rank 1), and it returned
passages for all three absent skills (`analyst-docker`, `engineer-snowflake`,
`marketing-python`). The highest top score for an absent skill was 0.420 and the lowest score
of a correct passage was 0.399. The ranges overlap, so no minimum similarity would separate
them on these cases, and the minimum similarity is still 0.

Why absent skills return passages: semantic search always has a nearest passage, the minimum
similarity is 0, and these resumes have only 4 to 6 passages with 3 returned per query. Which
passages came back for the three absent skills has not been recorded. Run with `--scores` to
print them.

Some embedding models are trained with a short label in front of each text. For
`nomic-embed-text` the label is `search_query:` for requirements and `search_document:` for
resume passages. Without them the model may separate related and unrelated text less well.
Whether the labels help on these cases is untested. They are off by default and can be tried
without changing code:

```powershell
$env:EMBEDDING_QUERY_PREFIX = "search_query:"
$env:EMBEDDING_DOCUMENT_PREFIX = "search_document:"
python evals/run_retrieval_eval.py --semantic --scores
```

Compare the result with a run that does not set them. The labels are added only to the text
that is embedded. The passages shown in a report never contain them.

To measure it, set up an Ollama embedding model (see the README) and run:

```powershell
python evals/run_retrieval_eval.py --semantic
```

The output has the same layout as above. If the server is unreachable or `EMBEDDING_MODEL` is
not set, the script stops with a message and does not fall back to keyword search. Add the
output to this file when you have it.

To see the similarity scores behind the numbers, add `--scores`. It prints the top score and
the gold passage score for every query, then a sweep that shows, for each possible minimum
similarity, how many absent skill queries would return nothing and how many gold passages
would be lost. It also prints the top three scores of every query and the passages behind
absent skills and behind answerable queries that were not at rank 1. It only reports. The
minimum similarity stays at 0, so semantic search still returns passages for absent skills
until a cutoff is chosen from real scores. `--scores` needs `--semantic`, because keyword
scores are not similarities.

## What these results suggest

- Keyword search is reliable when the resume uses the requirement's words, a known alias or a
  known related skill, and it labels negated and learning mentions correctly.
- It fails quietly when the resume paraphrases the skill. The report then says "No matching
  passage found", which does not mean the candidate lacks the skill.
- It returns nothing for absent skills, where semantic search returned its nearest passages
  anyway.
- Semantic search found the paraphrases that keyword search missed, and returned passages for
  skills the resume does not have. In a report those passages are labelled as related text,
  and no status or score depends on them.

## What they do not prove

- The cases are small (40 queries) and invented. The same AI assistant wrote the retrieval
  code and the cases, but it knew how the code works, so the cases are probably easier than
  real resumes. The last 18 were written after a real result was known.
- Real resumes are longer, messier and come from PDFs with scrambled text. None were used.
- A hit means a passage containing a quote was in the top three. It does not mean the passage
  shows the candidate has the skill.
- Semantic numbers come from one model on 22 queries, run by the project owner. They are not
  a benchmark, and nothing here measures speed or any effect on a hiring decision.