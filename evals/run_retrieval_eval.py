"""Small synthetic check of passage retrieval. Run it from the project folder:

    python evals/run_retrieval_eval.py              keyword search, no network
    python evals/run_retrieval_eval.py --semantic   embeddings, needs Ollama and EMBEDDING_MODEL

It uses the same requirement mapping as the pipeline. A query counts as a hit when a returned
passage contains one of the quotes listed for it in retrieval_cases.py. See
docs/retrieval_evaluation.md for what the numbers mean and do not mean.
"""

import argparse
from collections import defaultdict

from pydantic import ValidationError
from retrieval_cases import RESUMES

from hr_screening.config import get_settings
from hr_screening.errors import ScreeningError
from hr_screening.models.report import MatchStatus, RequirementAssessment, RequirementKind
from hr_screening.rag.factory import build_embedder
from hr_screening.rag.requirements import retrieve_for_assessments
from hr_screening.rag.retrieval import LexicalRetriever
from hr_screening.rag.semantic import SemanticRetriever

TYPES = (
    "exact",
    "alias",
    "version",
    "related",
    "paraphrase",
    "negated",
    "aspirational",
    "text",
    "absent",
)


def check_cases() -> None:
    """Stop early if a case file mistake would make the numbers meaningless."""
    seen = set()
    for resume in RESUMES:
        for q in resume["queries"]:
            if q["id"] in seen:
                raise ValueError(f"Duplicate query id: {q['id']}")
            seen.add(q["id"])
            if q["type"] not in TYPES:
                raise ValueError(f"Unknown type in {q['id']}")
            RequirementKind(q["kind"])
            for quote in q["gold"] or []:
                if resume["text"].count(quote) != 1:
                    raise ValueError(f"The quote in {q['id']} must appear exactly once")
            if q["gold"] is None and q["mention"]:
                raise ValueError(f"{q['id']} has a mention but no gold quote")


def score_query(q: dict, passages: list) -> dict:
    row = {"id": q["id"], "type": q["type"], "returned": len(passages), "rank": None}
    row["answerable"] = q["gold"] is not None
    row["mention_ok"] = None
    for position, passage in enumerate(passages, start=1):
        if row["answerable"] and any(quote in passage.text for quote in q["gold"]):
            row["rank"] = position
            if q["mention"]:
                row["mention_ok"] = q["mention"] in passage.mention_kinds
            break
    if q["mention"] and row["rank"] is None:
        row["mention_ok"] = False
    return row


def run(make_retriever) -> list[dict]:
    """One row per query. make_retriever(resume_text) returns a retriever."""
    rows = []
    for resume in RESUMES:
        retriever = make_retriever(resume["text"])
        for q in resume["queries"]:
            assessment = RequirementAssessment(
                requirement=q["requirement"],
                kind=RequirementKind(q["kind"]),
                status=MatchStatus.NOT_FOUND,
                explanation="evaluation only",
            )
            entry = retrieve_for_assessments(retriever, [assessment])[0]
            rows.append(score_query(q, entry.passages))
    return rows


def describe(row: dict) -> str:
    if not row["answerable"]:
        return "correct, nothing returned" if row["returned"] == 0 else (
            f"wrong, {row['returned']} passages returned"
        )
    if row["rank"] is None:
        return f"miss ({row['returned']} passages returned)"
    text = f"hit at rank {row['rank']}"
    if row["mention_ok"] is not None:
        text += ", mention label " + ("ok" if row["mention_ok"] else "wrong")
    return text


def print_report(rows: list[dict], label: str) -> None:
    print(f"Retrieval check: {label}. Synthetic cases, {len(rows)} queries.\n")
    for row in rows:
        print(f"{row['id']:<26}{row['type']:<14}{describe(row)}")

    by_type = defaultdict(list)
    for row in rows:
        if row["answerable"]:
            by_type[row["type"]].append(row)
    print("\nAnswerable queries (a correct passage exists in the resume)")
    print(f"{'type':<14}{'queries':>8}{'rank 1':>8}{'top 3':>8}")
    for kind in TYPES:
        group = by_type.get(kind)
        if group:
            first = sum(r["rank"] == 1 for r in group)
            top3 = sum(r["rank"] is not None for r in group)
            print(f"{kind:<14}{len(group):>8}{first:>8}{top3:>8}")
    everyone = [r for group in by_type.values() for r in group]
    print(
        f"{'all':<14}{len(everyone):>8}"
        f"{sum(r['rank'] == 1 for r in everyone):>8}"
        f"{sum(r['rank'] is not None for r in everyone):>8}"
    )

    absent = [r for r in rows if not r["answerable"]]
    empty = sum(r["returned"] == 0 for r in absent)
    print(f"\nAbsent skills: nothing returned for {empty} of {len(absent)}.")
    flagged = [r for r in rows if r["mention_ok"] is not None]
    print(f"Mention labels correct: {sum(r['mention_ok'] for r in flagged)} of {len(flagged)}.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Small synthetic retrieval check.")
    parser.add_argument("--semantic", action="store_true", help="Use the Ollama embedding model")
    args = parser.parse_args(argv)
    check_cases()
    try:
        if args.semantic:
            embedder = build_embedder(get_settings())
            rows = run(lambda text: SemanticRetriever(text, embedder))
            label = f"semantic search, model {getattr(embedder, 'model', 'unknown')}"
        else:
            rows = run(LexicalRetriever)
            label = "keyword search (lexical)"
    except ValidationError:
        print("Error: invalid settings. Check your environment variables or .env.")
        return 2
    except ScreeningError as exc:
        print(f"Error: {exc}")
        return 2
    print_report(rows, label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())