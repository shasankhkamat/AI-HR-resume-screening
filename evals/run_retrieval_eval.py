"""Small synthetic check of passage retrieval. Run it from the project folder:

    python evals/run_retrieval_eval.py              keyword search, no network
    python evals/run_retrieval_eval.py --semantic   embeddings, needs Ollama and EMBEDDING_MODEL
    python evals/run_retrieval_eval.py --semantic --scores
                                                    also print similarity scores and a
                                                    threshold sweep, to choose a cutoff

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
    row["top_score"] = passages[0].score if passages else None
    row["top_scores"] = [p.score for p in passages]
    row["trace"] = [
        {"score": p.score, "section": p.section, "start": p.start, "end": p.end, "text": p.text}
        for p in passages
    ]
    row["gold_score"] = None
    row["answerable"] = q["gold"] is not None
    row["mention_ok"] = None
    for position, passage in enumerate(passages, start=1):
        if row["answerable"] and any(quote in passage.text for quote in q["gold"]):
            row["rank"] = position
            row["gold_score"] = passage.score
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


SWEEP_THRESHOLDS = tuple(round(step * 0.05, 2) for step in range(20))  # 0.00 to 0.95


def threshold_sweep(rows: list[dict], thresholds=SWEEP_THRESHOLDS) -> list[dict]:
    """For each cutoff, what a minimum similarity would have done to this run.

    A passage is kept when its score is above the cutoff, the same rule SemanticRetriever
    uses. Scores only fall as rank falls, so a cutoff can remove the lower ranked passages of
    a query but never reorder them. An absent skill query becomes empty when its best score is
    at or below the cutoff. An answerable query keeps its gold passage while the gold passage
    scores above the cutoff. A query whose gold passage was not in the top 3 to begin with
    has no gold score and counts as already missed.
    """
    absent = [r for r in rows if not r["answerable"]]
    answerable = [r for r in rows if r["answerable"]]
    missed = sum(r["gold_score"] is None for r in answerable)
    table = []
    for cutoff in thresholds:
        empty = [r for r in absent if r["top_score"] is None or r["top_score"] <= cutoff]
        kept = [r for r in answerable if r["gold_score"] is not None and r["gold_score"] > cutoff]
        table.append(
            {
                "threshold": cutoff,
                "absent_empty": len(empty),
                "absent_total": len(absent),
                "gold_kept": len(kept),
                "answerable_total": len(answerable),
                "already_missed": missed,
            }
        )
    return table


def _score(value) -> str:
    return "none" if value is None else f"{value:.3f}"


def _snippet(text: str, width: int = 88) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 3] + "..."


def print_traces(rows: list[dict]) -> None:
    """The passages behind the queries that did not simply work: absent skills, and answerable
    queries whose gold passage was not first. Without this the scores cannot be explained."""
    shown = [r for r in rows if not r["answerable"] or r.get("rank") != 1]
    shown = [r for r in shown if r.get("trace")]
    if not shown:
        return
    print("\nPassages returned for absent skills and for answerable queries not at rank 1")
    for row in shown:
        print(f"{row['id']} ({row['type']})")
        for number, item in enumerate(row["trace"], start=1):
            where = f"{item['section'] or 'no section'}, chars {item['start']}-{item['end']}"
            print(f"  {number}. {item['score']:.3f}  [{where}]  {_snippet(item['text'])}")


def print_scores(rows: list[dict]) -> None:
    """Similarity scores behind the report, for choosing a minimum similarity."""
    print("\nScore diagnostics (cosine similarity, higher is closer)")
    print(f"{'query':<26}{'type':<14}{'top 1':>8}{'gold passage':>14}   top 3 scores")
    for row in rows:
        gold = _score(row["gold_score"]) if row["answerable"] else "-"
        top3 = " ".join(f"{s:.3f}" for s in row.get("top_scores", [])) or "none"
        print(
            f"{row['id']:<26}{row['type']:<14}{_score(row['top_score']):>8}{gold:>14}   {top3}"
        )
    print_traces(rows)

    absent = [r for r in rows if not r["answerable"] and r["top_score"] is not None]
    found = [r for r in rows if r["gold_score"] is not None]
    if absent and found:
        highest = max(absent, key=lambda r: r["top_score"])
        lowest = min(found, key=lambda r: r["gold_score"])
        print(f"\nHighest top score, absent skill: {highest['top_score']:.3f} ({highest['id']})")
        print(f"Lowest gold passage score:       {lowest['gold_score']:.3f} ({lowest['id']})")
        if lowest["gold_score"] > highest["top_score"]:
            print("A cutoff between those two would separate this run.")
        else:
            print("The two ranges overlap, so no single cutoff separates this run.")

    print("\nThreshold sweep (what a minimum similarity would have done)")
    print(f"{'cutoff':>8}{'absent empty':>16}{'gold kept':>14}{'gold lost':>12}")
    for line in threshold_sweep(rows):
        kept = line["gold_kept"]
        lost = line["answerable_total"] - kept - line["already_missed"]
        print(
            f"{line['threshold']:>8.2f}"
            f"{line['absent_empty']:>10} of {line['absent_total']}"
            f"{kept:>9} of {line['answerable_total']}"
            f"{lost:>12}"
        )
    missed = threshold_sweep(rows, (0.0,))[0]["already_missed"]
    print(f"Gold lost counts only passages found at cutoff 0.00. {missed} were missed already.")
    print("Small synthetic set. These numbers show the trade off, they do not prove a safe cutoff.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Small synthetic retrieval check.")
    parser.add_argument("--semantic", action="store_true", help="Use the Ollama embedding model")
    parser.add_argument(
        "--scores",
        action="store_true",
        help="Also print similarity scores and a threshold sweep (needs --semantic)",
    )
    args = parser.parse_args(argv)
    if args.scores and not args.semantic:
        parser.error("--scores needs --semantic, because keyword scores are not similarities")
    check_cases()
    try:
        if args.semantic:
            embedder = build_embedder(get_settings())
            rows = run(lambda text: SemanticRetriever(text, embedder))
            label = f"semantic search, model {getattr(embedder, 'model', 'unknown')}"
            if getattr(embedder, "query_prefix", "") or getattr(embedder, "document_prefix", ""):
                label += (
                    f", query prefix {getattr(embedder, 'query_prefix', '').strip()!r}"
                    f", document prefix {getattr(embedder, 'document_prefix', '').strip()!r}"
                )
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
    if args.scores:
        print_scores(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())