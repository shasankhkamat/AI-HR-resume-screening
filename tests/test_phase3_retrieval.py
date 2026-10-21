
import pytest

from hr_screening.extraction.job_parser import parse_job_description
from hr_screening.extraction.skills_vocab import SKILL_ALIASES
from hr_screening.pipeline import screen_resume
from hr_screening.rag.chunking import Chunk
from hr_screening.rag.lexical import LexicalIndex, skill_query
from hr_screening.rag.retrieval import LexicalRetriever

ANALYST = """SKILLS
Languages: Python, SQL

EXPERIENCE
Senior Analyst, Example Corp   Jan 2022 - Present
- Predicted monthly demand with ARIMA models and reported forecast error to the planning team.
"""


@pytest.fixture
def screen_file_rag(tmp_path, make_docx, today):
    """The same resume screened without and with keyword retrieval."""

    def _run(lines, job):
        path = make_docx(tmp_path / "cv.docx", lines)
        plain = screen_resume(path, job, today=today)
        with_rag = screen_resume(path, job, today=today, rag_mode="lexical")
        return plain, with_rag

    return _run


def chunks_of(*texts):
    return [Chunk(i, 0, len(t), t, None) for i, t in enumerate(texts)]


def ranked(texts, skill):
    return [h.chunk.index for h in LexicalIndex(chunks_of(*texts)).search(skill_query(skill), 3)]


# --- analyst-forecasting: the same word in another form -------------------------------------


def test_forecasting_finds_the_forms_a_resume_uses():
    assert ranked(["Reported forecast error weekly"], "forecasting") == [0]
    assert ranked(["Forecasted demand for retail"], "forecasting") == [0]
    assert ranked(["Weekly forecasts for finance"], "forecasting") == [0]


def test_the_exact_word_still_outranks_another_form():
    assert ranked(["Reported forecast error", "Built forecasting models"], "forecasting") == [1, 0]


def test_forecasting_returns_nothing_when_the_resume_has_no_form_of_the_word():
    assert ranked(["Built dashboards in Power BI", "Fore and aft sails"], "forecasting") == []


def test_the_passage_found_for_forecasting_is_an_exact_slice_of_the_resume():
    passages = LexicalRetriever(ANALYST).for_skill("forecasting")
    assert len(passages) == 1
    passage = passages[0]
    assert ANALYST[passage.start : passage.end] == passage.text
    assert "reported forecast error" in passage.text
    assert passage.matched_terms == ("forecast",)
    assert passage.mention_kinds == ()  # found through a word form, not the skill name itself


def test_the_word_forms_do_not_change_screening(screen_file_rag):
    plain, with_rag = screen_file_rag(ANALYST.splitlines(), "Requirements\n- Forecasting\n")
    assert plain.assessments == with_rag.assessments
    assert plain.coverage_score == with_rag.coverage_score
    # The evaluator still reads only the skill's own name, so the skill is not found, even
    # though keyword search now shows a passage for a person to read.
    assert plain.assessments[0].status.value == "not_found"
    assert with_rag.retrieval[0].passages


# --- why a general "ing" rule was rejected ---------------------------------------------------


def test_scikit_learn_does_not_match_the_word_learning():
    text = ["Currently learning Kubernetes", "Wrote Python scripts"]
    assert ranked(text, "scikit-learn") == []


def test_a_b_testing_does_not_match_any_test_or_testing_word():
    assert ranked(["Shut down the idle test server every night"], "a/b testing") == []


# --- the three misses that stay, and why ------------------------------------------------------


def test_ab_testing_is_not_found_in_a_controlled_test_paraphrase():
    # Keyword search cannot know that a controlled test is an A/B test. Semantic search can.
    assert ranked(["Compared two email subject lines with a controlled test"], "a/b testing") == []


def test_ci_cd_is_not_found_in_a_pipeline_description():
    # 'pipeline' is not an alias on purpose: data pipelines and ML pipelines are not CI/CD.
    text = ["Automated build and release pipelines", "Built data pipelines in Airflow"]
    assert ranked(text, "ci/cd") == []


def test_monitoring_is_not_found_in_an_alerting_description():
    assert ranked(["Set up dashboards and paging alerts"], "monitoring") == []


def test_monitoring_cannot_be_asked_for_by_a_job_description():
    # The job parser only reads skills from the built in vocabulary, so this eval query can
    # never come out of the command line.
    assert "monitoring" not in SKILL_ALIASES
    assert parse_job_description("Requirements\n- Monitoring\n").required_skills == []