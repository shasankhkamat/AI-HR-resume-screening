"""Date handling: implausible and future ranges must not crash or inflate experience."""

from datetime import date

import pytest

from hr_screening.evaluation.evaluator import evaluate
from hr_screening.extraction.dates import month_span, total_years, unusable_date_ranges
from hr_screening.extraction.resume_parser import parse_resume
from hr_screening.models.candidate import WorkExperience
from hr_screening.models.job import JobRequirements
from hr_screening.models.report import MatchStatus

TODAY = date(2026, 10, 21)


def _role(start: str, end: str) -> WorkExperience:
    return WorkExperience(heading="Role", start=start, end=end)


def test_ranges_that_end_in_the_past_are_unchanged():
    assert month_span("Jan 2019", "Dec 2021", TODAY) == 35
    assert total_years([_role("Jan 2019", "Dec 2021")], TODAY) == 2.9


def test_range_entirely_in_the_future_is_not_counted():
    assert month_span("Jan 2030", "Dec 2035", TODAY) is None
    assert total_years([_role("Jan 2030", "Dec 2035")], TODAY) is None
    assert total_years([_role("Nov 2026", "Dec 2026")], TODAY) is None


def test_range_ending_in_the_future_stops_at_today():
    assert month_span("Jan 2026", "Dec 2030", TODAY) == 9
    assert total_years([_role("Oct 2025", "Dec 2030")], TODAY) == 1.0


def test_present_still_means_today():
    assert month_span("Oct 2025", "Present", TODAY) == 12


def test_future_role_does_not_add_to_real_experience():
    roles = [_role("Jan 2020", "Dec 2020"), _role("Jan 2031", "Dec 2040")]
    assert total_years(roles, TODAY) == 0.9


def test_unusable_ranges_are_counted():
    roles = [
        _role("Jan 2030", "Dec 2031"),
        _role("Jan 2020", "Dec 2020"),
        _role("Jun 2021", "Jan 2021"),
    ]
    assert unusable_date_ranges(roles, TODAY) == 2


@pytest.mark.parametrize(
    "line",
    ["Engineer   Jan 1900 - Present", "Engineer   1900 - 2099", "Engineer   01/1900 - 12/2002"],
)
def test_implausible_ranges_do_not_crash_the_parser(line):
    profile = parse_resume("EXPERIENCE\n" + line, today=TODAY)
    assert profile.total_years_experience is None
    assert profile.experience[0].duration_months is None
    assert any("could not be counted" in note for note in profile.extraction_notes)


def test_total_above_eighty_years_is_treated_as_unusable():
    roles = [_role("Jan 1900", "Dec 1950"), _role("Jan 1960", "Present")]
    assert total_years(roles, TODAY) is None


def test_future_dated_role_gives_unverified_experience_in_the_report():
    text = "EXPERIENCE\nAnalyst, Alpha Co   Jan 2030 - Dec 2032\n"
    profile = parse_resume(text, today=TODAY)
    report = evaluate(profile, JobRequirements(min_years_experience=1), text)
    assert report.assessments[0].status == MatchStatus.UNVERIFIED
    assert any("could not be counted" in item for item in report.limitations)


def test_future_dates_do_not_fall_back_to_a_wrong_total():
    text = (
        "EXPERIENCE\n"
        "Analyst, Alpha Co   Jan 2030 - Dec 2035\n"
        "Analyst, Beta Co   Jan 2020 - Jan 2021\n"
    )
    profile = parse_resume(text, today=TODAY)
    assert profile.total_years_experience == 1.0