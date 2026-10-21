"""Synthetic cases for the small retrieval check in run_retrieval_eval.py.

Everything here is invented. The three resumes are short and contain no real people. The
cases were written by the AI assistant that also wrote the retrieval code, before any results
were seen, and they were not changed to improve the scores. That makes them optimistic: the
author knew how the code works. About two dozen queries say very little about real resumes.

Each query has:
  type         what the case is meant to test
  kind         requirement kind, as the pipeline uses it
  requirement  the requirement text, as the job parser would produce it
  gold         quotes from the resume. A returned passage is correct if it contains one of
               them. None means the resume has nothing relevant and nothing should be returned.
  mention      (skill queries only) how the skill is mentioned in the gold passage
"""

ANALYST = """SUMMARY
Analyst with five years of experience turning sales and operations data into weekly decisions.

SKILLS
Languages: Python, SQL
Tools: pandas, Git, Excel
Reporting: Power BI

EXPERIENCE
Senior Analyst, Sample Ltd   Jan 2022 - Present
- Built weekly revenue dashboards in Power BI for four regional managers.
- Wrote SQL against PostgreSQL to reconcile billing data and cut the month end close by two days.
- Predicted monthly demand with ARIMA models and reported forecast error to the planning team.
Data Analyst, Example Corp   Jan 2019 - Dec 2021
- Automated a monthly report in Python and pandas, replacing a manual Excel process.
- Ran A/B tests on pricing pages and presented confidence intervals to the product owner.

EDUCATION
B.Sc. in Statistics, Example University, 2018

CERTIFICATIONS
Example Data Analytics Certificate
"""

ENGINEER = """SUMMARY
Backend engineer who also maintains the data pipelines for a small analytics team.

SKILLS
Python3, JS, Docker, PostgreSQL, scikit-learn, Airflow
Currently learning Spark through an online course.

EXPERIENCE
Backend Engineer, Sample Ltd   Mar 2020 - Present
- Maintain REST services in Python3 and deploy them with Docker.
- Moved nightly jobs to Airflow and added alerts for failed runs.
- Has not used Kubernetes because the platform team manages the clusters.
- Trained a churn model with sklearn and handed it to the analytics team.
Junior Developer, Example Corp   Jun 2018 - Feb 2020
- Built internal tools in JS and wrote unit tests for them.

EDUCATION
Bachelor of Technology in Computer Engineering, Example Institute, 2018
"""

MARKETING = """SUMMARY
Marketing analyst who measures campaigns and explains the numbers to non technical teams.

EXPERIENCE
Marketing Analyst, Example Corp   Feb 2021 - Present
- Pulled campaign results from the company data warehouse every week using queries.
- Created interactive charts for the sales leads in Looker Studio.
- Compared two email subject lines with a controlled test and reported which one won.
- Taught colleagues how to read the weekly numbers.

EDUCATION
Master of Business Administration, Example Business School, 2020
Bachelor of Commerce, Example University, 2016

CERTIFICATIONS
Google Analytics Certification, 2022
"""


def query(id, type, kind, requirement, gold, mention=None):
    return {
        "id": id,
        "type": type,
        "kind": kind,
        "requirement": requirement,
        "gold": gold,
        "mention": mention,
    }


RESUMES = [
    {
        "id": "analyst",
        "text": ANALYST,
        "queries": [
            query("analyst-python", "exact", "required_skill", "python",
                  ["Languages: Python, SQL", "Automated a monthly report in Python and pandas"]),
            query("analyst-sql", "exact", "required_skill", "sql",
                  ["Languages: Python, SQL", "Wrote SQL against PostgreSQL"]),
            query("analyst-git", "exact", "required_skill", "git",
                  ["Tools: pandas, Git, Excel"]),
            query("analyst-tableau", "related", "required_skill", "tableau",
                  ["Reporting: Power BI", "Built weekly revenue dashboards in Power BI"]),
            query("analyst-forecasting", "paraphrase", "required_skill", "forecasting",
                  ["Predicted monthly demand with ARIMA models"]),
            query("analyst-docker", "absent", "preferred_skill", "docker", None),
            query("analyst-degree", "text", "education", "Bachelor level education or higher",
                  ["B.Sc. in Statistics, Example University, 2018"]),
            query("analyst-cert", "text", "certification", "Data analytics certification",
                  ["Example Data Analytics Certificate"]),
        ],
    },
    {
        "id": "engineer",
        "text": ENGINEER,
        "queries": [
            query("engineer-python", "version", "required_skill", "python",
                  ["Python3, JS, Docker", "Maintain REST services in Python3"]),
            query("engineer-javascript", "alias", "required_skill", "javascript",
                  ["Python3, JS, Docker", "Built internal tools in JS"]),
            query("engineer-sklearn", "alias", "required_skill", "scikit-learn",
                  ["scikit-learn, Airflow", "Trained a churn model with sklearn"]),
            query("engineer-airflow", "exact", "preferred_skill", "airflow",
                  ["Moved nightly jobs to Airflow", "scikit-learn, Airflow"]),
            query("engineer-kubernetes", "negated", "required_skill", "kubernetes",
                  ["Has not used Kubernetes"], mention="negated"),
            query("engineer-spark", "aspirational", "preferred_skill", "spark",
                  ["Currently learning Spark"], mention="aspirational"),
            query("engineer-snowflake", "absent", "preferred_skill", "snowflake", None),
            query("engineer-degree", "text", "education", "Bachelor level education or higher",
                  ["Bachelor of Technology in Computer Engineering"]),
        ],
    },
    {
        "id": "marketing",
        "text": MARKETING,
        "queries": [
            query("marketing-looker", "exact", "required_skill", "looker",
                  ["Created interactive charts for the sales leads in Looker Studio"]),
            query("marketing-visualization", "paraphrase", "required_skill", "data visualization",
                  ["Created interactive charts for the sales leads in Looker Studio"]),
            query("marketing-abtesting", "paraphrase", "required_skill", "a/b testing",
                  ["Compared two email subject lines with a controlled test"]),
            query("marketing-python", "absent", "required_skill", "python", None),
            query("marketing-degree", "text", "education", "Master level education or higher",
                  ["Master of Business Administration"]),
            query("marketing-cert", "text", "certification", "Google Analytics certification",
                  ["Google Analytics Certification, 2022"]),
        ],
    },
]