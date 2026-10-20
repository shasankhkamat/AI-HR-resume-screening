"""Create a synthetic DOCX resume for demos. It contains no real personal data.

The file is only meant for this project's loader and may not open in Word.
Output goes to data/incoming/, which is git ignored.
"""

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

LINES = [
    "SKILLS",
    "Python, SQL, Pandas, scikit-learn, Power BI",
    "EXPERIENCE",
    "Data Analyst, Example Corp   Jan 2019 - Dec 2021",
    "Built weekly reporting dashboards in Power BI",
    "Senior Analyst, Sample Ltd   Jan 2022 - Present",
    "EDUCATION",
    "B.Tech in Computer Science, Example Institute, 2018",
    "CERTIFICATIONS",
    "Example Cloud Practitioner",
    "PROJECTS",
    "Churn prediction model using scikit-learn",
]
NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def main() -> None:
    target = Path("data/incoming/sample_resume.docx")
    target.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"<w:p><w:r><w:t>{escape(line)}</w:t></w:r></w:p>" for line in LINES)
    document = f'<w:document xmlns:w="{NAMESPACE}"><w:body>{body}</w:body></w:document>'
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("word/document.xml", document)
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()