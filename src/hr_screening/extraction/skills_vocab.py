"""Built-in skill vocabulary, aliases and transferable skill hints.

Single letter or common word skills (R, Go, Excel) are left out on purpose
because keyword matching would produce many false positives.
"""

import re
from functools import lru_cache

SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    "python": (),
    "sql": (),
    "java": (),
    "javascript": ("js",),
    "typescript": (),
    "c++": (),
    "c#": (),
    "scala": (),
    "matlab": (),
    "sas": (),
    "pandas": (),
    "numpy": (),
    "scikit-learn": ("sklearn", "scikit learn"),
    "tensorflow": (),
    "pytorch": (),
    "keras": (),
    "matplotlib": (),
    "machine learning": ("ml",),
    "deep learning": (),
    "natural language processing": ("nlp",),
    "computer vision": (),
    "statistics": (),
    "data analysis": ("data analytics",),
    "data visualization": ("data visualisation",),
    "a/b testing": ("ab testing",),
    "forecasting": (),
    "regression": (),
    "etl": (),
    "data modeling": ("data modelling",),
    "data warehousing": (),
    "spark": ("pyspark", "apache spark"),
    "hadoop": (),
    "airflow": (),
    "dbt": (),
    "kafka": (),
    "tableau": (),
    "power bi": ("powerbi",),
    "looker": (),
    "postgresql": ("postgres",),
    "mysql": (),
    "mongodb": (),
    "snowflake": (),
    "bigquery": (),
    "aws": ("amazon web services",),
    "azure": (),
    "gcp": ("google cloud",),
    "docker": (),
    "kubernetes": ("k8s",),
    "git": (),
    "linux": (),
    "ci/cd": (),
    "rest api": ("rest apis", "restful api"),
    "fastapi": (),
    "flask": (),
    "django": (),
}

# Required skill -> related skills that may be transferable (partial credit only).
TRANSFERABLE: dict[str, tuple[str, ...]] = {
    "tableau": ("power bi", "looker"),
    "power bi": ("tableau", "looker"),
    "looker": ("tableau", "power bi"),
    "mysql": ("postgresql",),
    "postgresql": ("mysql",),
    "sql": ("postgresql", "mysql", "bigquery", "snowflake"),
    "aws": ("azure", "gcp"),
    "azure": ("aws", "gcp"),
    "gcp": ("aws", "azure"),
    "tensorflow": ("pytorch", "keras"),
    "pytorch": ("tensorflow", "keras"),
    "machine learning": ("scikit-learn", "tensorflow", "pytorch", "regression", "deep learning"),
    "deep learning": ("tensorflow", "pytorch", "keras"),
    "data visualization": ("tableau", "power bi", "matplotlib", "looker"),
    "data analysis": ("pandas", "statistics"),
    "spark": ("hadoop",),
    "airflow": ("dbt",),
    "kubernetes": ("docker",),
    "javascript": ("typescript",),
    "typescript": ("javascript",),
}


@lru_cache(maxsize=512)
def skill_pattern(skill: str) -> re.Pattern[str]:
    """Whole word, case insensitive pattern for a skill and its aliases.

    The boundaries stop 'sql' matching inside 'postgresql' and 'java' inside 'javascript'.
    """
    key = skill.strip().lower()
    names = {key, *SKILL_ALIASES.get(key, ())}
    alternatives = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    return re.compile(rf"(?<![A-Za-z0-9+#])(?:{alternatives})(?![A-Za-z0-9+#])", re.IGNORECASE)