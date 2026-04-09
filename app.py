"""
CareerPath AI — Flask application with rule-based + lightweight scoring engine.

The "AI" layer combines:
- Deterministic rules (role requirements, career ladders)
- A simple vector similarity score (cosine-like) between employee skill profile
  and target-role ideal vectors — no external ML libraries required.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

# -----------------------------------------------------------------------------
# Paths & app setup
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "employees.json"
UPLOAD_DIR = BASE_DIR / "uploads"
RESUME_CACHE_DIR = DATA_DIR / "resume_analysis"
# Max upload size (Flask-level); keep in sync with client messaging
MAX_RESUME_BYTES = 2 * 1024 * 1024

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "careerpath-dev-secret-change-in-prod")
app.config["MAX_CONTENT_LENGTH"] = MAX_RESUME_BYTES


# -----------------------------------------------------------------------------
# Static knowledge: roles, industry trends, mock courses & certifications
# -----------------------------------------------------------------------------

# Skills required for common target roles (normalized lowercase keys elsewhere)
ROLE_REQUIREMENTS: dict[str, list[str]] = {
    "software engineer": [
        "python",
        "javascript",
        "git",
        "sql",
        "system design",
        "testing",
        "apis",
    ],
    "senior software engineer": [
        "python",
        "system design",
        "leadership",
        "mentoring",
        "cloud",
        "kubernetes",
        "architecture",
        "ci/cd",
    ],
    "tech lead": [
        "architecture",
        "leadership",
        "mentoring",
        "system design",
        "cloud",
        "stakeholder management",
        "roadmapping",
        "security",
    ],
    "data scientist": [
        "python",
        "statistics",
        "machine learning",
        "sql",
        "visualization",
        "experimentation",
        "pandas",
    ],
    "product manager": [
        "roadmapping",
        "stakeholder management",
        "analytics",
        "ux",
        "prioritization",
        "communication",
    ],
}

# Ordered career steps: current family → next → future (for visualization)
CAREER_LADDERS: dict[str, tuple[str, str, str]] = {
    "software engineer": ("Software Engineer", "Senior Software Engineer", "Tech Lead"),
    "senior software engineer": ("Senior Software Engineer", "Tech Lead", "Engineering Manager"),
    "tech lead": ("Tech Lead", "Engineering Manager", "Director of Engineering"),
    "data scientist": ("Data Scientist", "Senior Data Scientist", "ML Lead"),
    "product manager": ("Product Manager", "Senior PM", "Group PM"),
}

# "Market" skills — weighted by perceived demand (1–10) for scoring
TRENDING_SKILLS: dict[str, float] = {
    "python": 9.5,
    "machine learning": 9.2,
    "cloud": 8.8,
    "kubernetes": 8.5,
    "system design": 8.7,
    "generative ai": 9.0,
    "sql": 8.0,
    "javascript": 8.3,
    "cybersecurity": 8.4,
    "data engineering": 8.6,
    "leadership": 7.9,
    "apis": 7.5,
}

# Mock learning catalog: skill keyword → resources
MOCK_COURSES: dict[str, list[dict[str, str]]] = {
    "default": [
        {"title": "CareerPath Foundations", "provider": "Internal LMS", "hours": "8h"},
        {"title": "Communication for Engineers", "provider": "Coursera (mock)", "hours": "12h"},
    ],
    "python": [
        {"title": "Advanced Python Patterns", "provider": "Internal LMS", "hours": "10h"},
        {"title": "Python for Data & APIs", "provider": "Udemy (mock)", "hours": "14h"},
    ],
    "cloud": [
        {"title": "AWS/GCP Core Services", "provider": "Cloud Guild", "hours": "16h"},
    ],
    "kubernetes": [
        {"title": "K8s for Developers", "provider": "Internal LMS", "hours": "12h"},
    ],
    "system design": [
        {"title": "System Design Interview Prep", "provider": "Educative (mock)", "hours": "20h"},
    ],
    "machine learning": [
        {"title": "ML Fundamentals", "provider": "Stanford Online (mock)", "hours": "24h"},
    ],
    "leadership": [
        {"title": "Leading Technical Teams", "provider": "Internal Leadership", "hours": "8h"},
    ],
}

MOCK_CERTIFICATIONS: list[dict[str, str]] = [
    {"name": "AWS Solutions Architect", "issuer": "Amazon", "relevance": "Cloud-heavy roles"},
    {"name": "Kubernetes CKA", "issuer": "CNCF", "relevance": "Platform / DevOps paths"},
    {"name": "PMP / Agile PM", "issuer": "PMI", "relevance": "Product & program leadership"},
]

MOCK_INTERNAL_OPPORTUNITIES: list[str] = [
    "Join the Architecture Guild monthly review",
    "Mentor a junior engineer in the buddy program",
    "Present a tech talk at the internal summit",
    "Shadow a Tech Lead for one sprint",
]

# Base salary mock anchors (USD, simplified)
ROLE_SALARY_BASE: dict[str, int] = {
    "software engineer": 95000,
    "senior software engineer": 135000,
    "tech lead": 165000,
    "data scientist": 110000,
    "product manager": 105000,
}

BADGE_DEFINITIONS: list[dict[str, Any]] = [
    {"id": "starter", "name": "Profile Complete", "criteria": "Filled career goals and 3+ skills"},
    {"id": "learner", "name": "Learning Path", "criteria": "At least 2 gap skills with courses assigned"},
    {"id": "market_aligned", "name": "Market Aligned", "criteria": "50%+ overlap with trending skills"},
    {"id": "growth_ready", "name": "Growth Ready", "criteria": "Similarity to target role ≥ 0.55"},
]

# Resume analyzer: target-role skills (keyword database for scoring & gaps)
ROLE_SKILLS: dict[str, list[str]] = {
    "software engineer": ["python", "java", "sql", "git", "data structures"],
    "data scientist": ["python", "machine learning", "pandas", "numpy", "statistics"],
    "web developer": ["html", "css", "javascript", "react", "node"],
    "tech lead": ["system design", "leadership", "architecture", "cloud"],
}

# Map profile desired_role_key → ROLE_SKILLS bucket (extends without duplicating lists)
RESUME_ROLE_ALIAS: dict[str, str] = {
    "software engineer": "software engineer",
    "senior software engineer": "software engineer",
    "data scientist": "data scientist",
    "tech lead": "tech lead",
    "product manager": "software engineer",
    "web developer": "web developer",
}

# Missing-skill token → key for COURSES / CERTIFICATIONS lookups
MISSING_SKILL_RESOURCE_KEY: dict[str, str] = {
    "python": "python",
    "java": "java",
    "sql": "sql",
    "git": "git",
    "data structures": "python",
    "machine learning": "machine learning",
    "pandas": "pandas",
    "numpy": "numpy",
    "statistics": "statistics",
    "html": "html",
    "css": "css",
    "javascript": "javascript",
    "react": "react",
    "node": "node",
    "system design": "system design",
    "leadership": "leadership",
    "architecture": "architecture",
    "cloud": "cloud",
}

# Structured recommendations (no external API)
RESUME_COURSES: dict[str, list[str]] = {
    "python": ["Python for Everybody – Coursera", "Python Bootcamp – Udemy"],
    "java": ["Java Programming & Software Engineering – Duke (Coursera)", "Modern Java – Udemy"],
    "sql": ["SQL for Data Science – UC Davis (Coursera)", "Mode SQL Tutorial"],
    "git": ["Git & GitHub Crash Course", "Atlassian Git tutorials"],
    "react": ["React – Meta Certification", "Frontend Dev – Coursera"],
    "javascript": ["The Complete JavaScript Course – Udemy", "JavaScript Algorithms – Udemy"],
    "machine learning": ["Andrew Ng ML Course – Coursera", "Fast.ai Practical Deep Learning"],
    "pandas": ["Data Analysis with Pandas – DataCamp", "Python Data Science Handbook (book)"],
    "numpy": ["NumPy & SciPy – Coursera", "Linear Algebra for ML – Coursera"],
    "statistics": ["Statistics with Python – Coursera", "Think Stats (book)"],
    "html": ["HTML/CSS – freeCodeCamp", "Web.dev Learn HTML"],
    "css": ["Advanced CSS – Udemy", "CSS for JS Developers"],
    "node": ["Node.js – OpenJS", "Backend APIs with Node – Coursera"],
    "system design": ["System Design Interview – Educative", "ByteByteGo System Design"],
    "leadership": ["Leading People & Teams – Michigan (Coursera)", "Harvard Leadership principles"],
    "architecture": ["Software Architecture Patterns", "AWS Well-Architected"],
    "cloud": ["AWS Cloud Practitioner Essentials", "Google Cloud Engineer – Coursera"],
    "default": ["CareerPath Foundations", "Communication for Engineers"],
}

RESUME_CERTIFICATIONS: dict[str, list[str]] = {
    "python": ["PCAP Python Certification", "Google IT Automation with Python"],
    "java": ["Oracle Java SE Certification"],
    "sql": ["Oracle SQL Certified Associate"],
    "git": ["GitLab Certified Associate"],
    "react": ["Meta Front-End Developer Certificate"],
    "javascript": ["OpenJS Node.js Certification"],
    "machine learning": ["TensorFlow Developer Certificate", "AWS ML Specialty (intro)"],
    "pandas": ["DataCamp Data Analyst with Python"],
    "numpy": ["NumPy skill track – DataCamp"],
    "statistics": ["Statistics for Data Science – edX"],
    "html": ["freeCodeCamp Responsive Web Design"],
    "css": ["freeCodeCamp Front End Libraries"],
    "node": ["OpenJS Node.js Application Developer"],
    "system design": ["AWS Solutions Architect Associate"],
    "leadership": ["PMP / Agile PM fundamentals"],
    "architecture": ["AWS Solutions Architect Professional"],
    "cloud": ["AWS Certified Developer", "Google Cloud Engineer"],
    "data": ["Google Data Analytics Certificate"],
    "default": ["Industry certification aligned to your target role"],
}

# Keywords for +5 “experience / projects” bonus
RESUME_PROJECT_KEYWORDS = ("project", "internship", "portfolio", "capstone", "hackathon")

# Keywords suggesting certifications on resume (+5 bonus)
RESUME_CERT_KEYWORDS = (
    "certified",
    "certification",
    "certificate",
    " aws ",
    "pmp",
    "cpa",
    "google cloud",
    "azure",
)


# -----------------------------------------------------------------------------
# Data persistence
# -----------------------------------------------------------------------------


def ensure_data_file() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        seed = sample_employees_seed()
        save_employees(seed)


def load_employees() -> list[dict[str, Any]]:
    ensure_data_file()
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_employees(employees: list[dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(employees, f, indent=2)


def sample_employees_seed() -> list[dict[str, Any]]:
    """Pre-loaded demo users so the app works immediately."""
    return [
        {
            "id": str(uuid.uuid4()),
            "name": "Alex Rivera",
            "role": "Software Engineer",
            "skills": ["Python", "Git", "SQL", "JavaScript"],
            "experience_years": 3,
            "interests": "Backend systems, APIs, open source",
            "career_goals": "Senior Software Engineer within 2 years",
            "desired_role_key": "senior software engineer",
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Jordan Kim",
            "role": "Data Analyst",
            "skills": ["SQL", "Excel", "Visualization", "Python"],
            "experience_years": 2,
            "interests": "Experimentation, storytelling with data",
            "career_goals": "Transition to Data Scientist",
            "desired_role_key": "data scientist",
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Sam Patel",
            "role": "Junior Developer",
            "skills": ["JavaScript", "HTML", "CSS", "React"],
            "experience_years": 1,
            "interests": "Web performance, UX",
            "career_goals": "Become a well-rounded Software Engineer",
            "desired_role_key": "software engineer",
        },
    ]


def normalize_skill(s: str) -> str:
    return s.strip().lower()


def normalize_skills(skills: list[str]) -> set[str]:
    return {normalize_skill(s) for s in skills if s and s.strip()}


# -----------------------------------------------------------------------------
# Skill gap + lightweight "ML" similarity (vector dot / cosine-style)
# -----------------------------------------------------------------------------


def build_skill_vector(skill_set: set[str], vocabulary: list[str]) -> list[float]:
    """Binary vector over a fixed vocabulary (simple bag-of-skills)."""
    return [1.0 if v in skill_set else 0.0 for v in vocabulary]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def merge_vocabulary(*skill_sets: set[str]) -> list[str]:
    merged: set[str] = set()
    for s in skill_sets:
        merged |= s
    return sorted(merged)


def skill_gap_analysis(
    employee_skills: set[str], desired_role_key: str
) -> dict[str, Any]:
    """
    Compare employee skills to required role skills and trending market skills.
    Returns missing skills, strengths, and overlap metrics.
    """
    desired_role_key = desired_role_key.strip().lower()
    required = set(ROLE_REQUIREMENTS.get(desired_role_key, ROLE_REQUIREMENTS["software engineer"]))
    trending_set = set(TRENDING_SKILLS.keys())

    missing = sorted(required - employee_skills)
    # Strengths: skills the employee has that are either required or high-demand
    strengths = sorted(
        (employee_skills & required) | (employee_skills & trending_set)
    )

    # Vocabulary for similarity: union of required + employee + trending slice
    vocab = merge_vocabulary(required, employee_skills, trending_set)
    v_emp = build_skill_vector(employee_skills, vocab)
    # Ideal target vector: required skills for role (binary)
    v_target = build_skill_vector(required, vocab)

    role_fit_score = cosine_similarity(v_emp, v_target)

    # Market alignment: weighted overlap with trending
    market_score = 0.0
    if employee_skills:
        weights = [TRENDING_SKILLS.get(s, 5.0) for s in employee_skills if s in trending_set]
        market_score = min(1.0, (sum(weights) / (len(trending_set) * 10.0)) * 2.0)

    return {
        "missing_skills": missing,
        "strength_areas": strengths,
        "role_fit_score": round(role_fit_score, 3),
        "market_alignment_hint": round(market_score, 3),
        "required_for_role": sorted(required),
    }


def estimate_future_salary(
    desired_role_key: str, experience_years: int, role_fit_score: float
) -> dict[str, Any]:
    """Mock salary model: base by role + experience bump + fit bonus."""
    key = desired_role_key.strip().lower()
    base = ROLE_SALARY_BASE.get(key, 90000)
    exp_bonus = min(25000, experience_years * 2500)
    fit_bonus = int(role_fit_score * 15000)
    estimated = base + exp_bonus + fit_bonus
    return {
        "estimated_annual_usd": estimated,
        "breakdown": {
            "role_base": base,
            "experience_bonus": exp_bonus,
            "readiness_bonus": fit_bonus,
        },
        "disclaimer": "Illustrative estimate only — not a guarantee or offer.",
    }


def build_learning_path(missing_skills: list[str]) -> list[dict[str, Any]]:
    """Attach mock courses to each missing skill (deduplicated)."""
    path: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for skill in missing_skills[:8]:
        key = skill if skill in MOCK_COURSES else "default"
        courses = MOCK_COURSES.get(skill, MOCK_COURSES[key])
        for c in courses:
            if c["title"] not in seen_titles:
                seen_titles.add(c["title"])
                path.append(
                    {
                        "focus_skill": skill,
                        "title": c["title"],
                        "provider": c["provider"],
                        "hours": c["hours"],
                    }
                )
    if not path:
        for c in MOCK_COURSES["default"]:
            path.append(
                {
                    "focus_skill": "general",
                    "title": c["title"],
                    "provider": c["provider"],
                    "hours": c["hours"],
                }
            )
    return path[:12]


def resume_tips(gaps: list[str], strengths: list[str]) -> list[str]:
    """Heuristic resume bullets tailored to gaps/strengths."""
    tips = [
        "Quantify impact: use metrics (latency, revenue, adoption) next to each project.",
        "Align your headline and summary with your stated target role keywords.",
    ]
    if gaps:
        tips.append(
            f"Add a 'Skills in development' line for: {', '.join(gaps[:3])} — shows growth mindset."
        )
    if strengths:
        tips.append(
            f"Feature proof points for top strengths first: {', '.join(strengths[:4])}."
        )
    tips.append("Keep to one page unless 10+ years with multiple major projects.")
    return tips


def compute_skill_progress(employee_skills: set[str], required: list[str]) -> dict[str, Any]:
    """Progress bars: % coverage of required skills + arbitrary 'mastery' for demo."""
    if not required:
        pct = 100
    else:
        have = len(set(required) & employee_skills)
        pct = int(round(100 * have / len(required)))
    per_skill = []
    for s in required:
        # Demo: if they have the skill, show high progress; else low random-stable from hash
        has = normalize_skill(s) in employee_skills
        progress = 92 if has else max(15, (hash(s) % 40) + 10)
        per_skill.append({"skill": s, "progress": progress, "has_skill": has})
    return {"overall_percent": pct, "per_skill": per_skill}


def award_badges(
    employee: dict[str, Any], gap_result: dict[str, Any], learning_path: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    earned = []
    skills = normalize_skills(employee.get("skills", []))
    goals = (employee.get("career_goals") or "").strip()

    if len(skills) >= 3 and len(goals) > 10:
        earned.append(BADGE_DEFINITIONS[0])
    if len(learning_path) >= 2:
        earned.append(BADGE_DEFINITIONS[1])
    if gap_result["market_alignment_hint"] >= 0.5:
        earned.append(BADGE_DEFINITIONS[2])
    if gap_result["role_fit_score"] >= 0.55:
        earned.append(BADGE_DEFINITIONS[3])
    # Deduplicate by badge id
    seen_ids: set[str] = set()
    unique = []
    for b in earned:
        if b["id"] not in seen_ids:
            seen_ids.add(b["id"])
            unique.append(b)
    return unique


def top_skills_market(limit: int = 8) -> list[dict[str, Any]]:
    ranked = sorted(TRENDING_SKILLS.items(), key=lambda x: -x[1])
    return [{"skill": k, "demand_score": v} for k, v in ranked[:limit]]


# -----------------------------------------------------------------------------
# Career opportunities: JSearch (RapidAPI) + LinkedIn + mock fallback
# -----------------------------------------------------------------------------

# Realistic listings used when API is unavailable or returns nothing.
# apply_url built at runtime so each row stays unique (LinkedIn search deep-link).
MOCK_JOBS_BY_ROLE: dict[str, list[dict[str, str]]] = {
    "software engineer": [
        {"title": "Software Engineer II — Core Services", "company": "Stripe", "location": "San Francisco, CA"},
        {"title": "Backend Engineer (Python)", "company": "Datadog", "location": "New York, NY"},
        {"title": "Full Stack Engineer", "company": "Notion", "location": "Remote — US"},
        {"title": "Software Engineer — APIs", "company": "Twilio", "location": "Denver, CO"},
        {"title": "Engineer — Developer Platform", "company": "GitHub", "location": "Remote — Global"},
        {"title": "Application Engineer", "company": "Figma", "location": "San Francisco, CA"},
    ],
    "senior software engineer": [
        {"title": "Senior Software Engineer — Infra", "company": "Snowflake", "location": "Bellevue, WA"},
        {"title": "Senior Backend Engineer", "company": "MongoDB", "location": "New York, NY"},
        {"title": "Senior Software Engineer", "company": "Cloudflare", "location": "Austin, TX"},
        {"title": "Staff Software Engineer", "company": "Databricks", "location": "San Francisco, CA"},
        {"title": "Senior Engineer — Platform", "company": "Elastic", "location": "Remote — US"},
    ],
    "tech lead": [
        {"title": "Engineering Lead — Product", "company": "Atlassian", "location": "Sydney / Remote"},
        {"title": "Tech Lead — Platform", "company": "Shopify", "location": "Toronto, ON"},
        {"title": "Principal Engineer", "company": "HashiCorp", "location": "Remote — US"},
        {"title": "Team Lead — Backend", "company": "Okta", "location": "San Jose, CA"},
        {"title": "Technical Lead", "company": "Confluent", "location": "Mountain View, CA"},
    ],
    "data scientist": [
        {"title": "Data Scientist — Growth", "company": "Spotify", "location": "Stockholm / Remote"},
        {"title": "Senior Data Scientist", "company": "Airbnb", "location": "San Francisco, CA"},
        {"title": "Applied Scientist", "company": "Amazon", "location": "Seattle, WA"},
        {"title": "Data Scientist — ML", "company": "Uber", "location": "Sunnyvale, CA"},
        {"title": "Quantitative Analyst", "company": "Two Sigma", "location": "New York, NY"},
        {"title": "Research Scientist", "company": "DeepMind", "location": "London, UK"},
    ],
    "product manager": [
        {"title": "Product Manager — B2B", "company": "Slack", "location": "San Francisco, CA"},
        {"title": "Senior PM — Platform", "company": "Zoom", "location": "San Jose, CA"},
        {"title": "Group Product Manager", "company": "Adobe", "location": "Remote — US"},
        {"title": "Technical PM", "company": "Microsoft", "location": "Redmond, WA"},
        {"title": "Product Manager", "company": "Asana", "location": "Vancouver, BC"},
    ],
}

# Cross-role variety so employees never see only one sector.
MOCK_JOBS_EXTRA: list[dict[str, str]] = [
    {"title": "Developer Advocate", "company": "Auth0", "location": "Remote — EU/US"},
    {"title": "Security Engineer", "company": "CrowdStrike", "location": "Irvine, CA"},
    {"title": "Site Reliability Engineer", "company": "Netflix", "location": "Los Gatos, CA"},
    {"title": "ML Engineer", "company": "Hugging Face", "location": "Paris / Remote"},
    {"title": "Solutions Architect", "company": "AWS", "location": "Arlington, VA"},
]


def linkedin_jobs_search_url(keywords: str) -> str:
    """Public LinkedIn job search URL (keywords query param)."""
    return "https://www.linkedin.com/jobs/search/?keywords=" + quote(keywords.strip(), safe="")


def _jsearch_location(item: dict[str, Any]) -> str:
    parts = []
    for key in ("job_city", "job_state", "job_country"):
        v = item.get(key)
        if v:
            parts.append(str(v))
    if parts:
        return ", ".join(parts)
    return str(item.get("job_is_remote") and "Remote" or "Location on listing")


def _fetch_jobs_jsearch(query: str, limit: int = 12) -> list[dict[str, Any]] | None:
    """
    Optional RapidAPI JSearch. Set RAPIDAPI_KEY in the environment to enable live data.
    Returns None on missing key, HTTP errors, or empty results.
    """
    api_key = os.environ.get("RAPIDAPI_KEY", "").strip()
    if not api_key:
        return None
    params = urlencode(
        {
            "query": query[:280],
            "page": "1",
            "num_pages": "1",
        }
    )
    url = f"https://jsearch.p.rapidapi.com/search?{params}"
    req = Request(
        url,
        headers={
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
        },
        method="GET",
    )
    try:
        with urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None

    rows = data.get("data")
    if not isinstance(rows, list) or not rows:
        return None

    out: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        title = item.get("job_title") or item.get("title") or "Open role"
        company = item.get("employer_name") or item.get("company_name") or "Company"
        loc = _jsearch_location(item)
        apply_link = (
            item.get("job_apply_link")
            or item.get("job_google_link")
            or linkedin_jobs_search_url(f"{title} {company}")
        )
        out.append(
            {
                "title": str(title)[:200],
                "company": str(company)[:120],
                "location": loc[:160],
                "apply_url": str(apply_link)[:2000],
                "source": "jsearch",
            }
        )
        if len(out) >= limit:
            break
    return out or None


def _apply_link_for_mock_row(title: str, company: str, skill_hint: str) -> str:
    """Real external URL: LinkedIn job search with role + company + skill context."""
    q = f"{title} {company} {skill_hint}".strip()
    return linkedin_jobs_search_url(q[:200])


def mock_jobs_for_employee(employee: dict[str, Any], limit: int = 10) -> list[dict[str, Any]]:
    """
    Deterministic, non-repetitive listings per employee: shuffle driven by profile id + role.
    Always returns at least five distinct rows when limit >= 5.
    """
    role_key = (employee.get("desired_role_key") or "software engineer").strip().lower()
    skills = employee.get("skills") or []
    skill_hint = (skills[0] if skills else role_key.replace("_", " "))[:40]

    pool: list[dict[str, str]] = []
    pool.extend(MOCK_JOBS_BY_ROLE.get(role_key, MOCK_JOBS_BY_ROLE["software engineer"]))
    pool.extend(MOCK_JOBS_EXTRA)

    # Dedupe by (title, company)
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, str]] = []
    for row in pool:
        k = (row["title"], row["company"])
        if k not in seen:
            seen.add(k)
            unique.append(row)

    seed = int(hashlib.sha256(employee["id"].encode()).hexdigest(), 16)
    rng = random.Random(seed)
    rng.shuffle(unique)

    jobs: list[dict[str, Any]] = []
    for row in unique[:limit]:
        jobs.append(
            {
                "title": row["title"],
                "company": row["company"],
                "location": row["location"],
                "apply_url": _apply_link_for_mock_row(row["title"], row["company"], skill_hint),
                "source": "mock",
            }
        )
    return jobs


def get_career_opportunities(employee: dict[str, Any]) -> dict[str, Any]:
    """
    Live jobs when RAPIDAPI_KEY + JSearch work; otherwise realistic mock data.
    Query blends desired role + top skills for relevance.
    """
    desired = (employee.get("desired_role_key") or "software engineer").strip().lower()
    skills = employee.get("skills") or []
    skill_part = " ".join(str(s).strip() for s in skills[:4] if s)
    query = f"{desired.replace('_', ' ')} {skill_part}".strip()

    jobs: list[dict[str, Any]] | None = _fetch_jobs_jsearch(query, limit=12)
    source = "live"

    if not jobs:
        jobs = mock_jobs_for_employee(employee, limit=10)
        source = "mock"

    # Guarantee minimum five listings for demo stability
    if len(jobs) < 5:
        extra = mock_jobs_for_employee(employee, limit=10)
        seen_urls = {j["apply_url"] for j in jobs}
        for j in extra:
            if j["apply_url"] not in seen_urls:
                jobs.append(j)
                seen_urls.add(j["apply_url"])
            if len(jobs) >= 8:
                break

    role_kw = desired.replace("_", " ").title()
    linkedin_url = linkedin_jobs_search_url(role_kw + (" " + skill_part if skill_part else ""))

    return {
        "jobs": jobs[:15],
        "source": source,
        "linkedin_jobs_url": linkedin_url,
        "search_query": query,
    }


def normalize_resume_text(raw: str) -> str:
    """Lowercase, strip symbols that break matching, collapse whitespace."""
    s = raw.lower()
    s = re.sub(r"[^\w\s+/#.\-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return f" {s} "


def skill_mentioned_in_resume(skill: str, text_norm: str) -> bool:
    """Keyword match for a canonical skill phrase (handles ci/cd, etc.)."""
    s = skill.lower().strip()
    if not s:
        return False
    if s in text_norm:
        return True
    if s == "ci/cd":
        return "cicd" in text_norm.replace("/", "") or "ci cd" in text_norm
    if s == "apis":
        return " api " in text_norm or "apis" in text_norm.replace(" ", "")
    return False


def resolve_resume_role_bucket(desired_key: str) -> str:
    """Map profile desired_role_key → ROLE_SKILLS bucket."""
    k = desired_key.strip().lower().replace("_", " ")
    if k in RESUME_ROLE_ALIAS:
        return RESUME_ROLE_ALIAS[k]
    if k in ROLE_SKILLS:
        return k
    return "software engineer"


def resume_has_project_signals(text_norm: str) -> bool:
    return any(kw in text_norm for kw in RESUME_PROJECT_KEYWORDS)


def resume_has_cert_signals(text_norm: str) -> bool:
    return any(kw in text_norm for kw in RESUME_CERT_KEYWORDS)


def extract_text_from_pdf(path: str | Path) -> str:
    """Extract text from a PDF file path; return lowercase for matching."""
    from PyPDF2 import PdfReader

    text = ""
    reader = PdfReader(str(path))
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.lower()


def recommend_resume_courses_and_certs(
    missing: list[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Build deduplicated course and certification lists (≥3 items each when possible)."""
    seen_c: set[str] = set()
    seen_cert: set[str] = set()
    courses_out: list[dict[str, str]] = []
    certs_out: list[dict[str, str]] = []

    def add_courses_for_skill(sk: str) -> None:
        key = MISSING_SKILL_RESOURCE_KEY.get(sk, sk)
        if key not in RESUME_COURSES:
            key = "default"
        for title in RESUME_COURSES.get(key, RESUME_COURSES["default"]):
            if title not in seen_c:
                seen_c.add(title)
                courses_out.append({"title": title, "skill": sk})

    def add_certs_for_skill(sk: str) -> None:
        key = MISSING_SKILL_RESOURCE_KEY.get(sk, sk)
        if key not in RESUME_CERTIFICATIONS:
            key = "default"
        for name in RESUME_CERTIFICATIONS.get(key, RESUME_CERTIFICATIONS["default"]):
            if name not in seen_cert:
                seen_cert.add(name)
                certs_out.append({"name": name, "skill": sk})

    for sk in missing:
        add_courses_for_skill(sk)
        add_certs_for_skill(sk)
        if len(courses_out) >= 8 and len(certs_out) >= 8:
            break

    if len(courses_out) < 3:
        for title in RESUME_COURSES["default"]:
            if title not in seen_c:
                seen_c.add(title)
                courses_out.append({"title": title, "skill": "general"})
            if len(courses_out) >= 5:
                break
        for key in ("python", "cloud", "machine learning"):
            for title in RESUME_COURSES.get(key, []):
                if title not in seen_c:
                    seen_c.add(title)
                    courses_out.append({"title": title, "skill": key})
                if len(courses_out) >= 5:
                    break
            if len(courses_out) >= 5:
                break

    if len(certs_out) < 3:
        for name in RESUME_CERTIFICATIONS.get("default", []):
            if name not in seen_cert:
                seen_cert.add(name)
                certs_out.append({"name": name, "skill": "general"})
            if len(certs_out) >= 5:
                break
        for key in ("python", "cloud", "data"):
            for name in RESUME_CERTIFICATIONS.get(key, []):
                if name not in seen_cert:
                    seen_cert.add(name)
                    certs_out.append({"name": name, "skill": key})
                if len(certs_out) >= 5:
                    break
            if len(certs_out) >= 5:
                break

    return courses_out[:8], certs_out[:8]


def build_resume_smart_suggestions(
    missing: list[str],
    has_project: bool,
    has_cert: bool,
) -> list[str]:
    suggestions: list[str] = []
    for sk in missing[:8]:
        suggestions.append(
            f"Learn {sk} to match industry requirements for your target role."
        )
    if not has_project:
        suggestions.append("Add 2 real-world projects with measurable outcomes.")
    if not has_cert:
        suggestions.append("Include at least 1 certification or formal credential.")
    # Dedupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for line in suggestions:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out[:12]


def analyze_resume(text: str, employee: dict[str, Any]) -> dict[str, Any]:
    """
    Score resume vs ROLE_SKILLS for the mapped target bucket:
    +10 per matched skill (scaled to 90 max), +5 project/internship signals, +5 certification signals; max 100.
    """
    from datetime import datetime, timezone

    desired = (employee.get("desired_role_key") or "software engineer").strip().lower()
    bucket = resolve_resume_role_bucket(desired)
    required = ROLE_SKILLS.get(bucket, ROLE_SKILLS["software engineer"])

    text_norm = normalize_resume_text(text)
    if len(text_norm.strip()) < 20:
        rec_c, rec_cert = recommend_resume_courses_and_certs(required[:5])
        return {
            "score": 0,
            "missing_skills": required[:8],
            "strengths": [],
            "suggestions": [
                "Upload a resume with more readable text (try exporting PDF as text or use .txt).",
                "Ensure the file is not image-only; use selectable text in your PDF.",
            ],
            "recommended_courses": rec_c,
            "recommended_certifications": rec_cert,
            "breakdown": {
                "matched_skills": 0,
                "required_skills": len(required),
                "skill_coverage_pct": 0,
                "project_bonus": 0,
                "cert_bonus": 0,
            },
            "target_role": desired,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "error": "insufficient_text",
        }

    matched = [s for s in required if skill_mentioned_in_resume(s, text_norm)]
    missing = [s for s in required if not skill_mentioned_in_resume(s, text_norm)]
    strengths = list(matched)

    project_bonus = 5 if resume_has_project_signals(text_norm) else 0
    cert_bonus = 5 if resume_has_cert_signals(text_norm) else 0

    if required:
        ratio = len(matched) / len(required)
        skill_coverage_pct = int(round(ratio * 100))
        score = int(round(ratio * 90 + project_bonus + cert_bonus))
        score = min(100, max(0, score))
    else:
        skill_coverage_pct = 0
        score = min(100, project_bonus + cert_bonus)

    suggestions = build_resume_smart_suggestions(missing, project_bonus > 0, cert_bonus > 0)
    rec_courses, rec_certs = recommend_resume_courses_and_certs(missing)

    return {
        "score": score,
        "missing_skills": missing,
        "strengths": strengths,
        "suggestions": suggestions,
        "recommended_courses": rec_courses,
        "recommended_certifications": rec_certs,
        "breakdown": {
            "matched_skills": len(matched),
            "required_skills": len(required),
            "skill_coverage_pct": skill_coverage_pct,
            "project_bonus": project_bonus,
            "cert_bonus": cert_bonus,
        },
        "target_role": desired,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "error": None,
    }


def save_resume_analysis(employee_id: str, data: dict[str, Any]) -> None:
    RESUME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = RESUME_CACHE_DIR / f"{employee_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_resume_analysis(employee_id: str) -> dict[str, Any] | None:
    path = RESUME_CACHE_DIR / f"{employee_id}.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def full_dashboard_payload(employee: dict[str, Any]) -> dict[str, Any]:
    """Assemble everything the dashboard template needs."""
    desired = employee.get("desired_role_key") or "software engineer"
    skills = normalize_skills(employee.get("skills", []))

    gap = skill_gap_analysis(skills, desired)
    salary = estimate_future_salary(desired, int(employee.get("experience_years", 0)), gap["role_fit_score"])
    learning = build_learning_path(gap["missing_skills"])
    tips = resume_tips(gap["missing_skills"], gap["strength_areas"])
    progress = compute_skill_progress(skills, gap["required_for_role"])
    badges = award_badges(employee, gap, learning)

    ladder = CAREER_LADDERS.get(desired, CAREER_LADDERS["software engineer"])

    career_opportunities = get_career_opportunities(employee)
    resume_analysis = load_resume_analysis(employee["id"])

    return {
        "employee": employee,
        "gap": gap,
        "salary": salary,
        "learning_path": learning,
        "certifications": MOCK_CERTIFICATIONS,
        "internal_ops": MOCK_INTERNAL_OPPORTUNITIES,
        "resume_tips": tips,
        "progress": progress,
        "badges": badges,
        "career_ladder": ladder,
        "top_market_skills": top_skills_market(),
        "chart_missing": gap["missing_skills"][:10],
        "chart_strengths": gap["strength_areas"][:10],
        "career_opportunities": career_opportunities,
        "resume_analysis": resume_analysis,
    }


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@app.route("/", methods=["GET", "POST"])
def index():
    employees = load_employees()
    error = None

    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        role = (request.form.get("role") or "").strip()
        skills_raw = request.form.get("skills") or ""
        skills = [s.strip() for s in skills_raw.replace(",", "\n").split("\n") if s.strip()]
        try:
            experience_years = int(request.form.get("experience_years") or 0)
        except ValueError:
            experience_years = 0
        interests = (request.form.get("interests") or "").strip()
        career_goals = (request.form.get("career_goals") or "").strip()
        desired_role_key = (request.form.get("desired_role_key") or "software engineer").strip().lower()

        if not name or not skills:
            error = "Name and at least one skill are required."
        else:
            new_emp = {
                "id": str(uuid.uuid4()),
                "name": name,
                "role": role or "Employee",
                "skills": skills,
                "experience_years": max(0, experience_years),
                "interests": interests,
                "career_goals": career_goals,
                "desired_role_key": desired_role_key,
            }
            employees.append(new_emp)
            save_employees(employees)
            return redirect(url_for("dashboard", employee_id=new_emp["id"]))

    role_choices = sorted(ROLE_REQUIREMENTS.keys())
    return render_template(
        "index.html",
        employees=employees,
        role_choices=role_choices,
        error=error,
    )


@app.route("/dashboard/<employee_id>")
def dashboard(employee_id: str):
    employees = load_employees()
    employee = next((e for e in employees if e["id"] == employee_id), None)
    if not employee:
        return redirect(url_for("index"))

    payload = full_dashboard_payload(employee)
    return render_template("dashboard.html", **payload, employees=employees)


@app.route("/jobs/<employee_id>")
def jobs_page(employee_id: str):
    """Career opportunities: job cards + LinkedIn search (live API or mock fallback)."""
    employees = load_employees()
    employee = next((e for e in employees if e["id"] == employee_id), None)
    if not employee:
        return redirect(url_for("index"))
    opp = get_career_opportunities(employee)
    return render_template(
        "jobs.html",
        employee=employee,
        employees=employees,
        career_opportunities=opp,
    )


@app.route("/api/jobs/<employee_id>")
def api_jobs(employee_id: str):
    """JSON mirror of career opportunities for integrations."""
    employees = load_employees()
    employee = next((e for e in employees if e["id"] == employee_id), None)
    if not employee:
        return jsonify({"error": "not found"}), 404
    return jsonify(get_career_opportunities(employee))


@app.route("/api/analyze/<employee_id>")
def api_analyze(employee_id: str):
    """Optional JSON API for the same analysis (extensibility)."""
    employees = load_employees()
    employee = next((e for e in employees if e["id"] == employee_id), None)
    if not employee:
        return {"error": "not found"}, 404
    return full_dashboard_payload(employee)


@app.route("/upload_resume/<employee_id>", methods=["POST"])
def upload_resume(employee_id: str):
    """Accept PDF/TXT resume, analyze, persist results, redirect to results page."""
    employees = load_employees()
    employee = next((e for e in employees if e["id"] == employee_id), None)
    if not employee:
        flash("Profile not found.", "error")
        return redirect(url_for("index"))

    file = request.files.get("resume")
    if not file or file.filename is None or str(file.filename).strip() == "":
        flash("No file uploaded. Please choose a PDF or TXT resume.", "error")
        return redirect(url_for("dashboard", employee_id=employee_id) + "#resume-analyzer")

    filename = secure_filename(file.filename)
    if not filename:
        flash("Invalid file name.", "error")
        return redirect(url_for("dashboard", employee_id=employee_id) + "#resume-analyzer")

    ext = Path(filename).suffix.lower()
    if ext not in (".pdf", ".txt"):
        flash("Only PDF or TXT files are allowed.", "error")
        return redirect(url_for("dashboard", employee_id=employee_id) + "#resume-analyzer")

    os.makedirs(str(UPLOAD_DIR), exist_ok=True)
    unique = f"{employee_id}_{uuid.uuid4().hex[:8]}_{filename}"
    filepath = os.path.join(str(UPLOAD_DIR), unique)

    try:
        file.save(filepath)
        if os.path.getsize(filepath) > MAX_RESUME_BYTES:
            flash("File too large (max 2 MB).", "error")
            return redirect(url_for("dashboard", employee_id=employee_id) + "#resume-analyzer")

        if ext == ".txt":
            with open(filepath, encoding="utf-8", errors="replace") as f:
                text = f.read()
        else:
            try:
                text = extract_text_from_pdf(filepath)
            except Exception:
                flash(
                    "Could not read this PDF (it may be scanned or encrypted). Try a text-based PDF or save as .txt.",
                    "error",
                )
                return redirect(url_for("dashboard", employee_id=employee_id) + "#resume-analyzer")

        if not text or len(text.strip()) < 10:
            flash("Could not extract enough text from this file.", "error")
            return redirect(url_for("dashboard", employee_id=employee_id) + "#resume-analyzer")

        result = analyze_resume(text, employee)
        if result.get("error") == "insufficient_text":
            flash("Not enough readable text in this file. Try a text-based PDF or a .txt resume.", "error")
            return redirect(url_for("dashboard", employee_id=employee_id) + "#resume-analyzer")
        save_resume_analysis(employee_id, result)
        flash("Resume analyzed successfully.", "success")
        return redirect(url_for("resume_page", employee_id=employee_id))
    finally:
        try:
            if os.path.isfile(filepath):
                os.remove(filepath)
        except OSError:
            pass


@app.route("/resume/<employee_id>")
def resume_page(employee_id: str):
    """Resume Analyzer results view."""
    employees = load_employees()
    employee = next((e for e in employees if e["id"] == employee_id), None)
    if not employee:
        return redirect(url_for("index"))
    analysis = load_resume_analysis(employee_id)
    if not analysis:
        flash("Upload a resume first to see your analysis.", "error")
        return redirect(url_for("dashboard", employee_id=employee_id) + "#resume-analyzer")
    return render_template(
        "resume.html",
        employee=employee,
        employees=employees,
        analysis=analysis,
    )


@app.route("/api/resume/<employee_id>")
def api_resume(employee_id: str):
    """JSON: last saved resume analysis for employee."""
    employees = load_employees()
    employee = next((e for e in employees if e["id"] == employee_id), None)
    if not employee:
        return jsonify({"error": "not found"}), 404
    data = load_resume_analysis(employee_id)
    if not data:
        return jsonify({"error": "no analysis yet"}), 404
    return jsonify(data)


@app.errorhandler(413)
def handle_file_too_large(_e):
    flash("Resume file too large (max 2 MB).", "error")
    ref = request.referrer
    if ref and "dashboard" in ref:
        return redirect(ref)
    return redirect(url_for("index"))


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    ensure_data_file()
    app.run(debug=True, host="127.0.0.1", port=5000)
