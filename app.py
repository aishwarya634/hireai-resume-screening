import os
import json
from pathlib import Path

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from groq import Groq
from flask import Flask, jsonify, send_from_directory, request

# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
INPUTS_DIR = BASE_DIR / "inputs"
PROFILES_DIR = INPUTS_DIR / "profiles"
import tempfile

OUTPUTS_DIR = Path(tempfile.gettempdir()) / "hireai_outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUTS_DIR.mkdir(exist_ok=True)

load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

client = Groq(api_key=api_key)


# ==========================================
# READ INPUT FILES
# ==========================================

def read_inputs():
    job_file = INPUTS_DIR / "job_description.md"

    if not job_file.exists():
        raise FileNotFoundError("inputs/job_description.md was not found.")

    job_description = job_file.read_text(encoding="utf-8").strip()

    if not job_description:
        raise ValueError("The job description is empty.")

    profiles = {}

    for file_path in sorted(PROFILES_DIR.glob("*.md")):
        profile = file_path.read_text(encoding="utf-8").strip()

        if profile:
            profiles[file_path.stem] = profile

    if not profiles:
        raise ValueError("No candidate profiles were found in inputs/profiles/.")

    return job_description, profiles


# ==========================================
# BUILD PROMPT
# ==========================================

def build_prompt(job_description, profiles):
    candidate_text = ""

    for candidate_name, profile in profiles.items():
        candidate_text += f"""
===== {candidate_name} =====
{profile}
"""

    return f"""
You are an expert HR recruitment and talent assessment assistant.

You must evaluate every candidate against the supplied job description.

================ JOB DESCRIPTION ================

{job_description}

================ CANDIDATE PROFILES ================

{candidate_text}

====================================================

IMPORTANT:
- Analyze the actual text provided above.
- Do not claim that the data is missing.
- Do not invent qualifications, experience, skills, education, or certifications.
- Do not reward irrelevant skills.
- Base conclusions only on evidence in the supplied documents.
- Ignore protected or unrelated personal attributes.

Evaluate EVERY candidate on:

1. Technical Skills
2. Experience
3. Education
4. Certifications
5. Domain Knowledge
6. Soft Skills

Give each category a score from 0 to 100.

Also provide:
- candidate name
- overall score
- matched skills
- missing skills
- strengths
- weaknesses
- remarks explaining the score

Then rank all candidates from highest to lowest.

Return ONLY valid JSON using exactly this structure:

{{
    "candidates": [
        {{
            "candidate_name": "candidate1",
            "overall_score": 0,
            "technical_skills_score": 0,
            "experience_score": 0,
            "education_score": 0,
            "certifications_score": 0,
            "domain_knowledge_score": 0,
            "soft_skills_score": 0,
            "matched_skills": [],
            "missing_skills": [],
            "strengths": [],
            "weaknesses": [],
            "remarks": ""
        }}
    ],
    "top_5": [],
    "hiring_recommendation": ""
}}
"""


# ==========================================
# AI SCREENING
# ==========================================

def run_screening():
    job_description, profiles = read_inputs()
    prompt = build_prompt(job_description, profiles)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert HR recruitment assistant. "
                    "Analyze the supplied job description and candidate profiles. "
                    "Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.1,
        max_completion_tokens=2500,
    )

    result = response.choices[0].message.content

    if not result:
        raise ValueError("Groq returned an empty response.")

    try:
        data = json.loads(result)
    except json.JSONDecodeError as exc:
        raise ValueError("Groq did not return valid JSON.") from exc

    candidates = data.get("candidates", [])

    if not candidates:
        raise ValueError("No candidate evaluations were returned by Groq.")

    # ==========================================
    # PYTHON-CALCULATED FINAL SCORE
    # ==========================================

    def calculate_score(candidate):
        return round(
            candidate.get("technical_skills_score", 0) * 0.25
            + candidate.get("experience_score", 0) * 0.20
            + candidate.get("education_score", 0) * 0.10
            + candidate.get("certifications_score", 0) * 0.10
            + candidate.get("domain_knowledge_score", 0) * 0.20
            + candidate.get("soft_skills_score", 0) * 0.15
        )

    for candidate in candidates:
        candidate["calculated_score"] = calculate_score(candidate)

        score = candidate["calculated_score"]

        if score >= 90:
            candidate["recommendation"] = "Strongly Recommended"
        elif score >= 80:
            candidate["recommendation"] = "Recommended"
        elif score >= 70:
            candidate["recommendation"] = "Consider"
        elif score >= 60:
            candidate["recommendation"] = "Weak Match"
        else:
            candidate["recommendation"] = "Not Recommended"

    candidates.sort(
        key=lambda candidate: candidate["calculated_score"],
        reverse=True,
    )

    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank

    top_5 = candidates[:5]

    # ==========================================
    # REPORT
    # ==========================================

    report_file = OUTPUTS_DIR / "report.md"

    with report_file.open("w", encoding="utf-8") as file:
        file.write("# AI HR Resume Screening Report\n\n")
        file.write("## Job Summary\n\n")
        file.write(job_description)
        file.write("\n\n")

        file.write("## Candidate Evaluation\n\n")
        file.write("| Rank | Candidate | Score | Recommendation | Remarks |\n")
        file.write("|---:|---|---:|---|---|\n")

        for candidate in candidates:
            file.write(
                f"| {candidate['rank']} "
                f"| {candidate['candidate_name']} "
                f"| {candidate['calculated_score']}/100 "
                f"| {candidate['recommendation']} "
                f"| {candidate.get('remarks', '')} |\n"
            )

        file.write("\n# Top 5 Candidates\n\n")

        for candidate in top_5:
            file.write(
                f"## #{candidate['rank']} "
                f"{candidate['candidate_name']} — "
                f"{candidate['calculated_score']}/100\n\n"
            )

            file.write("### Strengths\n\n")
            for strength in candidate.get("strengths", []):
                file.write(f"- {strength}\n")

            file.write("\n### Missing Skills\n\n")
            for skill in candidate.get("missing_skills", []):
                file.write(f"- {skill}\n")

            file.write("\n### Weaknesses\n\n")
            for weakness in candidate.get("weaknesses", []):
                file.write(f"- {weakness}\n")

            file.write("\n### Remarks\n\n")
            file.write(candidate.get("remarks", ""))
            file.write("\n\n")

        file.write("# Hiring Recommendation\n\n")
        file.write(
            f"**Recommended Candidate:** {candidates[0]['candidate_name']}\n\n"
        )
        file.write(
            f"**Final Score:** {candidates[0]['calculated_score']}/100\n\n"
        )
        file.write(data.get("hiring_recommendation", ""))
        file.write(
            "\n\n---\n\n"
            "*AI-assisted screening is decision support. "
            "Final hiring decisions should be made by qualified human reviewers.*"
        )

    # ==========================================
    # CHART
    # ==========================================

    names = [candidate["candidate_name"] for candidate in candidates]
    scores = [candidate["calculated_score"] for candidate in candidates]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, scores)
    plt.title("AI Resume Screening — Candidate Scores")
    plt.xlabel("Candidates")
    plt.ylabel("Score / 100")
    plt.ylim(0, 100)

    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            score + 2,
            f"{score}/100",
            ha="center",
            fontweight="bold",
        )

    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    chart_file = OUTPUTS_DIR / "scores.png"
    plt.savefig(chart_file, dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "job_title": extract_job_title(job_description),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "top_5": top_5,
        "hiring_recommendation": data.get("hiring_recommendation", ""),
    }


def extract_job_title(job_description):
    for line in job_description.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()

    return "Resume Screening"


# ==========================================
# WEB ROUTES
# ==========================================

@app.get("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.post("/api/screen")
def screen():
    try:
        results = run_screening()
        return jsonify({
            "success": True,
            "results": results,
        })

    except Exception as exc:
        return jsonify({
            "success": False,
            "error": str(exc),
        }), 500


@app.get("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUTS_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)
