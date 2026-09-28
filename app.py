import os
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from groq import Groq
from flask import Flask, jsonify, send_from_directory


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


# Vercel's project filesystem is read-only.
# Use /tmp on Vercel.
#
# Locally, continue using the normal outputs folder.
if os.getenv("VERCEL"):
    OUTPUTS_DIR = Path(tempfile.gettempdir()) / "hireai_outputs"
else:
    OUTPUTS_DIR = BASE_DIR / "outputs"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# ENVIRONMENT VARIABLES
# ==========================================

load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GROQ_API_KEY")

# Use the cheaper 20B model by default.
model_name = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. "
        "Add it to your .env file or Vercel environment variables."
    )

client = Groq(api_key=api_key)


# ==========================================
# READ INPUT FILES
# ==========================================

def read_inputs():

    job_file = INPUTS_DIR / "job_description.md"

    if not job_file.exists():
        raise FileNotFoundError(
            "inputs/job_description.md was not found."
        )

    job_description = job_file.read_text(
        encoding="utf-8"
    ).strip()

    if not job_description:
        raise ValueError(
            "The job description is empty."
        )

    profiles = {}

    if not PROFILES_DIR.exists():
        raise FileNotFoundError(
            "inputs/profiles/ directory was not found."
        )

    for file_path in sorted(
        PROFILES_DIR.glob("*.md")
    ):

        profile = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if profile:
            profiles[file_path.stem] = profile

    if not profiles:
        raise ValueError(
            "No candidate profiles were found in "
            "inputs/profiles/."
        )

    print(
        f"Loaded job description: "
        f"{len(job_description)} characters"
    )

    print(
        f"Loaded {len(profiles)} candidate profiles"
    )

    return job_description, profiles


# ==========================================
# BUILD PROMPT
# ==========================================

def build_prompt(job_description, profiles):

    candidate_text = ""

    for candidate_name, profile in profiles.items():

        candidate_text += (
            f"\n===== {candidate_name} =====\n"
            f"{profile}\n"
        )

    return f"""
You are an expert HR recruitment assistant.

Evaluate every candidate against the job description.

Use ONLY information contained in the supplied documents.
Do not invent experience, skills, education, certifications,
projects, or qualifications.

Do not use protected or unrelated personal attributes.

JOB DESCRIPTION
================
{job_description}

CANDIDATE PROFILES
==================
{candidate_text}

Evaluate every candidate on these six dimensions:

1. Technical Skills
2. Experience
3. Education
4. Certifications
5. Domain Knowledge
6. Soft Skills

Give each dimension a score from 0 to 100.

Also provide:

- matched_skills: short list
- missing_skills: short list
- strengths: maximum 3 items
- weaknesses: maximum 3 items
- remarks: maximum 2 short sentences

IMPORTANT:

Python will calculate the final overall score.
Do NOT calculate an overall score.
Do NOT calculate ranking.
Do NOT generate a Top 5 list.

Return ONLY one valid JSON object.

Use EXACTLY this structure:

{{
  "candidates": [
    {{
      "candidate_name": "candidate1",
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
  ]
}}

Every candidate must appear exactly once.

Do not include Markdown.
Do not include ```json.
Do not include explanations outside the JSON object.
"""


# ==========================================
# SAFE NUMBER CONVERSION
# ==========================================

def safe_score(value):

    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0

    value = max(0, min(100, value))

    return value


# ==========================================
# AI SCREENING
# ==========================================

def run_screening():

    job_description, profiles = read_inputs()

    prompt = build_prompt(
        job_description,
        profiles
    )

    print("\nSending candidates to Groq...")
    print(f"Model: {model_name}")

    # ==========================================
    # GROQ REQUEST
    # ==========================================

    response = client.chat.completions.create(

        model=model_name,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert HR recruitment "
                    "assistant. "
                    "Evaluate candidates strictly from "
                    "the supplied documents. "
                    "Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],

        temperature=0.1,

        # Keep output compact to reduce token usage.
        max_completion_tokens=1800,

        # Force valid JSON.
        response_format={
            "type": "json_object"
        },

        # GPT-OSS supports low reasoning effort.
        reasoning_effort="low",

        # Do not return reasoning content.
        include_reasoning=False,
    )


    # ==========================================
    # GET MODEL RESPONSE
    # ==========================================

    result = response.choices[0].message.content

    if not result:

        raise ValueError(
            "Groq returned an empty response."
        )

    print("\n===== GROQ RESPONSE =====")
    print(result)
    print("=========================\n")


    # ==========================================
    # PARSE JSON
    # ==========================================

    try:

        data = json.loads(result)

    except json.JSONDecodeError as exc:

        raise ValueError(
            "Groq returned malformed JSON. "
            "The model response could not be parsed."
        ) from exc


    candidates = data.get(
        "candidates",
        []
    )


    if not candidates:

        raise ValueError(
            "No candidate evaluations were "
            "returned by Groq."
        )


    # ==========================================
    # PYTHON-CALCULATED FINAL SCORE
    # ==========================================

    def calculate_score(candidate):

        technical = safe_score(
            candidate.get(
                "technical_skills_score",
                0
            )
        )

        experience = safe_score(
            candidate.get(
                "experience_score",
                0
            )
        )

        education = safe_score(
            candidate.get(
                "education_score",
                0
            )
        )

        certifications = safe_score(
            candidate.get(
                "certifications_score",
                0
            )
        )

        domain = safe_score(
            candidate.get(
                "domain_knowledge_score",
                0
            )
        )

        soft_skills = safe_score(
            candidate.get(
                "soft_skills_score",
                0
            )
        )

        final_score = (

            technical * 0.25

            + experience * 0.20

            + education * 0.10

            + certifications * 0.10

            + domain * 0.20

            + soft_skills * 0.15
        )

        return round(final_score)


    # ==========================================
    # NORMALIZE CANDIDATES
    # ==========================================

    for candidate in candidates:

        candidate["calculated_score"] = (
            calculate_score(candidate)
        )

        score = candidate[
            "calculated_score"
        ]

        # Make sure lists always exist.

        for field in [
            "matched_skills",
            "missing_skills",
            "strengths",
            "weaknesses",
        ]:

            value = candidate.get(field, [])

            if not isinstance(value, list):
                value = [str(value)]

            candidate[field] = value


        # Recommendation is deterministic.
        # The LLM does NOT decide this.

        if score >= 90:

            candidate["recommendation"] = (
                "Strongly Recommended"
            )

        elif score >= 80:

            candidate["recommendation"] = (
                "Recommended"
            )

        elif score >= 70:

            candidate["recommendation"] = (
                "Consider"
            )

        elif score >= 60:

            candidate["recommendation"] = (
                "Weak Match"
            )

        else:

            candidate["recommendation"] = (
                "Not Recommended"
            )


    # ==========================================
    # RANK CANDIDATES
    # ==========================================

    candidates.sort(
        key=lambda candidate:
            candidate["calculated_score"],
        reverse=True
    )


    for rank, candidate in enumerate(
        candidates,
        start=1
    ):

        candidate["rank"] = rank


    # ==========================================
    # TOP 5
    # ==========================================

    top_5 = candidates[:5]


    # ==========================================
    # HIRING RECOMMENDATION
    # ==========================================

    best_candidate = candidates[0]

    hiring_recommendation = (

        f"{best_candidate['candidate_name']} "
        f"is the strongest candidate with a "
        f"score of "
        f"{best_candidate['calculated_score']}/100. "
        f"The final hiring decision should be "
        f"made by a human recruiter after reviewing "
        f"the candidate's full profile."
    )


    # ==========================================
    # REPORT
    # ==========================================

    report_file = (
        OUTPUTS_DIR / "report.md"
    )

    with report_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "# AI HR Resume Screening Report\n\n"
        )

        file.write(
            "## Job Summary\n\n"
        )

        file.write(
            job_description
        )

        file.write("\n\n")


        # Candidate evaluation

        file.write(
            "## Candidate Evaluation\n\n"
        )

        file.write(
            "| Rank | Candidate | Score | "
            "Recommendation | Remarks |\n"
        )

        file.write(
            "|---:|---|---:|---|---|\n"
        )


        for candidate in candidates:

            file.write(
                f"| {candidate['rank']} "
                f"| {candidate['candidate_name']} "
                f"| {candidate['calculated_score']}/100 "
                f"| {candidate['recommendation']} "
                f"| {candidate.get('remarks', '')} |\n"
            )


        # Top 5

        file.write(
            "\n# Top 5 Candidates\n\n"
        )


        for candidate in top_5:

            file.write(
                f"## #{candidate['rank']} "
                f"{candidate['candidate_name']} — "
                f"{candidate['calculated_score']}/100\n\n"
            )


            file.write(
                "### Matched Skills\n\n"
            )

            for skill in candidate.get(
                "matched_skills",
                []
            ):

                file.write(
                    f"- {skill}\n"
                )


            file.write(
                "\n### Missing Skills\n\n"
            )

            for skill in candidate.get(
                "missing_skills",
                []
            ):

                file.write(
                    f"- {skill}\n"
                )


            file.write(
                "\n### Strengths\n\n"
            )

            for strength in candidate.get(
                "strengths",
                []
            ):

                file.write(
                    f"- {strength}\n"
                )


            file.write(
                "\n### Weaknesses\n\n"
            )

            for weakness in candidate.get(
                "weaknesses",
                []
            ):

                file.write(
                    f"- {weakness}\n"
                )


            file.write(
                "\n### Remarks\n\n"
            )

            file.write(
                candidate.get(
                    "remarks",
                    ""
                )
            )

            file.write("\n\n")


        # Hiring recommendation

        file.write(
            "# Hiring Recommendation\n\n"
        )

        file.write(
            f"**Recommended Candidate:** "
            f"{best_candidate['candidate_name']}\n\n"
        )

        file.write(
            f"**Final Score:** "
            f"{best_candidate['calculated_score']}/100\n\n"
        )

        file.write(
            hiring_recommendation
        )

        file.write(
            "\n\n---\n\n"
        )

        file.write(
            "*AI-assisted screening is decision support. "
            "Final hiring decisions should be made by "
            "qualified human reviewers.*"
        )


    # ==========================================
    # CHART
    # ==========================================

    names = [
        candidate["candidate_name"]
        for candidate in candidates
    ]

    scores = [
        candidate["calculated_score"]
        for candidate in candidates
    ]


    plt.figure(
        figsize=(10, 6)
    )

    bars = plt.bar(
        names,
        scores
    )

    plt.title(
        "AI Resume Screening — Candidate Scores"
    )

    plt.xlabel(
        "Candidates"
    )

    plt.ylabel(
        "Score / 100"
    )

    plt.ylim(
        0,
        100
    )


    for bar, score in zip(
        bars,
        scores
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,

            score + 2,

            f"{score}/100",

            ha="center",

            fontweight="bold"
        )


    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    plt.tight_layout()


    chart_file = (
        OUTPUTS_DIR / "scores.png"
    )

    plt.savefig(
        chart_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


    print(
        "\n===== FINAL RANKING ====="
    )

    for candidate in candidates:

        print(
            f"#{candidate['rank']} "
            f"{candidate['candidate_name']} "
            f"— "
            f"{candidate['calculated_score']}/100 "
            f"— "
            f"{candidate['recommendation']}"
        )

    print(
        "=========================\n"
    )


    # ==========================================
    # RETURN RESULTS TO FRONTEND
    # ==========================================

    return {

        "job_title":
            extract_job_title(
                job_description
            ),

        "candidate_count":
            len(candidates),

        "candidates":
            candidates,

        "top_5":
            top_5,

        "hiring_recommendation":
            hiring_recommendation,
    }


# ==========================================
# EXTRACT JOB TITLE
# ==========================================

def extract_job_title(
    job_description
):

    for line in (
        job_description.splitlines()
    ):

        line = line.strip()

        if line.startswith("# "):

            return line[2:].strip()


    return "Resume Screening"


# ==========================================
# WEB ROUTES
# ==========================================

@app.get("/")
def home():

    return send_from_directory(
        BASE_DIR,
        "index.html"
    )


@app.post("/api/screen")
def screen():

    try:

        results = run_screening()

        return jsonify({

            "success": True,

            "results": results,
        })


    except Exception as exc:

        print(
            "\n===== APPLICATION ERROR ====="
        )

        print(
            repr(exc)
        )

        print(
            "=============================\n"
        )

        return jsonify({

            "success": False,

            "error": str(exc),
        }), 500


# ==========================================
# OUTPUT FILES
# ==========================================

@app.get(
    "/outputs/<path:filename>"
)
def outputs(filename):

    return send_from_directory(
        OUTPUTS_DIR,
        filename
    )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                5000
            )
        ),
        debug=True
    )