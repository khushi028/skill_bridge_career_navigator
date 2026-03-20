import os
import re
import json
import requests
import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

app = FastAPI(title="Career Analyzer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# health test
@app.get("/health")
def health():
    return {"status": "ok"}


# ─── PDF Text Extraction ──────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()


# ─── Build Prompt ─────────────────────────────────────────────────────────────

def build_prompt(resume_text: str, github_url: Optional[str]) -> str:
    if github_url:
        github_section = f"""
Also analyze this GitHub profile URL: {github_url}
For github_analysis provide:
- overall_score: number 0-100
- strengths: list of 3 positive observations
- gaps: list of 3 weak areas
- improvements: list of 3 specific actionable improvements
- note: ""
"""
    else:
        github_section = """
No GitHub URL provided. Infer github_analysis from resume projects:
- overall_score: estimated 0-100 based on projects mentioned
- strengths: 2-3 tech strengths inferred from resume
- gaps: 2-3 likely GitHub gaps based on resume
- improvements: 3 specific suggestions to improve their GitHub
- note: "Estimated from resume — share your GitHub for accurate analysis"
"""

    return f"""You are a career coach and ATS specialist. Analyze this resume and infer the best job roles — do NOT ask for input.

Resume:
{resume_text}

{github_section}

Return ONLY valid JSON, no markdown, no code fences, no extra text:
{{
  "ats_score": <0-100>,
  "ats_breakdown": {{"keyword_match": <0-100>, "formatting": <0-100>, "section_completeness": <0-100>, "action_verbs": <0-100>}},
  "ats_summary": "<2 sentences>",
  "top_roles": [
    {{"role": "<role>", "match": <0-100>, "reason": "<one line>"}},
    {{"role": "<role>", "match": <0-100>, "reason": "<one line>"}},
    {{"role": "<role>", "match": <0-100>, "reason": "<one line>"}}
  ],
  "matched_skills": ["<skill>"],
  "missing_skills": [
    {{"skill": "<skill>", "priority": "high|medium|low", "why": "<one line>"}},
    {{"skill": "<skill>", "priority": "high|medium|low", "why": "<one line>"}},
    {{"skill": "<skill>", "priority": "high|medium|low", "why": "<one line>"}},
    {{"skill": "<skill>", "priority": "high|medium|low", "why": "<one line>"}},
    {{"skill": "<skill>", "priority": "high|medium|low", "why": "<one line>"}}
  ],
  "resume_improvements": [
    {{"area": "<area>", "issue": "<issue>", "fix": "<fix>"}},
    {{"area": "<area>", "issue": "<issue>", "fix": "<fix>"}},
    {{"area": "<area>", "issue": "<issue>", "fix": "<fix>"}},
    {{"area": "<area>", "issue": "<issue>", "fix": "<fix>"}}
  ],
  "roadmap": [
    {{"week": "Week 1-2", "focus": "<skill>", "goal": "<goal>", "resources": [{{"title": "<title>", "platform": "YouTube|Coursera|Udemy|Docs", "url": "#"}}]}},
    {{"week": "Week 3-4", "focus": "<skill>", "goal": "<goal>", "resources": [{{"title": "<title>", "platform": "YouTube|Coursera|Udemy|Docs", "url": "#"}}]}},
    {{"week": "Week 5-6", "focus": "<skill>", "goal": "<goal>", "resources": [{{"title": "<title>", "platform": "YouTube|Coursera|Udemy|Docs", "url": "#"}}]}},
    {{"week": "Week 7-8", "focus": "<skill>", "goal": "<goal>", "resources": [{{"title": "<title>", "platform": "YouTube|Coursera|Udemy|Docs", "url": "#"}}]}}
  ],
  "interview_questions": {{
    "technical": [
      {{"question": "<q>", "tip": "<tip>"}},
      {{"question": "<q>", "tip": "<tip>"}},
      {{"question": "<q>", "tip": "<tip>"}},
      {{"question": "<q>", "tip": "<tip>"}},
      {{"question": "<q>", "tip": "<tip>"}}
    ],
    "behavioural": [
      {{"question": "<q>", "tip": "<tip>"}},
      {{"question": "<q>", "tip": "<tip>"}},
      {{"question": "<q>", "tip": "<tip>"}}
    ]
  }},
  "github_analysis": {{
    "overall_score": <0-100>,
    "strengths": ["<s>", "<s>", "<s>"],
    "gaps": ["<g>", "<g>", "<g>"],
    "improvements": ["<i>", "<i>", "<i>"],
    "note": "<note or empty string>"
  }}
}}"""


# ─── Parse JSON ───────────────────────────────────────────────────────────────

def parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


# ─── Fallback ─────────────────────────────────────────────────────────────────

def fallback_response(error: str) -> dict:
    return {
        "ats_score": 0,
        "ats_breakdown": {"keyword_match": 0, "formatting": 0, "section_completeness": 0, "action_verbs": 0},
        "ats_summary": "Analysis unavailable — AI service is down. Please try again in a few minutes.",
        "top_roles": [
            {"role": "Unavailable", "match": 0, "reason": "Could not connect to AI service"},
            {"role": "Unavailable", "match": 0, "reason": "Could not connect to AI service"},
            {"role": "Unavailable", "match": 0, "reason": "Could not connect to AI service"}
        ],
        "matched_skills": [],
        "missing_skills": [{"skill": "Unavailable", "priority": "high", "why": "AI service unreachable"}],
        "resume_improvements": [{"area": "Error", "issue": "Could not analyze", "fix": "Try again in a few minutes"}],
        "roadmap": [{"week": "Week 1-2", "focus": "Unavailable", "goal": "Retry when service is back", "resources": []}],
        "interview_questions": {
            "technical":   [{"question": "Service unavailable", "tip": "Try again later"}],
            "behavioural": [{"question": "Service unavailable", "tip": "Try again later"}]
        },
        "github_analysis": {"overall_score": 0, "strengths": [], "gaps": [], "improvements": [], "note": "Service unavailable"},
        "_error": error
    }


# ─── Main Analysis ────────────────────────────────────────────────────────────

def analyze_with_ai(resume_text: str, github_url: Optional[str] = None) -> dict:
    prompt = build_prompt(resume_text, github_url)
    system_msg = "You are a career analysis expert. Return only valid JSON. No markdown, no code fences, no extra text."

    # Groq first (fast, free) → NVIDIA fallback
    providers = [
        {
            "name": "Groq",
            "url": "https://api.groq.com/openai/v1/chat/completions",
            "key": GROQ_API_KEY,
            "payload": {
                "model": "llama3-8b-8192", 
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": prompt}
                ],
                "max_tokens": 4096,
                "temperature": 0.3,
                "stream": False
            }
        },
        {
            "name": "NVIDIA",
            "url": "https://integrate.api.nvidia.com/v1/chat/completions",
            "key": NVIDIA_API_KEY,
            "payload": {
                "model": "qwen/qwen3.5-122b-a10b",
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": prompt}
                ],
                "max_tokens": 8192,
                "temperature": 0.4,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False}
            }
        }
    ]

    last_error = "No providers attempted"

    for provider in providers:
        if not provider["key"]:
            print(f"Skipping {provider['name']} — key not in .env")
            continue

        headers = {
            "Authorization": f"Bearer {provider['key']}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        try:
            print(f"Trying {provider['name']}...")
            response = requests.post(provider["url"], headers=headers, json=provider["payload"], timeout=60)

            if response.status_code == 200:
                raw = response.json()["choices"][0]["message"]["content"]
                print(f"Success with {provider['name']}")
                return parse_json(raw)

            last_error = f"{provider['name']} {response.status_code}: {response.text[:200]}"
            print(last_error)

        except requests.exceptions.Timeout:
            last_error = f"{provider['name']} timed out"
            print(last_error)
        except requests.exceptions.ConnectionError:
            last_error = f"{provider['name']} connection error"
            print(last_error)
        except Exception as e:
            last_error = f"{provider['name']} error: {str(e)}"
            print(last_error)

    return fallback_response(last_error)


# ─── Endpoint ─────────────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze_resume(
    file: UploadFile = File(...),
    github_url: Optional[str] = Form(None)
):
    content = await file.read()
    resume_text = extract_text_from_pdf(content)
    if len(resume_text) < 50:
        return {"error": "Could not extract text. Please upload a text-based PDF."}
    return analyze_with_ai(resume_text, github_url)


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")
