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
MAX_FILE_SIZE  = 5 * 1024 * 1024  # 5 MB

app = FastAPI(title="Career Analyzer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "providers": {
            "groq":   "configured" if GROQ_API_KEY   else "missing — set GROQ_API_KEY in .env",
            "nvidia": "configured" if NVIDIA_API_KEY else "missing — set NVIDIA_API_KEY in .env"
        }
    }


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

    return f"""DOMAIN RULES — critical, follow strictly:
- Read the resume carefully and identify the candidate's PRIMARY domain (e.g. if they have Python/ML/data skills → their domain is Data Science/ML, NOT DevOps or cloud)
- missing_skills MUST only contain skills relevant to THEIR domain — do NOT suggest skills from unrelated fields
- If a candidate is in ML/AI, gaps should be things like MLOps, model deployment, deep learning — NOT Kubernetes, Terraform, or network security
- If a candidate is in frontend, gaps should be React advanced patterns, TypeScript, testing — NOT machine learning or databases
- If a candidate is in backend, gaps should be system design, caching, message queues — NOT UI design or data science
- roadmap MUST directly address ONLY the missing_skills listed — each week's focus must be one of those exact missing skills
- Do NOT add weeks for skills the candidate already has
- Do NOT suggest cloud/DevOps skills unless the resume clearly shows cloud or DevOps work
- roadmap weeks must follow this order: highest priority missing skill first, lowest priority last
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
    {{"week": "Week 1-2", "focus": "<missing_skill #1 — highest priority>", "goal": "<what they will be able to do after this week>", "resources": [{{"title": "<real resource name>", "platform": "YouTube|Coursera|Udemy|Docs", "url": "#"}}]}},
    {{"week": "Week 3-4", "focus": "<missing_skill #2>", "goal": "<what they will be able to do after this week>", "resources": [{{"title": "<real resource name>", "platform": "YouTube|Coursera|Udemy|Docs", "url": "#"}}]}},
    {{"week": "Week 5-6", "focus": "<missing_skill #3>", "goal": "<what they will be able to do after this week>", "resources": [{{"title": "<real resource name>", "platform": "YouTube|Coursera|Udemy|Docs", "url": "#"}}]}},
    {{"week": "Week 7-8", "focus": "<missing_skill #4>", "goal": "<what they will be able to do after this week>", "resources": [{{"title": "<real resource name>", "platform": "YouTube|Coursera|Udemy|Docs", "url": "#"}}]}}
  ],
  ROADMAP RULES: Each week focus MUST match one of the missing_skills exactly. Do NOT include skills the candidate already has. Stay within their domain only.
  "interview_questions": {{
    "technical": [
      {{"question": "<technical q 1 specific to their skills>", "tip": "<answer tip>"}},
      {{"question": "<technical q 2 specific to their skills>", "tip": "<answer tip>"}},
      {{"question": "<technical q 3 specific to their skills>", "tip": "<answer tip>"}},
      {{"question": "<technical q 4 specific to their skills>", "tip": "<answer tip>"}},
      {{"question": "<technical q 5 specific to their skills>", "tip": "<answer tip>"}}
    ],
    "behavioural": [
      {{"question": "<behavioural q 1>", "tip": "<answer tip>"}},
      {{"question": "<behavioural q 2>", "tip": "<answer tip>"}},
      {{"question": "<behavioural q 3>", "tip": "<answer tip>"}}
    ]
  }},
  IMPORTANT: You MUST return EXACTLY 5 technical questions and EXACTLY 3 behavioural questions. No more, no less.
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


# ─── Main Analysis — explicit calls, no loop ─────────────────────────────────

def analyze_with_ai(resume_text: str, github_url: Optional[str] = None) -> dict:
    prompt     = build_prompt(resume_text, github_url)
    system_msg = "You are a career analysis expert. Return only valid JSON. No markdown, no code fences, no extra text."

    # ── Groq (primary — fast, free) ───────────────────────────────────────────
    if GROQ_API_KEY:
        try:
            print("Trying Groq...")
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user",   "content": prompt}
                    ],
                    "max_tokens": 6000,
                    "temperature": 0.3,
                    "stream": False
                },
                timeout=60
            )
            if resp.status_code == 200:
                print("Success with Groq")
                return parse_json(resp.json()["choices"][0]["message"]["content"])
            print(f"Groq {resp.status_code}: {resp.text[:300]}")
        except requests.exceptions.Timeout:
            print("Groq timed out")
        except requests.exceptions.ConnectionError:
            print("Groq connection error")
        except Exception as e:
            print(f"Groq error: {e}")

    # ── NVIDIA (fallback) ─────────────────────────────────────────────────────
    if NVIDIA_API_KEY:
        try:
            print("Trying NVIDIA...")
            resp = requests.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {NVIDIA_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "qwen/qwen3.5-122b-a10b",
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user",   "content": prompt}
                    ],
                    "max_tokens": 8192,
                    "temperature": 0.4,
                    "stream": False,
                    "chat_template_kwargs": {"enable_thinking": False}
                },
                timeout=60
            )
            if resp.status_code == 200:
                print("Success with NVIDIA")
                return parse_json(resp.json()["choices"][0]["message"]["content"])
            print(f"NVIDIA {resp.status_code}: {resp.text[:300]}")
        except requests.exceptions.Timeout:
            print("NVIDIA timed out")
        except requests.exceptions.ConnectionError:
            print("NVIDIA connection error")
        except Exception as e:
            print(f"NVIDIA error: {e}")

    return fallback_response("All providers failed or no API keys configured")


# ─── Endpoint ─────────────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze_resume(
    file: UploadFile = File(...),
    github_url: Optional[str] = Form(None)
):
    # File type check
    if file.content_type != "application/pdf" and not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are accepted. Please upload a .pdf resume."}

    content = await file.read()

    # File size check
    if len(content) > MAX_FILE_SIZE:
        return {"error": f"File too large ({len(content)//1024}KB). Maximum allowed is 5MB."}

    # Text extraction
    resume_text = extract_text_from_pdf(content)
    if len(resume_text) < 50:
        return {"error": "Could not extract text. Please upload a text-based PDF, not a scanned image."}

    return analyze_with_ai(resume_text, github_url or None)


# ─── Serve Frontend ───────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")
