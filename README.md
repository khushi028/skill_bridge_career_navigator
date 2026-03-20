# Skill_Bridge_Career_Navigator

> AI-powered resume intelligence — ATS scoring, job role matching, skill gap analysis, learning roadmap, and interview prep in one shot.

---

## Candidate Name
Khushi Jain

## Scenario Chosen"
"Skill-Bridge-Career-Navigator

It's an AI-Powered Career Analyzer — Resume intelligence tool that parses a PDF resume, infers best-fit job roles automatically, scores ATS compatibility, identifies skill gaps, generates a personalized learning roadmap, and produces role-specific interview questions. Optionally analyzes a GitHub profile or infers GitHub health from resume projects.

## Estimated Time Spent
~5 hours
- 1 hr — System setup & planning
- 1.5 hrs — FastAPI backend (PDF extraction, provider fallback logic, JSON parsing)
- 1.5 hrs — Frontend dashboard (upload flow, 7-tab dashboard, animations)
- 1 hr — Debugging (Windows .env encoding issue, httpx version conflict)

---

## Quick Start

### Prerequisites
- Python 3.10+
- pip
- A Groq API key → https://console.groq.com
- Optional: NVIDIA API key → https://build.nvidia.com

### Environment Setup

Create a `.env` file in the project root. **On Windows, use Notepad (File → Save As → Encoding: UTF-8)** — do not use `echo` in PowerShell as it creates UTF-16 files that break python-dotenv.

```

```

### Run Commands

**Windows (PowerShell)**
```powershell
cd career-analyzer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Linux / macOS**
```bash
cd career-analyzer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open `http://127.0.0.1:8000` in your browser.

### Docker (optional)
```bash
docker build -t skill-bridge .
docker run -p 8000:8000 --env-file .env skill-bridge
```

---

## Test Commands

### 1. Health check — confirm server is running
```bash
curl http://127.0.0.1:8000/health
# Expected: {"status":"ok"}
```

### 2. End-to-end API test — upload a real PDF resume
```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -F "file=@/path/to/your/resume.pdf"
```
**Expected:** JSON response with `ats_score`, `top_roles`, `missing_skills`, `roadmap`, `interview_questions`, `github_analysis`.

### 3. Test with GitHub URL
```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -F "file=@/path/to/resume.pdf" \
  -F "github_url=https://github.com/your-username"
```

### 4. Test fallback — verify graceful failure
Temporarily set both keys to invalid values in `.env`, then run the analyze request. Expected: a structured JSON response with `ats_score: 0` and `_error` field — **not** a server crash or 500 error.

### 5. Test invalid file type
```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -F "file=@/path/to/image.png"
# Expected: {"error": "Could not extract text. Please upload a text-based PDF."}
```

### Production-level testing checklist
| Test | What to check |
|------|--------------|
| Valid PDF resume | All 7 dashboard sections populate with real data |
| Scanned/image PDF | Returns the short-text error message gracefully |
| No GitHub URL | GitHub section shows "Estimated from resume" note |
| With GitHub URL | GitHub section shows "Analysis based on your GitHub profile" |
| Groq key missing | App skips Groq, falls through to NVIDIA |
| Both keys missing | Returns fallback JSON, no 500 error |
| File > 5MB | Browser blocks upload (client-side file input constraint) |
| Button disabled check | Upload button stays disabled until a PDF is selected |

---

## AI Disclosure

**Did you use an AI assistant?**
Yes — Claude (Anthropic) was used throughout the build.

**How did you verify the suggestions?**
Every code block was read line by line before applying. The FastAPI endpoint, prompt structure, and fallback logic were traced manually to verify correctness. The frontend JS was tested in-browser with the actual API — not just reviewed in isolation. Bugs that emerged (undefined `role` variable, UTF-16 `.env` encoding, `httpx`/`openai` version conflict) were caught through real execution, not just code review.

**One example of a suggestion I rejected or changed:**
The initial suggestion used `llama-3.3-70b-versatile` on Groq as the primary model. During testing this was taking 15–25 seconds per request, which was too slow for a user-facing tool. I changed it to `llama3-8b-8192` (the 8B model) which responds in 3–5 seconds. The quality difference for structured JSON extraction is negligible — the smaller model follows the JSON schema just as reliably — but the speed improvement is significant for UX.

---

## Tradeoffs & Prioritization

**What was cut to stay within the time limit?**
- No user authentication or session persistence — every analysis is stateless
- No database — results are held in browser memory only; refreshing loses the analysis
- No streaming response — the full analysis waits for the complete LLM response before rendering
- No rate limiting or request queuing on the backend
- GitHub analysis is prompt-based inference, not actual GitHub API calls — the LLM reads the URL but cannot fetch live data without OAuth integration
- No resume file size validation on the server side (client-side only)
- No unit tests written — manual curl + browser testing only

**What would be built next with more time?**
1. Real GitHub API integration with OAuth — fetch actual commit history, language stats, README coverage, and repo quality metrics instead of prompting the LLM to guess
2. PostgreSQL persistence — save analysis results per user so they can return and track improvement over time
3. Streaming responses — stream the LLM output token-by-token so the dashboard populates progressively instead of waiting 5–10 seconds on a blank loading screen
4. Multiple resume comparison — upload two versions of your resume and see which scores higher
5. Role-specific job description input — paste a real job description and get a gap analysis against that exact role rather than generic role profiles

**Known limitations:**
- GitHub analysis quality depends entirely on what is written in the resume — the LLM cannot fetch live GitHub data without real API integration
- ATS scores are LLM-estimated, not based on a real ATS parser; they should be treated as directional guidance, not exact measurements
- Resume PDFs that are scanned images (no text layer) will fail extraction — the app returns an error message rather than using OCR (Tesseract was excluded to reduce setup complexity)
- Groq's `llama3-8b-8192` model has an 8,192 token context limit — very long resumes (4+ pages) may be truncated
- The NVIDIA Qwen model can be slow or unavailable during peak hours; the Groq fallback handles this but if both are down, the user sees a "service unavailable" dashboard
- No input sanitization on the GitHub URL field — a malformed URL is passed directly into the prompt
