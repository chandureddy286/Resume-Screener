# AI Resume Screener (Simplified, Single-File Version)

A leaner version of the resume screener project — one file, five libraries,
no API server. Streamlit calls the scoring logic directly.

## What changed from the original version
- No FastAPI / uvicorn / API layer — Streamlit calls the Python functions directly
- No spaCy (was only used for optional named-entity extraction that wasn't
  used in scoring anyway) — skills/experience/education are extracted with
  plain regex instead
- No pandas/numpy — they weren't actually used by the scoring logic
- Everything lives in ONE file (`app.py`) instead of 6 separate files

## What's still here (the core value of the project)
- PDF/DOCX/TXT parsing
- Skill, experience, and education extraction
- Hybrid scoring: TF-IDF (keyword) + sentence embeddings (semantic meaning) + skill overlap
- Explainability: matched/missing skills + most relevant resume lines

## Setup

```bash
python -m venv venv
source venv/bin/activate      # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

No `spacy download` step needed this time — one less thing that can go wrong.

## Run

```bash
streamlit run app.py
```

That's the only command you need. It opens `http://localhost:8501` automatically.

## Why this version is simpler to explain in an interview

- **One file, one command** — the whole pipeline is readable top to bottom
- **5 libraries instead of 12** — easier to justify every dependency
- **No API layer** — appropriate for a single-user tool; if asked "why no
  API?", the honest answer is: "For a tool used by one person at a time
  through this UI, a separate API added complexity without benefit. I'd add
  one back if this needed to serve multiple applications or scale to many
  concurrent users."

## Known limitations (same as before, worth mentioning proactively)
- No OCR — scanned/image PDFs won't extract text
- Regex-based experience/education extraction misses non-standard phrasing
- Skill taxonomy is a curated list (~55 skills) — not exhaustive
