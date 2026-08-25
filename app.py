"""
app.py — AI Resume Screener (simplified, single-file version)

Everything in one file: text extraction, skill/experience extraction,
scoring, explanation, and the Streamlit UI. No API, no backend server —
Streamlit calls the functions directly.

Run with:
    streamlit run app.py

Only 5 libraries needed: streamlit, pdfplumber, python-docx, scikit-learn,
sentence-transformers.
"""

import io
import re
import streamlit as st
import pdfplumber
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util


# ---------------------------------------------------------------------------
# 1. TEXT EXTRACTION — pull raw text out of PDF / DOCX / TXT files
# ---------------------------------------------------------------------------

def extract_text(filename: str, file_bytes: bytes) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    buffer = io.BytesIO(file_bytes)

    if ext == "pdf":
        chunks = []
        with pdfplumber.open(buffer) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    chunks.append(page_text)
        return "\n".join(chunks)

    elif ext == "docx":
        doc = Document(buffer)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    elif ext == "txt":
        return file_bytes.decode("utf-8", errors="ignore")

    else:
        raise ValueError(f"Unsupported file type: .{ext}")


# ---------------------------------------------------------------------------
# 2. STRUCTURED EXTRACTION — skills, years of experience, education
#    (Pure regex — no spaCy/NER needed, keeps dependencies minimal)
# ---------------------------------------------------------------------------

SKILLS_TAXONOMY = [
    "python", "java", "c++", "c#", "javascript", "typescript", "sql", "r",
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "data analysis", "data visualization", "statistics",
    "pandas", "numpy", "scikit-learn", "sklearn", "tensorflow", "pytorch",
    "keras", "xgboost", "lightgbm", "spacy", "nltk", "opencv", "matplotlib",
    "flask", "fastapi", "django", "streamlit", "rest api",
    "docker", "kubernetes", "aws", "azure", "gcp", "git", "github", "linux",
    "mysql", "postgresql", "mongodb", "power bi", "tableau", "excel",
    "html", "css", "react", "node.js", "bash", "agile", "scrum",
    "communication", "leadership", "teamwork", "problem solving",
]


def extract_skills(text: str) -> list:
    text_lower = text.lower()
    found = []
    for skill in SKILLS_TAXONOMY:
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, text_lower):
            found.append(skill)
    return sorted(set(found))


def extract_years_experience(text: str) -> float:
    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?experience",
        r"experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*\+?\s*years?",
    ]
    years_found = []
    for pattern in patterns:
        years_found.extend(float(m) for m in re.findall(pattern, text.lower()))
    return max(years_found) if years_found else 0.0


def extract_education_level(text: str) -> str:
    text_lower = text.lower()
    keywords = {
        "phd": ["phd", "doctorate", "ph.d"],
        "masters": ["master's", "masters", "m.tech", "mtech", "msc", "mba"],
        "bachelors": ["bachelor's", "bachelors", "b.tech", "btech", "bsc", "b.e."],
    }
    for level in ["phd", "masters", "bachelors"]:
        if any(kw in text_lower for kw in keywords[level]):
            return level
    return "unspecified"


def profile_document(text: str) -> dict:
    return {
        "skills": extract_skills(text),
        "years_experience": extract_years_experience(text),
        "education_level": extract_education_level(text),
    }


# ---------------------------------------------------------------------------
# 3. SCORING — TF-IDF (lexical) + sentence embeddings (semantic) + skill overlap
# ---------------------------------------------------------------------------

@st.cache_resource
def load_embedding_model():
    # Cached so Streamlit only loads this once per session, not on every click
    return SentenceTransformer("all-MiniLM-L6-v2")


def tfidf_similarity(jd_text: str, resume_text: str) -> float:
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        vectors = vectorizer.fit_transform([jd_text, resume_text])
    except ValueError:
        return 0.0
    return float(cosine_similarity(vectors[0], vectors[1])[0][0])


def semantic_similarity(model, jd_text: str, resume_text: str) -> float:
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    return float(util.cos_sim(jd_emb, resume_emb).item())


def score_resume(model, jd_text: str, resume_text: str) -> dict:
    jd_profile = profile_document(jd_text)
    resume_profile = profile_document(resume_text)

    jd_skills, resume_skills = set(jd_profile["skills"]), set(resume_profile["skills"])
    matched = sorted(jd_skills & resume_skills)
    missing = sorted(jd_skills - resume_skills)
    overlap_ratio = len(matched) / len(jd_skills) if jd_skills else 0.0

    jd_years = jd_profile["years_experience"]
    resume_years = resume_profile["years_experience"]
    if jd_years <= 0:
        exp_score = 1.0
    else:
        exp_score = 1.0 if resume_years >= jd_years else max(0.0, resume_years / jd_years)

    tfidf_score = tfidf_similarity(jd_text, resume_text)
    semantic_score = semantic_similarity(model, jd_text, resume_text)

    # Weighted hybrid score: skills matter most, then semantic meaning, then experience
    final_score = 0.5 * overlap_ratio + 0.4 * semantic_score + 0.1 * exp_score

    return {
        "final_score": round(final_score, 4),
        "tfidf_score": round(tfidf_score, 4),
        "semantic_score": round(semantic_score, 4),
        "skill_overlap_ratio": round(overlap_ratio, 4),
        "matched_skills": matched,
        "missing_skills": missing,
        "resume_years_experience": resume_years,
        "jd_years_experience": jd_years,
        "resume_education": resume_profile["education_level"],
        "jd_education": jd_profile["education_level"],
    }


# ---------------------------------------------------------------------------
# 4. EXPLANATION — human-readable summary + top matching resume sentences
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list:
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 15]


def top_matching_sentences(model, jd_text: str, resume_text: str, top_n: int = 3) -> list:
    sentences = split_sentences(resume_text)
    if not sentences:
        return []
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    sent_embs = model.encode(sentences, convert_to_tensor=True)
    scores = util.cos_sim(jd_emb, sent_embs)[0].tolist()
    ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
    return [{"sentence": s, "relevance": round(sc, 4)} for s, sc in ranked[:top_n]]


def generate_summary(result: dict) -> str:
    pct = round(result["final_score"] * 100)
    matched_str = ", ".join(result["matched_skills"][:6]) or "no direct skill overlap found"
    missing_str = ", ".join(result["missing_skills"][:6]) or "none"
    return f"Overall match: {pct}%. Strong on: {matched_str}. Missing: {missing_str}."


# ---------------------------------------------------------------------------
# 5. STREAMLIT UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("📄 AI Resume Screener")
st.caption("Paste a job description, upload resumes, and get ranked, explained matches.")

model = load_embedding_model()

with st.form("screen_form"):
    jd_text = st.text_area("Job Description", height=200, placeholder="Paste the job description here...")
    uploaded_files = st.file_uploader(
        "Upload Resumes (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    submitted = st.form_submit_button("Screen Resumes")

if submitted:
    if not jd_text.strip():
        st.error("Please paste a job description.")
    elif not uploaded_files:
        st.error("Please upload at least one resume.")
    else:
        results = []
        with st.spinner("Screening resumes..."):
            for f in uploaded_files:
                try:
                    resume_text = extract_text(f.name, f.getvalue())
                    if not resume_text.strip():
                        results.append({"filename": f.name, "error": "Could not extract text."})
                        continue
                    score_result = score_resume(model, jd_text, resume_text)
                    score_result["filename"] = f.name
                    score_result["summary"] = generate_summary(score_result)
                    score_result["top_sentences"] = top_matching_sentences(model, jd_text, resume_text)
                    results.append(score_result)
                except Exception as e:
                    results.append({"filename": f.name, "error": str(e)})

        ranked = sorted(results, key=lambda r: r.get("final_score", -1), reverse=True)

        st.subheader("Ranked Candidates")
        for rank, cand in enumerate(ranked, start=1):
            if "error" in cand:
                st.warning(f"**{cand['filename']}** — could not be scored: {cand['error']}")
                continue

            pct = round(cand["final_score"] * 100)
            with st.expander(f"#{rank}  {cand['filename']}  —  {pct}% match", expanded=(rank == 1)):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Overall Match", f"{pct}%")
                    st.metric("Semantic Similarity", f"{round(cand['semantic_score']*100)}%")
                    st.metric("Skill Overlap", f"{round(cand['skill_overlap_ratio']*100)}%")
                with col2:
                    st.write(f"**Experience:** {cand['resume_years_experience']} yrs "
                             f"(required: {cand['jd_years_experience']} yrs)")
                    st.write(f"**Education:** {cand['resume_education']}")

                st.success(f"✅ Matched skills: {', '.join(cand['matched_skills']) or 'none'}")
                st.error(f"❌ Missing skills: {', '.join(cand['missing_skills']) or 'none'}")
                st.info(cand["summary"])

                if cand.get("top_sentences"):
                    st.write("**Most relevant lines from resume:**")
                    for s in cand["top_sentences"]:
                        st.write(f"- \"{s['sentence']}\" (relevance: {round(s['relevance']*100)}%)")
