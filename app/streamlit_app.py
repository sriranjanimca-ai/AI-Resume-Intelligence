import streamlit as st
import pandas as pd
import numpy as np
import re
import fitz
from docx import Document
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import io

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Resume Intelligence", page_icon="🚀", layout="wide")

# ---------------- CLEAN CSS ----------------
st.markdown("""
<style>
.job-card {
    padding: 20px;
    border-radius: 12px;
    background-color: #111827;
    margin-bottom: 20px;
    color: white;
}
.feedback-box {
    background-color: #1f2937;
    padding: 15px;
    border-radius: 10px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 AI Resume Intelligence System")
st.markdown("Smart Matching • Skill Intelligence • ATS Optimization • Career Strategy")
st.divider()

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    bi = SentenceTransformer("all-MiniLM-L6-v2")
    cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return bi, cross

bi_encoder, cross_encoder = load_models()

# ---------------- LOAD DATA ----------------
jobs = pd.read_csv("data/raw/jobs/job_dataset.csv")
job_embeddings = np.load("data/processed/job_embeddings.npy")

# ---------------- SKILL WEIGHTS ----------------
def build_skill_weights(df):
    counter = Counter()
    for skills in df["Skills"]:
        if pd.isna(skills): 
            continue
        for s in str(skills).split(";"):
            s = s.strip().lower()
            if s:
                counter[s] += 1
    total = len(df)
    return {s: np.log(total / c) for s, c in counter.items()}

skill_weights = build_skill_weights(jobs)

# ---------------- TEXT EXTRACTION ----------------
def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# ---------------- EXPERIENCE ----------------
def extract_experience(text):
    matches = re.findall(r'(\d+)\s*\+?\s*years?', text.lower())
    return max([int(m) for m in matches]) if matches else None

def experience_score(resume_years, job_years):
    if resume_years is None:
        return 0.5
    match = re.search(r'\d+', str(job_years))
    if not match:
        return 0.5
    job_years = int(match.group())
    diff = resume_years - job_years
    if abs(diff) <= 2:
        return 1
    if diff > 2 and diff <= 5:
        return 0.85
    if diff > 5:
        return 0.7
    if diff < 0:
        return 1 / (1 + abs(diff))
    return 0.5

# ---------------- SKILL MATCH ----------------
def weighted_skill_overlap(resume_text, job_skills):
    total = 0
    matched = 0
    resume_lower = resume_text.lower()

    if pd.isna(job_skills):
        return 0

    for s in str(job_skills).split(";"):
        s = s.strip().lower()
        if not s:
            continue
        weight = skill_weights.get(s, 1)
        total += weight
        if s in resume_lower:
            matched += weight

    return matched / total if total else 0

# ---------------- ATS ANALYSIS ----------------
def ats_keyword_analysis(resume_text, job_skills):

    resume_words = re.findall(r'\b\w+\b', resume_text.lower())
    total_words = len(resume_words)

    if pd.isna(job_skills):
        return 0, [], []

    job_keywords = [s.strip().lower() for s in str(job_skills).split(";") if s.strip()]

    matched = []
    missing = []
    keyword_counts = {}

    for keyword in job_keywords:
        count = resume_text.lower().count(keyword)
        keyword_counts[keyword] = count

        if count > 0:
            matched.append(keyword)
        else:
            missing.append(keyword)

    match_percentage = int((len(matched) / len(job_keywords)) * 100) if job_keywords else 0

    density_score = 0
    if total_words > 0:
        avg_density = sum(keyword_counts.values()) / total_words
        density_score = min(int(avg_density * 1000), 100)

    ats_score = int((match_percentage * 0.7) + (density_score * 0.3))

    return ats_score, matched[:5], missing[:5]

# ---------------- ADVANCED FEEDBACK ----------------
def generate_advanced_feedback(resume_text, job_row, skill_score, semantic_score, exp_score):

    feedback = []
    title = job_row["Title"]

    ats_score, matched_keywords, missing_keywords = ats_keyword_analysis(
        resume_text,
        job_row["Skills"]
    )

    if ats_score < 70:
        feedback.append(
            f"ATS keyword alignment is {ats_score}%. Improving coverage for terms like {', '.join(missing_keywords[:3])} would increase shortlisting probability."
        )

    if missing_keywords:
        feedback.append(
            f"Strengthening core skills such as {', '.join(missing_keywords[:3])} would significantly increase alignment with the {title} role."
        )

    if semantic_score < 0.55:
        feedback.append(
            f"Resume alignment with the {title} role is moderate. Tailor summary and achievements more specifically."
        )

    if skill_score < 0.6:
        feedback.append(
            "Add measurable impact (%, scale, performance improvements) to strengthen credibility."
        )

    if exp_score < 0.8:
        feedback.append(
            "Experience positioning could be optimized for stronger recruiter perception."
        )

    if not feedback:
        feedback.append(
            "Strong alignment detected. Minor refinements could further improve visibility."
        )

    return feedback

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Resume (.docx or .pdf)", type=["docx", "pdf"])

if uploaded_file:

    resume_text = extract_text_from_docx(uploaded_file) if uploaded_file.name.endswith(".docx") else extract_text_from_pdf(uploaded_file)
    resume_years = extract_experience(resume_text)

    with st.expander("📄 Resume Preview"):
        st.text_area("Extracted Text", resume_text[:3000], height=300)

    # Scoring
    resume_embedding = bi_encoder.encode([resume_text])
    semantic_scores = cosine_similarity(resume_embedding, job_embeddings)[0]
    skill_scores = jobs["Skills"].apply(lambda x: weighted_skill_overlap(resume_text, x))
    experience_scores = jobs["YearsOfExperience"].apply(lambda x: experience_score(resume_years, x))

    hybrid_scores = np.array(
        0.5 * semantic_scores +
        0.3 * skill_scores.values +
        0.2 * experience_scores.values
    )

    # Cross Encoder Re-ranking
    top20 = np.argsort(hybrid_scores)[-20:][::-1]

    pairs = [
        (
            resume_text,
            jobs.iloc[i]["Title"] + " " +
            str(jobs.iloc[i]["Skills"]) + " " +
            str(jobs.iloc[i]["Responsibilities"])
        )
        for i in top20
    ]

    cross_scores = cross_encoder.predict(pairs)

    reranked = []
    for i in range(len(top20)):
        idx = top20[i]
        combined = 0.6 * cross_scores[i] + 0.4 * hybrid_scores[idx]
        reranked.append((idx, combined))

    reranked.sort(key=lambda x: x[1], reverse=True)

    st.subheader("💼 Top 5 Matching Jobs")

    shown = set()
    final_results = []

    for idx, _ in reranked:
        title = jobs.iloc[idx]["Title"]
        if title in shown:
            continue
        shown.add(title)

        overall = int(hybrid_scores[idx] * 100)
        skill = int(skill_scores.iloc[idx] * 100)
        similarity = int(semantic_scores[idx] * 100)
        exp_fit = int(experience_scores.iloc[idx] * 100)

        final_results.append((title, overall, skill, similarity, exp_fit, idx))

        if len(final_results) == 5:
            break

    for title, overall, skill, similarity, exp_fit, idx in final_results:

        st.markdown(f'<div class="job-card"><h3>{title}</h3></div>', unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Overall", f"{overall}%")
        col2.metric("Skill Match", f"{skill}%")
        col3.metric("Similarity", f"{similarity}%")
        col4.metric("Experience Fit", f"{exp_fit}%")

        ats_score, _, _ = ats_keyword_analysis(resume_text, jobs.iloc[idx]["Skills"])
        col5.metric("ATS Readiness", f"{ats_score}%")

        feedback = generate_advanced_feedback(
            resume_text,
            jobs.iloc[idx],
            skill_scores.iloc[idx],
            semantic_scores[idx],
            experience_scores.iloc[idx]
        )

        st.markdown("🧠 AI Career Insight:")
        st.markdown('<div class="feedback-box">', unsafe_allow_html=True)
        for f in feedback:
            st.write("•", f)
        st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

    # ---------------- PDF REPORT ----------------
    if st.button("📥 Download PDF Report"):

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer)
        elements = []
        styles = getSampleStyleSheet()

        elements.append(Paragraph("<b>AI Resume Matching Report</b>", styles["Title"]))
        elements.append(Spacer(1, 0.3 * inch))

        for title, overall, skill, similarity, exp_fit, idx in final_results:

            elements.append(Paragraph(f"<b>{title}</b>", styles["Heading2"]))
            elements.append(Spacer(1, 0.2 * inch))

            ats_score, _, _ = ats_keyword_analysis(resume_text, jobs.iloc[idx]["Skills"])

            elements.append(Paragraph(f"Overall Match: {overall}%", styles["Normal"]))
            elements.append(Paragraph(f"Skill Match: {skill}%", styles["Normal"]))
            elements.append(Paragraph(f"Role Similarity: {similarity}%", styles["Normal"]))
            elements.append(Paragraph(f"Experience Fit: {exp_fit}%", styles["Normal"]))
            elements.append(Paragraph(f"ATS Readiness: {ats_score}%", styles["Normal"]))
            elements.append(Spacer(1, 0.2 * inch))

            feedback = generate_advanced_feedback(
                resume_text,
                jobs.iloc[idx],
                skill_scores.iloc[idx],
                semantic_scores[idx],
                experience_scores.iloc[idx]
            )

            elements.append(Paragraph("AI Career Insight:", styles["Heading3"]))
            for f in feedback:
                elements.append(Paragraph(f"• {f}", styles["Normal"]))

            elements.append(Spacer(1, 0.4 * inch))

        doc.build(elements)
        buffer.seek(0)

        st.download_button(
            "Download Report",
            data=buffer,
            file_name="resume_matching_report.pdf",
            mime="application/pdf"
        )