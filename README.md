AI Resume Intelligence System
An AI-powered resume matching and career intelligence platform built using transformer-based semantic similarity, cross-encoder re-ranking, hybrid scoring, and ATS keyword analysis.
This system simulates a real-world hiring pipeline by combining machine learning and rule-based techniques to evaluate resume-job alignment.

Project Overview
Modern recruitment pipelines use:
•	Semantic similarity models
•	Applicant Tracking Systems (ATS)
•	Experience filtering
•	Skill-based ranking mechanisms
This project recreates that pipeline using:
•	Transformer embeddings
•	IDF-weighted skill scoring
•	Experience intelligence
•	Cross-encoder contextual re-ranking
•	ATS keyword optimization
•	Dynamic AI feedback generation
•	PDF report generation
The goal is to provide candidates with actionable insights into how well their resume aligns with job roles.

System Architecture
1. Resume Processing Layer
•	DOCX parsing using python-docx
•	PDF parsing using PyMuPDF
•	Text normalization

2. Feature Engineering Layer
Experience Extraction
Regex-based detection of years of experience:
(\d+)\s*\+?\s*years?
Used to calculate an Experience Fit score relative to job requirements.

Weighted Skill Scoring (IDF-Based)
Skill weights are calculated using:
weight = log(total_jobs / skill_frequency)
This ensures:
•	Rare skills receive higher importance
•	Frequently occurring skills receive lower importance

ATS Keyword Analysis
ATS readiness score is calculated as:
•	70% Keyword Coverage
•	30% Keyword Density
This simulates how applicant tracking systems evaluate resumes.

3. Semantic Matching Layer
Model used:
all-MiniLM-L6-v2
Purpose:
•	Generate resume embeddings
•	Generate job embeddings
•	Compute cosine similarity for initial ranking

4. Hybrid Scoring Engine
Final hybrid score:
0.5 × Semantic Similarity
+ 0.3 × Skill Match
+ 0.2 × Experience Fit
This forms the primary ranking stage before re-ranking.

5. Cross Encoder Re-Ranking
Model used:
cross-encoder/ms-marco-MiniLM-L-6-v2
Purpose:
•	Deep contextual evaluation of resume-job pairs
•	Re-rank top 20 jobs
•	Improve final ranking precision
Re-ranking formula:
0.6 × CrossScore
+ 0.4 × HybridScore

6. AI Career Insight Engine
Generates structured feedback based on:
•	ATS readiness score
•	Missing keywords
•	Skill gaps
•	Semantic alignment
•	Experience positioning
Provides improvement suggestions tailored to each job role.

7. PDF Report Generation
Downloadable structured report including:
•	Overall match score
•	Skill match score
•	Semantic similarity
•	Experience fit
•	ATS readiness score
•	AI-generated career insights
ai-resume-intelligence/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   └── jobs/
│   │       └── job_dataset.csv
│   │
│   └── processed/
│       └── job_embeddings.npy
│
├── requirements.txt
├── README.md
└── .gitignore
