# 🧠 ResumeIQ — AI-Powered Resume Screening System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-red?style=for-the-badge&logo=scikit-learn)
![HTML5](https://img.shields.io/badge/HTML5-Live%20Demo-E34F26?style=for-the-badge&logo=html5)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An end-to-end machine learning pipeline that screens, ranks, and analyzes resumes against job descriptions — with a fully interactive browser-based demo.**

[▶ Try the Live HTML Demo](#-live-html-demo) · [📓 Open in Colab](#-open-in-google-colab) · [📊 See the Pipeline](#-ml-pipeline-architecture)

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Live HTML Demo](#-live-html-demo)
- [Features](#-features)
- [ML Pipeline Architecture](#-ml-pipeline-architecture)
- [Scoring Formula](#-scoring-formula)
- [Skill Taxonomy](#-skill-taxonomy)
- [How to Use the Notebook](#-how-to-use-the-notebook)
- [How to Use the HTML Demo](#-how-to-use-the-html-demo)
- [Visualizations & Outputs](#-visualizations--outputs)
- [Datasets](#-datasets)
- [Tech Stack](#-tech-stack)
- [Results & Sample Output](#-results--sample-output)
- [Future Improvements](#-future-improvements)

---

## 🎯 Project Overview

ResumeIQ is a final-year data science project that builds a complete **ML-based hiring pipeline** from scratch — no pre-trained LLMs, no black boxes. It uses classical NLP techniques (TF-IDF, cosine similarity) combined with rule-based skill extraction and heuristic scoring to produce ranked candidate lists with rich visual analytics.

The system works in two modes:

| Mode | Where | Best For |
|------|-------|----------|
| **Python Notebook** | Google Colab / local | Batch screening with full charts & CSV export |
| **HTML Single-File App** | Any browser | Interactive demo, no installation needed |

Both use the **exact same scoring logic** — the HTML demo is a faithful JavaScript port of the Python pipeline.

---

## 🌐 Live HTML Demo

The file `resume_screener.html` is a **zero-dependency, single-file web app** that runs entirely in your browser. No server, no API key, no installation.

**To use it:**
1. Download [`resume_screener.html`](./resume_screener.html) from this repo
2. Open it in any modern browser (Chrome, Firefox, Edge, Safari)
3. Upload `.txt` resume files or paste text directly
4. Select a target job role
5. Click **⚡ Analyze Resumes** and get instant ranked results

> 💡 **Don't have resumes handy?** Click the **🧪 Load Demo Resumes** button (bottom-left corner) to load 4 pre-built sample resumes and see the system in action immediately.

### What the HTML Demo Does

The browser app replicates the full Python pipeline in JavaScript:

- **Drag & drop or file upload** for `.txt`, `.pdf`, `.doc` files
- **Paste resume text** directly into a textarea
- **Role selector** with 6 job profiles (Data Scientist, ML Engineer, Full Stack Dev, DevOps, Data Analyst, Software Engineer)
- **5-factor composite scoring** with animated progress bars
- **4 result tabs:** Rankings, Skill Heatmap, Gap Analysis, Role Comparison
- **Report Card tab** with radar chart for the top candidate
- **Export to CSV** with one click

---

## ✨ Features

### Core Pipeline (Python Notebook)
- ✅ **148-skill taxonomy** across 9 technical categories
- ✅ **TF-IDF cosine similarity** using sklearn's `TfidfVectorizer` (bigrams, 8000 features, sublinear TF)
- ✅ **Greedy skill extraction** — longest-first matching to avoid false positives (e.g., "machine learning" matched before "learning")
- ✅ **Heuristic experience extraction** — regex patterns for "X years of experience" plus seniority-title counting
- ✅ **Education level scoring** — detects Ph.D., M.S., MBA, B.S., and more via regex
- ✅ **Category coverage scoring** — rewards breadth across skill domains, not just raw skill count
- ✅ **Multi-role comparison** — runs the same resumes against 6 roles simultaneously
- ✅ **Personal resume analyzer** — upload your own resume (PDF or TXT) and get a personalized report

### Visualizations (6 Matplotlib Panels)
- 📊 **Recruiter Dashboard** — composite ranking bar chart, score distribution histogram, experience vs. score scatter, skills coverage heatmap, education breakdown pie, top missing skills bar
- 🔥 **Skill Gap Matrix** — heatmap showing ✓/✗ for every required skill across top 15 candidates
- 📈 **Analytics Deep-Dive** — category leaderboard, resume count per category, score percentile distribution, skill category frequency
- 🔄 **Role Comparison** — grouped bar chart comparing top/mean scores across all 6 roles

### Exports
- 📁 CSV with all scores, matched/missing/bonus skills, snippets
- 🖼️ PNG files for all dashboard charts (auto-downloaded in Colab)

---

## 🏗️ ML Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                               │
│   Resume Text (CSV / PDF / TXT)  +  Job Description (CSV)  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 PREPROCESSING                                │
│  • Lowercase + URL strip + special char removal             │
│  • Custom stopword removal (domain-aware)                   │
│  • Whitespace normalization                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌──────────────────┐   ┌──────────────────────────────────────┐
│  SKILL EXTRACTOR  │   │         TF-IDF VECTORIZER            │
│  • 148 skills     │   │  • ngram_range=(1,2)                 │
│  • 9 categories   │   │  • max_features=8000                 │
│  • Longest-first  │   │  • sublinear_tf=True                 │
│    greedy match   │   │  • Cosine similarity (resume vs JD)  │
└────────┬─────────┘   └────────────────┬─────────────────────┘
         │                              │
         ▼                              ▼
┌──────────────────────────────────────────────────────────────┐
│                    SCORING MODULE                             │
│                                                              │
│   skill_match_score  =  matched / required   (→ 0–100%)     │
│   category_coverage  =  matched_cats / jd_cats              │
│   experience_score   =  min(years / 8, 1.0)                 │
│   education_score    =  PhD→1.0 | MS→0.85 | BS→0.70 …      │
│                                                              │
│   COMPOSITE = 0.35·skill + 0.30·tfidf + 0.15·catcov         │
│             + 0.12·exp  + 0.08·edu                          │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    OUTPUT LAYER                               │
│   • Ranked DataFrame with status labels                      │
│   • Matched / Missing / Bonus skills per candidate           │
│   • Category leaderboard & skill gap report                  │
│   • CSV export + 4 matplotlib dashboard PNGs                 │
└──────────────────────────────────────────────────────────────┘
```

---

## 📐 Scoring Formula

Each resume receives a **composite score (0–100%)** calculated as a weighted sum of five factors:

| Factor | Weight | How It's Calculated |
|--------|--------|---------------------|
| **Skill Match** | **35%** | `matched_skills / required_skills_in_JD` |
| **TF-IDF Similarity** | **30%** | Cosine similarity between resume and JD using bigram TF-IDF vectors |
| **Category Coverage** | **15%** | `unique_skill_categories_matched / total_JD_categories` |
| **Experience** | **12%** | `min(estimated_years / 8, 1.0)` — capped at 8 years for full score |
| **Education** | **8%** | Ph.D.→100, M.S.→85, MBA→82, B.S.→70, Associate/Diploma→55 |

**Status thresholds:**

| Score | Status | Label |
|-------|--------|-------|
| ≥ 70% | ✅ | Highly Recommended |
| 50–69% | ⚠️ | Potential Fit |
| < 50% | ❌ | Not Recommended |

---

## 🗂️ Skill Taxonomy

The system recognizes **148 skills** across **9 categories**, used for both extraction and category-coverage scoring:

| # | Category | Count | Examples |
|---|----------|-------|---------|
| 1 | `programming_languages` | 20 | Python, Java, JavaScript, TypeScript, C++, Go, Rust, SQL |
| 2 | `ml_ai` | 28 | Machine Learning, Deep Learning, NLP, BERT, GPT, XGBoost, LLM |
| 3 | `ml_frameworks` | 15 | TensorFlow, PyTorch, Keras, Scikit-learn, Hugging Face, MLflow |
| 4 | `data_tools` | 17 | Pandas, NumPy, Spark, Tableau, Power BI, Airflow, Databricks |
| 5 | `databases` | 13 | MySQL, PostgreSQL, MongoDB, Redis, Snowflake, BigQuery |
| 6 | `cloud_devops` | 19 | AWS, Azure, GCP, Docker, Kubernetes, Terraform, CI/CD, Git |
| 7 | `web_frameworks` | 14 | Django, Flask, FastAPI, React, Angular, Vue, GraphQL |
| 8 | `soft_skills` | 12 | Communication, Leadership, Agile, Scrum, Problem Solving |
| 9 | `statistics` | 10 | Statistics, A/B Testing, Bayesian, Hypothesis Testing |

**Why greedy (longest-first) matching?**
Skills are sorted by length before matching so that multi-word phrases like `"machine learning"` are matched before single words like `"learning"`. This prevents over-counting and false positives.

---

## 📓 How to Use the Notebook

The notebook is organized into **14 sequential steps**. Here's what each step does:

| Step | Cell | What It Does |
|------|------|-------------|
| 1 | Install & Import | Loads pandas, numpy, matplotlib, sklearn |
| 2 | Upload Datasets | Kaggle CSV upload widget (Colab) or synthetic fallback |
| 3 | Skill Taxonomy | Defines 148 skills across 9 categories |
| 4 | Text Preprocessing | `clean_text()`, `extract_skills()`, `extract_years_experience()`, `extract_education_score()` |
| 5 | Synthetic Data | Generates 100 realistic resumes + 6 JDs if no CSVs uploaded |
| 6 | Load Datasets | Smart CSV loader with column-name normalization |
| 7 | ML Pipeline Class | `ResumeScreeningPipeline` — core scoring engine |
| 8 | Run Pipeline | Screens all resumes for `TARGET_ROLE = "Data Scientist"` |
| 9 | Analytics | Category leaderboard + skill gap report (top 20 missing skills) |
| 10 | Recruiter Dashboard | 6-panel matplotlib figure — main output chart |
| 11 | Skill Gap Heatmap | Matrix of required skills × top 15 candidates |
| 12 | Analytics Deep-Dive | 4-chart panel: category scores, resume distribution, percentile curve, skill frequency |
| 13 | Multi-Role Comparison | Runs pipeline for all 6 roles, generates comparison bar chart |
| 14 | Export & Download | CSV + all PNGs downloaded automatically in Colab |
| BONUS | Personal Analyzer | Upload your own resume → visual personal report |

### Changing the Target Role

In **Step 8**, simply change `TARGET_ROLE`:

```python
TARGET_ROLE = "Data Scientist"   # default
# Other options:
TARGET_ROLE = "ML Engineer"
TARGET_ROLE = "Full Stack Dev"
TARGET_ROLE = "DevOps Engineer"
TARGET_ROLE = "Data Analyst"
TARGET_ROLE = "Software Engineer"
```

---

## 🌐 How to Use the HTML Demo

The HTML demo mirrors the Python notebook but runs 100% in your browser using JavaScript.

### Step-by-Step

**1. Upload Resumes**
- Drag & drop `.txt` files onto the upload zone, or click to browse
- Or paste resume text into the textarea and click **+ Add Resume**
- Or click **🧪 Load Demo Resumes** to instantly load 4 sample profiles

**2. Select a Role**
- Click one of the 6 role cards (Data Scientist, ML Engineer, etc.)
- Each card shows how many skills are required for that role

**3. Analyze**
- Click **⚡ Analyze Resumes**
- Watch the per-resume progress indicator
- Results appear automatically when done

**4. Explore Results**

| Tab | What You See |
|-----|-------------|
| **Rankings** | Ranked candidate cards with score bars, factor breakdown, and expandable skill chips (click any card) |
| **Skill Heatmap** | ✓/✗ matrix of required skills × top 15 candidates |
| **Gap Analysis** | Most-missed skills + required skill category distribution |
| **Role Comparison** | How the top candidate scores across all 6 roles |
| **Top Report Card** | Grade (A+/A/B/C/F), key metrics, and radar chart for #1 candidate |

**5. Export**
- Click **⬇ Export CSV** to download results as a spreadsheet

### Scoring Weights (shown in the UI)

```
🔵 Skill Match    35%
🟣 TF-IDF Sim     30%
🟢 Category Cov   15%
🟡 Experience     12%
🔴 Education       8%
```

---

## 📊 Visualizations & Outputs

### Recruiter Dashboard (Step 10)
A 6-panel 24×30 inch figure showing:
- **Top-15 composite scores** (color-coded by status)
- **Score distribution** histogram with normal curve
- **Experience vs. composite score** scatter plot
- **Skill category coverage** grouped bar chart
- **Education breakdown** pie chart
- **Top missing skills** horizontal bar chart

### Skill Gap Heatmap (Step 11)
A matrix where rows are the top 15 candidates and columns are all required skills. Green ✓ = present, red ✗ = absent. A side bar shows overall coverage % per candidate.

### Analytics Deep-Dive (Step 12)
Four charts in a 2×2 grid:
- Average & top score per resume category
- Number of resumes per category
- Score percentile distribution curve
- Required skill category frequencies

### Role Comparison (Step 13)
Grouped bar chart showing top score and mean score for all 6 job roles.

---

## 📦 Datasets

The notebook supports **three optional Kaggle datasets** uploaded in Step 2. If none are uploaded, it falls back to 100 fully synthetic resumes + 6 synthetic JDs.

| Dataset | Kaggle Link | Use |
|---------|-------------|-----|
| Resume Dataset | [gauravduttakiit/resume-dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset) | Resume pool to screen |
| Job Descriptions | [ravindrasinghrana/job-description-dataset](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset) | JD profiles for each role |
| Monster.com Jobs | [promptcloud/us-jobs-on-monstercom](https://www.kaggle.com/datasets/promptcloud/us-jobs-on-monstercom) | Additional JD source |

**The synthetic fallback generates:**
- 100 resumes across 10 categories (Data Science, ML Engineering, Full Stack Dev, DevOps, Data Analytics, Accounting, Marketing, Healthcare, Education, Cybersecurity)
- Realistic names, emails, companies, locations, EDUCATION levels, and years of experience
- Skills drawn probabilistically based on role templates

---

## 🛠️ Tech Stack

### Python Notebook
| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | 2.2.2 | Data manipulation, results DataFrame |
| `numpy` | 2.0.2 | Numerical ops, matrix operations |
| `scikit-learn` | latest | `TfidfVectorizer`, `cosine_similarity` |
| `matplotlib` | latest | All visualization panels |
| `pdfplumber` | latest | PDF text extraction (personal analyzer) |
| `re` | stdlib | Regex for skill/experience extraction |

### HTML Demo
| Technology | Purpose |
|------------|---------|
| Vanilla JavaScript (ES6+) | Full ML pipeline ported to browser |
| Canvas API | Radar chart drawing |
| FileReader API | File upload handling |
| CSS Custom Properties | Dark theme design system |
| Google Fonts (Syne, DM Mono, DM Sans) | Typography |

No npm, no build step, no backend — just one `.html` file.

---

## 📋 Results & Sample Output

Running the pipeline on 100 synthetic resumes for **Data Scientist** role (with 33 required JD skills):

```
══════════════════════════════════════════════════════════════════
    PIPELINE COMPLETE — Data Scientist
══════════════════════════════════════════════════════════════════
  Total Screened  : 100
   Recommended  : 18  (≥ 70%)
    Potential   : 31  (50–69%)
   Not Rec.     : 51  (< 50%)
  Highest Score  : 88.4%
  Mean Score     : 48.2%  |  Std dev: 18.6%
══════════════════════════════════════════════════════════════════
```

**Top 3 Candidates:**

| Rank | Name | Score | Status | Skills |
|------|------|-------|--------|--------|
| #1 | Alice Johnson | 88.4% | ✅ Highly Recommended | 31/33 matched |
| #2 | Emma Wilson | 82.1% | ✅ Highly Recommended | 28/33 matched |
| #3 | Noah Anderson | 75.6% | ✅ Highly Recommended | 25/33 matched |

---

## 🔮 Future Improvements

- [ ] **BERT/Sentence-Transformers** — replace TF-IDF with semantic embeddings for better similarity
- [ ] **Named Entity Recognition** — extract candidate name, email, phone automatically from resume text
- [ ] **PDF parsing** — proper `.pdf` support in the HTML demo (currently limited to text-based PDFs)
- [ ] **Custom JD input** — allow users to paste their own job description instead of using preset roles
- [ ] **Explainability panel** — show exactly which sentences triggered each score
- [ ] **Flask/FastAPI web app** — deploy the Python backend as a REST API
- [ ] **Database integration** — store and query historical screening results
- [ ] **Resume anonymization** — strip names/emails for bias-reduced screening

---

## 👤 Author

Built as a final project for a Data Science / Machine Learning course.

- **Notebook:** Python pipeline with full visualizations and synthetic data generation
- **HTML Demo:** JavaScript port for zero-dependency browser usage

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).

---

<div align="center">

⭐ **If this project helped you, please give it a star!** ⭐

</div>
