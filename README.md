# ResuMatch — AI Resume Matcher
### OpenCV + NLP Powered Resume Screening System

---

## 📌 Overview

ResuMatch is a full-stack resume screening tool that uses:
- **OpenCV** — Document image preprocessing (deskew, denoise, binarize, layout analysis)
- **TF-IDF NLP** — Semantic similarity scoring between resume & job description
- **Skill Extraction** — 60+ tech keywords matched with JD requirements
- **Section Parsing** — Regex-based section detection (Skills, Experience, Education, etc.)
- **Batch Ranking** — Rank multiple resumes against one job description

---

## 🗂️ Project Structure

```
resume_matcher/
├── resume_processor.py   # Core engine: OpenCV + NLP + Matcher
├── app.py                # Flask web server + REST API
├── cli.py                # Command-line interface
├── test_matcher.py       # End-to-end test suite
├── templates/
│   └── index.html        # Web UI (dark theme)
├── uploads/              # Temp file storage
└── requirements.txt
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Tests
```bash
python test_matcher.py
```

### 3. Web Interface
```bash
python app.py
# Open: http://localhost:5000
```

### 4. CLI Usage
```bash
# Single resume
python cli.py --resume john_doe.pdf --jd "Python ML engineer 3+ years"

# Match from JD file
python cli.py --resume resume.docx --jd-file job.txt

# Bulk rank a directory
python cli.py --resume-dir ./candidates/ --jd-file jd.txt --top 5

# Save JSON output
python cli.py --resume resume.pdf --jd-file jd.txt --output result.json
```

---

## 🔌 REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/match-single` | Match one resume file |
| POST | `/api/match-bulk` | Rank multiple resumes |
| POST | `/api/analyze-text` | Match pasted text (no file upload) |
| POST | `/api/extract-skills` | Extract skills from text |
| GET  | `/api/health` | Health check |

### Example: Match Single Resume
```bash
curl -X POST http://localhost:5000/api/match-single \
  -F "resume=@resume.pdf" \
  -F "job_description=Python developer 3+ years ML experience"
```

### Example: Analyze Text (JSON)
```bash
curl -X POST http://localhost:5000/api/analyze-text \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "...your resume...", "job_description": "...JD..."}'
```

---

## 🧠 Scoring Algorithm

The match score is computed from **4 signals**:

| Signal | Weight | Method |
|--------|--------|--------|
| TF-IDF Cosine Similarity | 35% | Bigram TF-IDF on full text |
| Skill Overlap | 35% | Keyword matching from 60+ skills |
| Section-Weighted Score | 20% | Skills(35%), Exp(30%), Edu(15%), etc. |
| Experience Match | 10% | Year extraction + ratio scoring |

**Grade thresholds:**
- ≥75% → Excellent Match 🟢
- ≥55% → Good Match 🔵
- ≥35% → Moderate Match 🟡
- <35%  → Low Match 🔴

---

## 🖼️ OpenCV Pipeline

For image-based resumes (PNG, JPG) and PDF first pages:

1. **Grayscale conversion** — `cv2.cvtColor`
2. **Denoising** — `cv2.fastNlMeansDenoising`
3. **Binarization** — Otsu's thresholding
4. **Deskew** — Angle correction via `cv2.minAreaRect`
5. **Layout Analysis** — Projection profiles + contour detection
   - Estimates line count, text density, column detection

---

## 📦 Supported File Types

| Format | Extraction Method |
|--------|-------------------|
| PDF    | pdfplumber |
| DOCX   | python-docx |
| TXT    | direct read |
| PNG/JPG | OpenCV preprocess + pytesseract (optional) |

---

## 🔧 Optional: OCR for Scanned Resumes

To enable OCR on scanned image resumes:
```bash
sudo apt-get install tesseract-ocr
pip install pytesseract
```

---

## 🧩 Key Classes

```python
from resume_processor import (
    OpenCVProcessor,   # Image preprocessing
    TextExtractor,     # PDF/DOCX/Image → text
    NLPProcessor,      # Tokenize, skills, contact, sections
    ResumeParser,      # Full parse pipeline
    MatcherEngine,     # Score a resume vs JD
    BatchRanker,       # Rank N resumes vs JD
)

# Quick usage
parser = ResumeParser()
engine = MatcherEngine()

resume = parser.parse("resume.pdf")
result = engine.match(resume, "job description text")
print(result['overall_score'], result['grade'])
```
