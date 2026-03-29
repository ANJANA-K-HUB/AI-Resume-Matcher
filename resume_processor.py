"""
resume_processor.py
Core NLP + OpenCV engine for ResuMatch.
Extracts resume data, scores against a JD, and generates
actionable improvement suggestions.
"""

import re
import os
import cv2
import numpy as np
import pdfplumber
from PIL import Image
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime

# ─────────────────────────────────────────────────────────────
#  SKILL TAXONOMY  (category → list of keywords)
# ─────────────────────────────────────────────────────────────
SKILL_TAXONOMY = {
    "Programming Languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "c", "ruby",
        "php", "swift", "kotlin", "golang", "go", "rust", "scala", "r", "matlab",
        "perl", "bash", "shell", "powershell",
    ],
    "Web & Frontend": [
        "html", "css", "react", "angular", "vue", "next.js", "node.js", "express",
        "django", "flask", "fastapi", "spring", "jquery", "bootstrap", "tailwind",
        "sass", "graphql", "rest", "rest api", "api",
    ],
    "Data & ML": [
        "machine learning", "deep learning", "nlp", "natural language processing",
        "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn",
        "sklearn", "opencv", "pandas", "numpy", "scipy", "matplotlib", "seaborn",
        "data science", "data analysis", "data engineering", "feature engineering",
        "model deployment", "mlops", "hugging face", "transformers", "bert", "llm",
    ],
    "Databases": [
        "sql", "mysql", "postgresql", "postgres", "mongodb", "nosql", "redis",
        "sqlite", "oracle", "cassandra", "dynamodb", "elasticsearch", "firebase",
    ],
    "Cloud & DevOps": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
        "jenkins", "ci/cd", "terraform", "ansible", "linux", "git", "github",
        "gitlab", "bitbucket", "devops", "microservices", "serverless",
    ],
    "Data Tools": [
        "spark", "hadoop", "kafka", "airflow", "dbt", "tableau", "power bi",
        "looker", "excel", "etl", "data pipeline", "data warehouse",
    ],
    "Soft Skills": [
        "leadership", "communication", "teamwork", "problem solving", "agile",
        "scrum", "project management", "critical thinking", "collaboration",
        "presentation", "mentoring", "stakeholder management",
    ],
}

ALL_SKILLS = {skill for skills in SKILL_TAXONOMY.values() for skill in skills}

# Section detection patterns
SECTION_PATTERNS = {
    "skills":          r"\b(skills|technologies|tools|expertise|competencies|technical|proficiencies)\b",
    "experience":      r"\b(experience|employment|work history|career|professional|internship)\b",
    "education":       r"\b(education|academic|university|college|degree|bachelor|master|phd|b\.?tech|m\.?tech|b\.?sc|m\.?sc)\b",
    "projects":        r"\b(projects|portfolio|work samples|assignments)\b",
    "certifications":  r"\b(certifications|certificates|licenses|courses|training)\b",
    "summary":         r"\b(summary|objective|profile|about|overview)\b",
    "achievements":    r"\b(achievements|accomplishments|awards|honors|recognition)\b",
}

EMAIL_RE    = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
PHONE_RE    = r'(\+?\d[\d\s\-().]{7,}\d)'
LINKEDIN_RE = r'linkedin\.com/in/[\w\-]+'
GITHUB_RE   = r'github\.com/[\w\-]+'


# ─────────────────────────────────────────────────────────────
#  OPENCV PROCESSOR
# ─────────────────────────────────────────────────────────────
class OpenCVProcessor:
    """Image preprocessing pipeline for scanned/image resumes."""

    @staticmethod
    def preprocess(image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return OpenCVProcessor._deskew(binary)

    @staticmethod
    def _deskew(img: np.ndarray) -> np.ndarray:
        coords = np.column_stack(np.where(img < 128))
        if len(coords) < 10:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) < 0.5:
            return img
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def layout_analysis(image_path: str) -> dict:
        """Detect text density, columns, and content regions."""
        img = cv2.imread(image_path)
        if img is None:
            return {}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        h_proj = np.sum(binary, axis=1)
        v_proj = np.sum(binary, axis=0)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blocks = [c for c in contours if cv2.contourArea(c) > 500]
        text_cols = np.where(v_proj > np.max(v_proj) * 0.05)[0]
        return {
            "text_lines":    int(np.sum(h_proj > np.max(h_proj) * 0.1)),
            "content_blocks": len(blocks),
            "is_multi_column": len(text_cols) > 0 and (text_cols[-1] - text_cols[0]) > img.shape[1] * 0.6,
            "text_density":  float(np.mean(binary) / 255),
        }

    @staticmethod
    def pdf_to_image(pdf_path: str) -> str | None:
        out = pdf_path.replace(".pdf", "_p0_cv.png")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if pdf.pages:
                    pdf.pages[0].to_image(resolution=180).save(out)
            return out if os.path.exists(out) else None
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────
#  TEXT EXTRACTOR
# ─────────────────────────────────────────────────────────────
class TextExtractor:
    @staticmethod
    def extract(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            return TextExtractor._pdf(path)
        elif ext in (".docx", ".doc"):
            return TextExtractor._docx(path)
        elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
            return TextExtractor._image(path)
        elif ext == ".txt":
            with open(path, encoding="utf-8", errors="ignore") as f:
                return f.read()
        raise ValueError(f"Unsupported file: {ext}")

    @staticmethod
    def _pdf(path):
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        return text

    @staticmethod
    def _docx(path):
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)

    @staticmethod
    def _image(path):
        try:
            import pytesseract
            processed = OpenCVProcessor.preprocess(path)
            return pytesseract.image_to_string(Image.fromarray(processed))
        except ImportError:
            return "[Install pytesseract for image OCR support]"


# ─────────────────────────────────────────────────────────────
#  NLP PROCESSOR
# ─────────────────────────────────────────────────────────────
class NLPProcessor:
    STOP = {
        "a","an","the","and","or","but","in","on","at","to","for","of","with",
        "by","from","is","are","was","were","be","been","have","has","had","do",
        "does","did","will","would","can","could","may","might","shall","should",
        "i","we","you","he","she","it","they","my","our","your","their","this",
        "that","these","those","not","no","so","if","as","into","through","also",
        "about","up","just","while","when","where","which","who","what","how",
        "all","both","each","few","more","most","other","such","here","there",
        "then","than","after","before","during","over","under","again","once",
    }

    @staticmethod
    def clean(text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s\.\+\#\/]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def tokens(text: str) -> list:
        raw = re.findall(r'\b[\w][\w\.\+\#]*\b', text.lower())
        return [t for t in raw if len(t) > 1 and t not in NLPProcessor.STOP]

    @staticmethod
    def extract_skills(text: str) -> dict:
        """Returns {category: [found skills]} and flat list."""
        tl = text.lower()
        found_by_cat = {}
        for cat, skills in SKILL_TAXONOMY.items():
            matched = [s for s in skills if re.search(r'\b' + re.escape(s) + r'\b', tl)]
            if matched:
                found_by_cat[cat] = matched
        flat = [s for lst in found_by_cat.values() for s in lst]
        return {"by_category": found_by_cat, "flat": flat}

    @staticmethod
    def extract_contact(text: str) -> dict:
        emails    = re.findall(EMAIL_RE, text)
        phones    = re.findall(PHONE_RE, text)
        linkedins = re.findall(LINKEDIN_RE, text, re.I)
        githubs   = re.findall(GITHUB_RE, text, re.I)
        return {
            "email":    emails[0] if emails else None,
            "phone":    phones[0].strip() if phones else None,
            "linkedin": linkedins[0] if linkedins else None,
            "github":   githubs[0] if githubs else None,
        }

    @staticmethod
    def extract_name(text: str) -> str:
        for line in text.split('\n')[:6]:
            line = line.strip()
            if 2 <= len(line.split()) <= 4 and re.match(r'^[A-Z][a-z]', line):
                if not re.search(r'[@\d/\\]', line) and len(line) < 50:
                    return line
        return "Candidate"

    @staticmethod
    def extract_education(text: str) -> list:
        pattern = r'(?:B\.?Tech|M\.?Tech|B\.?E|M\.?E|B\.?Sc|M\.?Sc|MBA|PhD|Bachelor|Master|Associate|Diploma)[^\n]{0,80}'
        return re.findall(pattern, text, re.IGNORECASE)

    @staticmethod
    def extract_experience_years(text: str) -> float:
        years = []
        for p in [r'(\d+)\+?\s*years?\s+(?:of\s+)?(?:experience|exp)',
                  r'(?:experience|exp)[^\d]*(\d+)\+?\s*years?']:
            years += [int(m) for m in re.findall(p, text, re.I)]
        cy = datetime.datetime.now().year
        for s, e in re.findall(r'(\d{4})\s*[-–]\s*(\d{4}|[Pp]resent)', text):
            ey = cy if e.lower() == 'present' else int(e)
            diff = ey - int(s)
            if 0 < diff < 45:
                years.append(diff)
        return round(sum(years) / max(len(years), 1), 1) if years else 0

    @staticmethod
    def parse_sections(text: str) -> dict:
        sections = {k: [] for k in SECTION_PATTERNS}
        current = "general"
        buf = {"general": []}
        for line in text.split('\n'):
            ll = line.lower().strip()
            hit = False
            for sec, pat in SECTION_PATTERNS.items():
                if re.search(pat, ll) and len(ll) < 70:
                    current = sec
                    buf.setdefault(sec, [])
                    hit = True
                    break
            if not hit:
                buf.setdefault(current, []).append(line)
        return {k: ' '.join(v) for k, v in buf.items() if k in sections}

    @staticmethod
    def extract_quantified_achievements(text: str) -> list:
        """Find sentences/phrases that contain numbers — signals measurable impact."""
        sentences = re.split(r'[.\n]', text)
        hits = []
        for s in sentences:
            s = s.strip()
            if re.search(r'\d+', s) and len(s) > 20:
                hits.append(s)
        return hits[:8]

    @staticmethod
    def detect_action_verbs(text: str) -> list:
        strong = [
            "led","built","developed","designed","implemented","architected","managed",
            "delivered","reduced","increased","improved","created","launched","automated",
            "deployed","optimized","scaled","mentored","established","generated",
        ]
        tl = text.lower()
        return [v for v in strong if re.search(r'\b' + v + r'\b', tl)]


# ─────────────────────────────────────────────────────────────
#  RESUME PARSER
# ─────────────────────────────────────────────────────────────
class ResumeParser:
    def parse(self, path: str) -> dict:
        raw  = TextExtractor.extract(path)
        nlp  = NLPProcessor()
        ext  = os.path.splitext(path)[1].lower()

        layout = {}
        if ext in (".png", ".jpg", ".jpeg"):
            try:
                layout = OpenCVProcessor.layout_analysis(path)
            except Exception:
                pass
        elif ext == ".pdf":
            try:
                img = OpenCVProcessor.pdf_to_image(path)
                if img:
                    layout = OpenCVProcessor.layout_analysis(img)
                    os.remove(img)
            except Exception:
                pass

        skills_data  = nlp.extract_skills(raw)
        sections     = nlp.parse_sections(raw)
        contact      = nlp.extract_contact(raw)
        education    = nlp.extract_education(raw)
        exp_years    = nlp.extract_experience_years(raw)
        name         = nlp.extract_name(raw)
        achievements = nlp.extract_quantified_achievements(raw)
        action_verbs = nlp.detect_action_verbs(raw)

        has_summary       = bool(sections.get("summary", "").strip())
        has_achievements  = bool(sections.get("achievements", "").strip()) or len(achievements) > 0
        has_certifications= bool(sections.get("certifications", "").strip())
        has_projects      = bool(sections.get("projects", "").strip())

        return {
            "file_name":        os.path.basename(path),
            "raw_text":         raw,
            "clean_text":       nlp.clean(raw),
            "name":             name,
            "contact":          contact,
            "skills":           skills_data["flat"],
            "skills_by_cat":    skills_data["by_category"],
            "education":        education,
            "experience_years": exp_years,
            "sections":         sections,
            "achievements":     achievements,
            "action_verbs":     action_verbs,
            "has_summary":      has_summary,
            "has_achievements": has_achievements,
            "has_certifications": has_certifications,
            "has_projects":     has_projects,
            "word_count":       len(raw.split()),
            "layout":           layout,
        }


# ─────────────────────────────────────────────────────────────
#  IMPROVEMENT ENGINE
# ─────────────────────────────────────────────────────────────
class ImprovementEngine:
    """
    Generates specific, actionable suggestions for improving the resume
    so it better matches the given job description.
    """

    @staticmethod
    def generate(resume: dict, jd_text: str, match_result: dict) -> dict:
        suggestions = {
            "critical":   [],   # Must fix — directly affects score
            "recommended":[],   # Should fix — improves chances
            "optional":   [],   # Nice to have
        }
        jd_lower = jd_text.lower()
        nlp = NLPProcessor()

        # 1. Missing skills from JD
        jd_skills_data = nlp.extract_skills(jd_text)
        jd_skills_flat = jd_skills_data["flat"]
        resume_skills  = set(resume["skills"])
        missing_skills = [s for s in jd_skills_flat if s not in resume_skills]

        if missing_skills:
            by_cat = {}
            for cat, skills in SKILL_TAXONOMY.items():
                ms = [s for s in missing_skills if s in skills]
                if ms:
                    by_cat[cat] = ms
            for cat, ms in by_cat.items():
                suggestions["critical"].append({
                    "icon": "🔧",
                    "title": f"Add missing {cat} skills",
                    "detail": f"The JD requires: {', '.join(ms[:6])}. Add these to your Skills section if you have experience with them.",
                })

        # 2. Experience gap
        jd_exp = nlp.extract_experience_years(jd_text)
        resume_exp = resume["experience_years"]
        if jd_exp > 0 and resume_exp < jd_exp:
            if resume_exp == 0:
                suggestions["critical"].append({
                    "icon": "📅",
                    "title": "Mention your years of experience explicitly",
                    "detail": f"The JD asks for {jd_exp}+ years. Add a clear statement like '3 years of experience in...' to your summary or experience section.",
                })
            else:
                suggestions["recommended"].append({
                    "icon": "📅",
                    "title": f"Experience gap: JD needs {jd_exp}+ years, resume shows {resume_exp}",
                    "detail": "Highlight all relevant experience, including freelance, internships, academic projects, and part-time work to bridge the gap.",
                })

        # 3. No professional summary
        if not resume["has_summary"]:
            suggestions["critical"].append({
                "icon": "📝",
                "title": "Add a Professional Summary",
                "detail": "A 3–4 line summary at the top dramatically increases ATS scores. Tailor it to match the JD's language and key requirements.",
            })
        else:
            # Check if summary contains JD keywords
            summary_text = resume["sections"].get("summary", "")
            jd_tokens = set(nlp.tokens(nlp.clean(jd_text)))
            summary_tokens = set(nlp.tokens(nlp.clean(summary_text)))
            overlap = jd_tokens & summary_tokens
            if len(overlap) < 4:
                suggestions["recommended"].append({
                    "icon": "✏️",
                    "title": "Rewrite your Summary to mirror the JD",
                    "detail": "Your summary doesn't reflect the JD's language. Include the job title, key skills, and years of experience that the JD mentions.",
                })

        # 4. No quantified achievements
        if len(resume["achievements"]) < 2:
            suggestions["critical"].append({
                "icon": "📊",
                "title": "Add measurable achievements with numbers",
                "detail": 'Use metrics to prove impact. Instead of "improved performance", write "reduced load time by 40%" or "led team of 6 engineers".',
            })

        # 5. Weak action verbs
        if len(resume["action_verbs"]) < 3:
            suggestions["recommended"].append({
                "icon": "💬",
                "title": "Use stronger action verbs",
                "detail": "Start bullet points with impactful verbs like: Built, Led, Designed, Reduced, Increased, Automated, Deployed, Optimized.",
            })

        # 6. No certifications
        if not resume["has_certifications"]:
            # Check if JD mentions certifications
            if re.search(r'\b(certif|certified|aws|azure|gcp|pmp|scrum|cissp)\b', jd_lower):
                suggestions["recommended"].append({
                    "icon": "🏆",
                    "title": "Add relevant certifications",
                    "detail": "The JD hints at certifications. Add any relevant ones (AWS, Azure, Scrum, etc.). If you have none, consider getting one that matches the JD.",
                })
            else:
                suggestions["optional"].append({
                    "icon": "🏆",
                    "title": "Consider adding certifications",
                    "detail": "Certifications boost credibility. Relevant online certifications from Coursera, AWS, or Google can strengthen your profile.",
                })

        # 7. No projects section
        if not resume["has_projects"]:
            if re.search(r'\b(project|portfolio|github|demo)\b', jd_lower):
                suggestions["recommended"].append({
                    "icon": "🗂️",
                    "title": "Add a Projects section",
                    "detail": "The JD values hands-on work. Add 2–3 relevant projects with tech stack, your role, and outcomes. Link to GitHub if possible.",
                })

        # 8. Missing contact details
        contact = resume["contact"]
        missing_contact = []
        if not contact.get("email"):
            missing_contact.append("email")
        if not contact.get("linkedin"):
            missing_contact.append("LinkedIn URL")
        if not contact.get("github") and re.search(r'\b(github|open.?source)\b', jd_lower):
            missing_contact.append("GitHub URL")
        if missing_contact:
            suggestions["recommended"].append({
                "icon": "🔗",
                "title": f"Add missing contact info: {', '.join(missing_contact)}",
                "detail": "Recruiters expect email and LinkedIn at minimum. GitHub is important for technical roles.",
            })

        # 9. Resume too short or too long
        wc = resume["word_count"]
        if wc < 200:
            suggestions["critical"].append({
                "icon": "📄",
                "title": "Resume is too short",
                "detail": f"Your resume has only {wc} words. A good resume has 400–800 words. Expand your experience descriptions and add more detail.",
            })
        elif wc > 900:
            suggestions["optional"].append({
                "icon": "✂️",
                "title": "Consider trimming your resume",
                "detail": f"Your resume has {wc} words. Keep it focused — aim for 1 page (fresh) or 2 pages (senior). Remove outdated or irrelevant content.",
            })

        # 10. JD-specific keyword gaps (non-skill)
        jd_keywords = set(nlp.tokens(nlp.clean(jd_text)))
        resume_keywords = set(nlp.tokens(resume["clean_text"]))
        important_jd_words = [
            w for w in jd_keywords
            if w not in resume_keywords and len(w) > 4
            and w not in NLPProcessor.STOP
            and not w.isdigit()
        ]
        if len(important_jd_words) > 5:
            sample = list(important_jd_words)[:6]
            suggestions["recommended"].append({
                "icon": "🔑",
                "title": "Use more JD-specific keywords",
                "detail": f"Your resume is missing key terms from the JD. Try naturally including: {', '.join(sample)}.",
            })

        # 11. No GitHub for tech roles
        if not contact.get("github"):
            is_tech_jd = any(k in jd_lower for k in ["software", "developer", "engineer", "coding", "programming"])
            if is_tech_jd:
                suggestions["optional"].append({
                    "icon": "💻",
                    "title": "Add your GitHub profile",
                    "detail": "For tech roles, a GitHub profile with active projects can significantly strengthen your application.",
                })

        return {
            "critical":    suggestions["critical"],
            "recommended": suggestions["recommended"],
            "optional":    suggestions["optional"],
            "total":       sum(len(v) for v in suggestions.values()),
        }


# ─────────────────────────────────────────────────────────────
#  MATCHER ENGINE  — human-friendly scoring
#
#  Four criteria shown to the user:
#    1. Skills Match       (45%)  — % of JD-required skills found in resume
#    2. Experience Match   (25%)  — resume years vs JD required years
#    3. Keywords Match     (20%)  — important JD words present in resume
#    4. Resume Quality     (10%)  — completeness: summary, achievements, etc.
# ─────────────────────────────────────────────────────────────
class MatcherEngine:
    def __init__(self):
        # TF-IDF used internally for keyword match only, never shown by name
        self.vec = TfidfVectorizer(ngram_range=(1, 2), max_features=8000,
                                   sublinear_tf=True, stop_words="english")

    # ── 1. SKILLS MATCH (45%) ──────────────────────────────────
    def _skill_score(self, resume_skills, jd_text):
        nlp = NLPProcessor()
        jd_skills = nlp.extract_skills(jd_text)["flat"]
        if not jd_skills:
            matched = [s for s in resume_skills if s in jd_text.lower()]
            return len(matched) / max(len(resume_skills), 1), matched, []
        matched = [s for s in resume_skills if s in jd_skills]
        missing = [s for s in jd_skills if s not in resume_skills]
        score   = len(matched) / max(len(jd_skills), 1)
        return score, matched, missing

    # ── 2. EXPERIENCE MATCH (25%) ──────────────────────────────
    def _exp_score(self, resume_exp, jd_text):
        nlp    = NLPProcessor()
        jd_exp = nlp.extract_experience_years(jd_text)
        if jd_exp == 0:
            # JD doesn't specify — give full marks
            return 1.0, jd_exp
        if resume_exp >= jd_exp:
            return 1.0, jd_exp
        if resume_exp > 0:
            # Partial credit — proportional
            return round(resume_exp / jd_exp, 3), jd_exp
        # No experience mentioned — give 30% base
        return 0.3, jd_exp

    # ── 3. KEYWORDS MATCH (20%) ────────────────────────────────
    # How well the resume language covers JD terminology
    # (uses TF-IDF internally but labelled as "Keywords Match")
    def _keyword_score(self, resume_clean: str, jd_text: str) -> float:
        try:
            m = self.vec.fit_transform([resume_clean, jd_text.lower()])
            return float(cosine_similarity(m[0:1], m[1:2])[0][0])
        except Exception:
            return 0.0

    # ── 4. RESUME QUALITY (10%) ────────────────────────────────
    # Completeness check — does resume have all key sections?
    def _quality_score(self, resume: dict) -> float:
        score = 0.0
        if resume.get("has_summary"):          score += 0.25
        if resume.get("has_projects"):         score += 0.20
        if resume.get("has_certifications"):   score += 0.15
        if len(resume.get("achievements", [])) >= 2: score += 0.25
        if len(resume.get("action_verbs", [])) >= 3: score += 0.15
        return min(score, 1.0)

    # ── OVERALL MATCH ──────────────────────────────────────────
    def match(self, resume: dict, jd_text: str) -> dict:
        sk_sc, matched, missing = self._skill_score(resume["skills"], jd_text)
        exp_sc, jd_exp          = self._exp_score(resume["experience_years"], jd_text)
        kw_sc                   = self._keyword_score(resume["clean_text"], jd_text)
        ql_sc                   = self._quality_score(resume)

        # Weighted final score
        overall = min(
            sk_sc  * 0.45 +
            exp_sc * 0.25 +
            kw_sc  * 0.20 +
            ql_sc  * 0.10,
            1.0
        )
        pct = round(overall * 100, 1)

        if pct >= 75:
            grade, gcolor = "Excellent Match", "#2d6a4f"
        elif pct >= 55:
            grade, gcolor = "Good Match",      "#1e4070"
        elif pct >= 35:
            grade, gcolor = "Moderate Match",  "#7a5c00"
        else:
            grade, gcolor = "Low Match",       "#8b2020"

        improvements = ImprovementEngine.generate(resume, jd_text, {
            "overall_score": pct,
            "skill_score":   round(sk_sc * 100, 1),
        })

        return {
            "overall_score":     pct,
            "skill_score":       round(sk_sc  * 100, 1),
            "experience_score":  round(exp_sc  * 100, 1),
            "keyword_score":     round(kw_sc   * 100, 1),
            "quality_score":     round(ql_sc   * 100, 1),
            "grade":             grade,
            "grade_color":       gcolor,
            "matched_skills":    matched,
            "missing_skills":    missing,
            "jd_exp_required":   jd_exp,
            "resume_skills":     resume["skills"],
            "skills_by_cat":     resume["skills_by_cat"],
            "name":              resume["name"],
            "contact":           resume["contact"],
            "experience_years":  resume["experience_years"],
            "education":         resume["education"],
            "achievements":      resume["achievements"],
            "word_count":        resume["word_count"],
            "file_name":         resume["file_name"],
            "improvements":      improvements,
            "has_summary":       resume["has_summary"],
            "has_projects":      resume["has_projects"],
            "has_certifications":resume["has_certifications"],
        }


# ─────────────────────────────────────────────────────────────
#  BATCH RANKER
# ─────────────────────────────────────────────────────────────
class BatchRanker:
    def __init__(self):
        self.parser = ResumeParser()
        self.engine = MatcherEngine()

    def rank(self, paths: list, jd_text: str) -> list:
        results = []
        for p in paths:
            try:
                r = self.parser.parse(p)
                m = self.engine.match(r, jd_text)
                results.append(m)
            except Exception as e:
                results.append({"file_name": os.path.basename(p),
                                 "error": str(e), "overall_score": 0,
                                 "grade": "Error", "grade_color": "#999"})
        results.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1
        return results
