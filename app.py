"""
app.py — ResuMatch Flask Web Server
"""
import os, uuid
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from resume_processor import ResumeParser, MatcherEngine, BatchRanker, NLPProcessor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

UPLOAD_FOLDER  = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXT    = {'pdf', 'docx', 'doc', 'txt', 'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

parser = ResumeParser()
engine = MatcherEngine()
ranker = BatchRanker()


def ok_file(f):
    return '.' in f and f.rsplit('.', 1)[1].lower() in ALLOWED_EXT


def save(file):
    fn = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    path = os.path.join(UPLOAD_FOLDER, fn)
    file.save(path)
    return path


@app.route('/')
def index():
    with open(os.path.join(os.path.dirname(__file__), 'templates', 'index.html'),
              encoding='utf-8') as f:
        return f.read()


@app.route('/api/match-single', methods=['POST'])
def match_single():
    if 'resume' not in request.files:
        return jsonify({'error': 'No resume uploaded'}), 400
    file = request.files['resume']
    jd   = request.form.get('job_description', '').strip()
    if not jd:
        return jsonify({'error': 'Job description required'}), 400
    if not ok_file(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400
    path = save(file)
    try:
        resume = parser.parse(path)
        result = engine.match(resume, jd)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(path): os.remove(path)


@app.route('/api/match-text', methods=['POST'])
def match_text():
    data = request.get_json() or {}
    resume_text = data.get('resume_text', '').strip()
    jd          = data.get('job_description', '').strip()
    if not resume_text: return jsonify({'error': 'Resume text required'}), 400
    if not jd:          return jsonify({'error': 'Job description required'}), 400
    path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_resume.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(resume_text)
    try:
        resume = parser.parse(path)
        result = engine.match(resume, jd)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(path): os.remove(path)


@app.route('/api/match-bulk', methods=['POST'])
def match_bulk():
    files = request.files.getlist('resumes')
    jd    = request.form.get('job_description', '').strip()
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No resumes uploaded'}), 400
    if not jd: return jsonify({'error': 'Job description required'}), 400
    paths = []
    for f in files:
        if f and ok_file(f.filename):
            paths.append(save(f))
    if not paths: return jsonify({'error': 'No valid files'}), 400
    try:
        results = ranker.rank(paths, jd)
        return jsonify({'success': True, 'results': results, 'total': len(results)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        for p in paths:
            if os.path.exists(p): os.remove(p)


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print("=" * 50)
    print("  ResuMatch Server Running")
    print("  Open: http://localhost:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
