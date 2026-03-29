#!/usr/bin/env python3
"""
ResuMatch CLI — Command-line resume matching tool
Usage:
  python cli.py --resume path/to/resume.pdf --jd "Job description text"
  python cli.py --resume-dir ./resumes/ --jd-file job.txt --top 5
  python cli.py --help
"""

import argparse
import os
import sys
import json

# Add parent dir to path
sys.path.insert(0, os.path.dirname(__file__))
from resume_processor import ResumeParser, MatcherEngine, BatchRanker


def color(text, code):
    return f"\033[{code}m{text}\033[0m"

def green(t): return color(t, "92")
def yellow(t): return color(t, "93")
def red(t): return color(t, "91")
def blue(t): return color(t, "94")
def bold(t): return color(t, "1")
def cyan(t): return color(t, "96")
def dim(t): return color(t, "2")


def score_color(score):
    if score >= 75: return green
    if score >= 55: return blue
    if score >= 35: return yellow
    return red


def print_banner():
    print()
    print(bold(cyan("  ██████╗ ███████╗███████╗██╗   ██╗███╗   ███╗ █████╗ ████████╗ ██████╗██╗  ██╗")))
    print(bold(cyan("  ██╔══██╗██╔════╝██╔════╝██║   ██║████╗ ████║██╔══██╗╚══██╔══╝██╔════╝██║  ██║")))
    print(bold(cyan("  ██████╔╝█████╗  ███████╗██║   ██║██╔████╔██║███████║   ██║   ██║     ███████║")))
    print(bold(cyan("  ██╔══██╗██╔══╝  ╚════██║██║   ██║██║╚██╔╝██║██╔══██║   ██║   ██║     ██╔══██║")))
    print(bold(cyan("  ██║  ██║███████╗███████║╚██████╔╝██║ ╚═╝ ██║██║  ██║   ██║   ╚██████╗██║  ██║")))
    print(bold(cyan("  ╚═╝  ╚═╝╚══════╝╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝")))
    print(dim("  OpenCV + NLP Resume Matching Engine"))
    print()


def print_result(r, show_rank=True, verbose=False):
    sc_fn = score_color(r.get('overall_score', 0))
    rank_str = f"#{r['rank']} " if show_rank and 'rank' in r else ""

    print(f"\n  {'─'*60}")
    print(f"  {bold(rank_str + (r.get('name') or r.get('file_name', 'Unknown')))}")
    print(f"  {dim('File: ' + r.get('file_name', ''))}")
    print()

    score = r.get('overall_score', 0)
    print(f"  {bold('Overall Match:')} {sc_fn(f'{score:.1f}%')}  →  {bold(r.get('grade', ''))}")
    print()

    # Score breakdown
    breakdown = [
        ("TF-IDF Similarity", r.get('tfidf_score', 0)),
        ("Skill Match",       r.get('skill_score', 0)),
        ("Section Score",     r.get('section_score', 0)),
        ("Experience",        r.get('experience_score', 0)),
    ]
    for label, val in breakdown:
        bar_len = int(val / 4)
        bar = "█" * bar_len + "░" * (25 - bar_len)
        fn = score_color(val)
        print(f"  {label:<22} {fn(bar)} {val:.1f}%")

    print()

    # Skills
    matched = r.get('matched_skills', [])
    missing = r.get('missing_skills', [])
    if matched:
        print(f"  {green('✓ Matched:')} {', '.join(matched[:10])}")
    if missing:
        print(f"  {red('✗ Missing:')} {', '.join(missing[:8])}")

    # Info
    infos = []
    if r.get('experience_years'):
        infos.append(f"⏳ {r['experience_years']}y exp")
    if r.get('word_count'):
        infos.append(f"📝 {r['word_count']} words")
    if r.get('contact', {}).get('email'):
        infos.append(f"✉ {r['contact']['email']}")
    if infos:
        print(f"\n  {dim('  ·  '.join(infos))}")

    if verbose and r.get('education'):
        print(f"\n  🎓 {r['education'][0][:80]}")


def cmd_single(args):
    print_banner()
    if not os.path.exists(args.resume):
        print(red(f"  Error: File not found: {args.resume}"))
        sys.exit(1)

    jd_text = get_jd(args)
    if not jd_text:
        print(red("  Error: Job description required (--jd or --jd-file)"))
        sys.exit(1)

    print(f"  {cyan('→')} Parsing resume: {bold(args.resume)}")
    parser = ResumeParser()
    engine = MatcherEngine()

    try:
        resume_data = parser.parse(args.resume)
        print(f"  {cyan('→')} Running NLP + OpenCV analysis...")
        result = engine.match(resume_data, jd_text)
    except Exception as e:
        print(red(f"  Error: {e}"))
        sys.exit(1)

    print_result(result, show_rank=False, verbose=args.verbose)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n  {green('✓')} Results saved to: {args.output}")
    print()


def cmd_bulk(args):
    print_banner()
    directory = args.resume_dir
    if not os.path.isdir(directory):
        print(red(f"  Error: Directory not found: {directory}"))
        sys.exit(1)

    jd_text = get_jd(args)
    if not jd_text:
        print(red("  Error: Job description required (--jd or --jd-file)"))
        sys.exit(1)

    exts = {'.pdf', '.docx', '.doc', '.txt', '.png', '.jpg', '.jpeg'}
    files = [os.path.join(directory, f) for f in os.listdir(directory)
             if os.path.splitext(f)[1].lower() in exts]

    if not files:
        print(red(f"  No supported resume files found in: {directory}"))
        sys.exit(1)

    print(f"  {cyan('→')} Found {bold(str(len(files)))} resume(s) in {directory}")
    print(f"  {cyan('→')} Ranking against job description...")

    ranker = BatchRanker()
    results = ranker.rank(files, jd_text)

    top_n = args.top if args.top else len(results)
    print(f"\n  {'═'*62}")
    print(f"  {bold('RANKING RESULTS')} — Top {min(top_n, len(results))} of {len(results)}")
    print(f"  {'═'*62}")

    for r in results[:top_n]:
        print_result(r, show_rank=True, verbose=args.verbose)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  {green('✓')} Results saved to: {args.output}")
    print()


def get_jd(args):
    if hasattr(args, 'jd') and args.jd:
        return args.jd
    if hasattr(args, 'jd_file') and args.jd_file:
        with open(args.jd_file) as f:
            return f.read()
    return None


def main():
    parser = argparse.ArgumentParser(
        description="ResuMatch — OpenCV + NLP Resume Matching CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Match single resume
  python cli.py --resume john_doe.pdf --jd "Python developer 3+ years ML experience"

  # Match from JD file
  python cli.py --resume resume.docx --jd-file job_description.txt

  # Bulk rank resumes
  python cli.py --resume-dir ./candidates/ --jd-file jd.txt --top 5

  # Save results as JSON
  python cli.py --resume resume.pdf --jd-file jd.txt --output result.json
        """
    )

    sub = parser.add_subparsers(dest='command')

    # ── Single ──
    sp = sub.add_parser('single', help='Match a single resume')
    sp.add_argument('--resume', required=True, help='Path to resume file')
    sp.add_argument('--jd', help='Job description text')
    sp.add_argument('--jd-file', help='Path to job description file')
    sp.add_argument('--output', help='Save results as JSON')
    sp.add_argument('--verbose', action='store_true', help='Verbose output')

    # ── Bulk ──
    bp = sub.add_parser('bulk', help='Rank multiple resumes')
    bp.add_argument('--resume-dir', required=True, help='Directory of resume files')
    bp.add_argument('--jd', help='Job description text')
    bp.add_argument('--jd-file', help='Path to job description file')
    bp.add_argument('--top', type=int, help='Show top N results')
    bp.add_argument('--output', help='Save results as JSON')
    bp.add_argument('--verbose', action='store_true', help='Verbose output')

    # Quick mode (no subcommand)
    parser.add_argument('--resume', help='Path to resume file (quick mode)')
    parser.add_argument('--resume-dir', help='Directory of resumes (quick mode)')
    parser.add_argument('--jd', help='Job description text')
    parser.add_argument('--jd-file', help='Path to job description file')
    parser.add_argument('--top', type=int, help='Show top N (bulk only)')
    parser.add_argument('--output', help='Save JSON results')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()

    if args.command == 'single' or (not args.command and args.resume):
        cmd_single(args)
    elif args.command == 'bulk' or (not args.command and args.resume_dir):
        args.resume_dir = args.resume_dir or getattr(args, 'resume_dir', None)
        cmd_bulk(args)
    else:
        print_banner()
        parser.print_help()


if __name__ == '__main__':
    main()
