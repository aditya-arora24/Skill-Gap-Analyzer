"""
build_dashboard_demo.py
========================
Build a static HTML "Career Path Dashboard" for ONE example resume from the
corpus, demonstrating the three product surfaces the trained model enables:

  Section 1 — Match score for a specific job, with confidence framing
              (percentile vs other candidates for that job).

  Section 2 — Skill-gap report: what the candidate has, what they're missing,
              missing skills ranked by ESCO depth-based importance.

  Section 3 — 2-year roadmap aggregating skill gaps across the candidate's
              top-5 matched jobs, bucketed into quarters by frequency ×
              importance. Includes a forward simulation: "if you learn the
              Q1-Q2 skills, your average match score across target jobs rises
              from X% to Y%."

Inputs (read-only):
  models/active_learning/{scaler,logreg}.pkl
  data/proccessed again/processed/pair_features_diversified.parquet
  data/proccessed again/processed/cleaned_resumes.parquet
  data/proccessed again/processed/cleaned_jobs.parquet
  data/proccessed again/esco_skill_depths.json

Outputs (NEW directory):
  outputs/dashboard_demo/index.html
  outputs/dashboard_demo/forward_simulation.csv

Run:
    python "src/build_dashboard_demo.py"
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR    = PROJECT_ROOT / "models" / "active_learning"
PROC         = PROJECT_ROOT / "data" / "proccessed again" / "processed"
DEPTH_JSON   = PROJECT_ROOT / "data" / "proccessed again" / "esco_skill_depths.json"

POOL_PARQUET    = PROC / "pair_features_diversified.parquet"
RESUMES_PARQUET = PROC / "cleaned_resumes.parquet"
JOBS_PARQUET    = PROC / "cleaned_jobs.parquet"

OUT_DIR = PROJECT_ROOT / "outputs" / "dashboard_demo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ALL_FEATURES = [
    "embedding_similarity",
    "tfidf_similarity",
    "skill_overlap",
    "weighted_skill_score",
    "num_missing_skills",
    "avg_missing_skill_importance",
    "years_of_experience",
    "title_similarity",
]
THRESHOLD = 0.52   # tuned in active_learning_evaluate.py
TOP_K_JOBS = 5     # target jobs for the roadmap aggregation
ROADMAP_BUCKETS = [
    ("Year 1 · Q1–Q2", "Foundation", 2),
    ("Year 1 · Q3–Q4", "Specialization", 2),
    ("Year 2 · Q1–Q2", "Advanced topics", 2),
    ("Year 2 · Q3–Q4", "Differentiation", 2),
]

# Words that come out of the ESCO vocabulary but aren't actually
# learnable / actionable skills to surface in the user-facing dashboard.
# These slipped through the build_esco_vocab.py BLOCKLIST but are too generic
# or non-skill-like to recommend. Filtering here is a render-time concern;
# the model still uses these tokens in its feature computation, this just
# cleans up what gets *displayed* on the dashboard.
DISPLAY_BLOCKLIST: set[str] = {
    # not actually skills, just nouns / words
    "source", "sources", "energy", "clean energy",
    "call", "calls", "job opportunities", "opportunity", "opportunities",
    "service", "services", "support", "team", "teams",
    "background", "field", "fields", "area", "areas",
    "improvement", "implementation",
    # generic verb-nouns that aren't actionable
    "operation", "operations",
    # domain-specific medical that show up unhelpfully for non-healthcare resumes
    "pregnancy", "primary care", "medicine", "patient care", "patient",
    # process descriptions, not skills
    "root cause analysis", "root causes",
}

# Resume category -> regex describing matching job position_titles. Used by
# the picker to ensure the top-5 matched jobs are coherent with the resume's
# domain (i.e., an IT resume should match IT/software/data jobs, not
# hospital supply chain).
JOB_PATTERN_BY_CATEGORY: dict[str, re.Pattern] = {
    "INFORMATION-TECHNOLOGY": re.compile(
        r"\b(software|developer|engineer|programmer|data|devops|cloud|"
        r"architect|qa\b|sre|system|web|mobile|security|database|"
        r"tech|technical|machine learning|ml engineer|backend|frontend|"
        r"full stack)\b", re.IGNORECASE),
    "FINANCE": re.compile(
        r"\b(finance|financial|accountant|accounting|controller|treasury|"
        r"audit|tax|budget|fp&a|investment|risk)\b", re.IGNORECASE),
    "HR": re.compile(
        r"\b(hr\b|human resources|recruiter|talent|people operations|"
        r"compensation|benefits|training and development)\b", re.IGNORECASE),
    "SALES": re.compile(
        r"\b(sales|account executive|business development|customer success|"
        r"sales representative|account manager|sales specialist)\b", re.IGNORECASE),
    "BANKING": re.compile(
        r"\b(bank|banking|loan|mortgage|teller|credit|financial advisor)\b",
        re.IGNORECASE),
    "ENGINEERING": re.compile(
        r"\b(engineer|mechanical|electrical|civil|chemical|industrial|"
        r"manufacturing|process)\b", re.IGNORECASE),
    "DESIGNER": re.compile(
        r"\b(designer|design|ux|ui|graphic|creative|illustrator)\b",
        re.IGNORECASE),
    "BUSINESS-DEVELOPMENT": re.compile(
        r"\b(business development|partnerships|strategy|operations manager|"
        r"product manager|growth)\b", re.IGNORECASE),
    "HEALTHCARE": re.compile(
        r"\b(nurse|physician|doctor|medical|healthcare|clinical|therapist|"
        r"pharmacist|dental)\b", re.IGNORECASE),
    "TEACHER": re.compile(
        r"\b(teacher|professor|instructor|tutor|education|academic)\b",
        re.IGNORECASE),
}


def is_displayable_skill(s: str) -> bool:
    """Should this skill name appear in the user-facing dashboard?"""
    s = s.strip().lower()
    if not s or len(s) < 3:
        return False
    if s in DISPLAY_BLOCKLIST:
        return False
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_skill_field(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        items = list(value)
    elif isinstance(value, list):
        items = value
    elif isinstance(value, str):
        try:
            items = ast.literal_eval(value)
        except Exception:
            items = []
    else:
        items = []
    out = []
    for it in items:
        if isinstance(it, dict):
            s = str(it.get("skill", "")).strip().lower()
        else:
            s = str(it).strip().lower()
        if s:
            out.append(s)
    return out


def get_skill_importance(depth_data: dict, default_importance: float = 2.5):
    """Return a function: skill_name -> importance score on 0-5 scale."""
    depth_by_label = depth_data.get("depth_by_label", {})
    max_depth = float(depth_data.get("max_depth", 1)) or 1.0
    def fn(skill: str) -> float:
        d = depth_by_label.get(skill.strip().lower())
        if d:
            return (float(d) / max_depth) * 5.0
        return default_importance
    return fn


def stars_for(importance: float) -> str:
    """5-star bar visualization. 4.5 → '★★★★▫'."""
    full = int(round(importance))
    return "★" * full + "▫" * (5 - full)


def pick_demo_resume(resumes_df, pool_df, pair_lookup, model, scaler,
                     jobs_df) -> int:
    """
    Choose a resume that:
      - Has many extracted skills (so the gap report is interesting)
      - Has top-5 job matches that COHERE with the resume's Category
        (e.g., IT resume → IT/software/data jobs, not hospital supply chain)
      - Has a mix of strong + moderate match probabilities for variety
    Returns the resume row index.
    """
    # Score every pair in the pool once
    pool_pairs = pool_df[["job_id", "resume_id"] + ALL_FEATURES].copy()
    X = pool_pairs[ALL_FEATURES].values
    pool_pairs["pred_prob"] = model.predict_proba(scaler.transform(X))[:, 1]

    # Per-resume stats
    per_resume = pool_pairs.groupby("resume_id").agg(
        max_prob=("pred_prob", "max"),
        median_prob=("pred_prob", "median"),
    ).reset_index()

    n_skills = resumes_df["extracted_skills"].apply(
        lambda v: len(set(parse_skill_field(v)))
    )
    per_resume["n_resume_skills"] = per_resume["resume_id"].map(
        {i: int(n_skills.iloc[i]) for i in range(len(resumes_df))}
    )
    per_resume["category"] = per_resume["resume_id"].map(
        {i: str(resumes_df.iloc[i]["Category"]) for i in range(len(resumes_df))}
    )

    # Basic quality filters
    candidates = per_resume[
        (per_resume["n_resume_skills"] >= 10)
        & (per_resume["max_prob"] >= 0.55)
        & (per_resume["median_prob"] >= 0.20)
        & (per_resume["category"].isin(JOB_PATTERN_BY_CATEGORY.keys()))
    ].copy()

    # Coherence check: top-5 jobs should match the resume's Category pattern
    # for at least 3 of 5 jobs (60% coherence).
    job_titles = {i: str(jobs_df.iloc[i]["position_title"]) for i in range(len(jobs_df))}

    def coherence_score(resume_id, category):
        patt = JOB_PATTERN_BY_CATEGORY.get(category)
        if patt is None:
            return 0
        top5 = (pool_pairs[pool_pairs["resume_id"] == resume_id]
                .sort_values("pred_prob", ascending=False)
                .head(TOP_K_JOBS))
        if len(top5) == 0:
            return 0
        return int(sum(
            1 for j in top5["job_id"]
            if patt.search(job_titles.get(int(j), ""))
        ))

    print(f"    scoring coherence for {len(candidates)} candidate resumes...")
    candidates["coherence"] = candidates.apply(
        lambda r: coherence_score(int(r["resume_id"]), r["category"]),
        axis=1,
    )
    # Require ≥3 of top-5 jobs match the resume's Category pattern
    coherent = candidates[candidates["coherence"] >= 3].copy()
    print(f"    {len(coherent)} candidates passed coherence filter "
          f"(≥{3}/5 top jobs in resume's domain)")

    if len(coherent) == 0:
        # Fallback: relax coherence
        coherent = candidates[candidates["coherence"] >= 2].copy()
        print(f"    [warn] no resumes with coherence≥3; relaxing to ≥2  "
              f"({len(coherent)} candidates)")

    if len(coherent) == 0:
        # Last resort: any candidate
        coherent = candidates.copy()

    # Among coherent, prefer max_prob in a healthy range (good signal,
    # not too easy). Sort by (coherence DESC, max_prob in [0.65, 0.85])
    coherent["pref_score"] = coherent.apply(
        lambda r: r["coherence"] * 10 + (
            1.0 - abs(r["max_prob"] - 0.78)
        ),
        axis=1,
    )
    coherent = coherent.sort_values("pref_score", ascending=False).reset_index(drop=True)
    pick_resume_id = int(coherent.iloc[0]["resume_id"])
    print(f"    [pick] resume_id={pick_resume_id}  category={coherent.iloc[0]['category']}  "
          f"coherence={int(coherent.iloc[0]['coherence'])}/5  "
          f"max_prob={coherent.iloc[0]['max_prob']:.2f}")
    return pick_resume_id


def forward_simulate(model, scaler, base_row: pd.Series, resume_skills: set,
                     job_skills: set, importance_fn, added_skills: list[str]):
    """
    Recompute the 4 skill-dependent features after adding `added_skills` to the
    resume's skill set, then re-predict the match probability.

    Returns (new_prob, new_feature_vector).
    """
    new_resume_skills = resume_skills | set(added_skills)
    inter = job_skills & new_resume_skills
    miss  = job_skills - new_resume_skills
    if not job_skills:
        skill_overlap = 0.0
        wss = 0.0
        avg_miss_imp = 0.0
    else:
        skill_overlap = len(inter) / len(job_skills)
        wj = sum(importance_fn(s) for s in job_skills)
        wi = sum(importance_fn(s) for s in inter)
        wss = (wi / wj) if wj > 0 else 0.0
        if miss:
            avg_miss_imp = float(np.mean([importance_fn(s) for s in miss]))
        else:
            avg_miss_imp = 0.0
    n_missing = len(miss)

    # Build the new feature vector (everything else stays the same)
    new_row = base_row.copy()
    new_row["skill_overlap"] = skill_overlap
    new_row["weighted_skill_score"] = wss
    new_row["num_missing_skills"] = n_missing
    new_row["avg_missing_skill_importance"] = avg_miss_imp

    x = new_row[ALL_FEATURES].values.reshape(1, -1).astype(np.float64)
    p = float(model.predict_proba(scaler.transform(x))[0, 1])
    return p, new_row


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------
def html_escape(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
                  .replace(">", "&gt;").replace('"', "&quot;"))


def render_dashboard(demo: dict) -> str:
    """Pure HTML+CSS dashboard. No JS. Self-contained styles."""
    # Section 1: candidate header + top job match
    headline = demo["headline_job"]
    candidate = demo["candidate"]
    skill_gap = demo["skill_gap"]
    roadmap = demo["roadmap"]
    fsim = demo["forward_sim"]

    css = """
    <style>
      :root {
        --bg: #0f172a; --card: #1e293b; --border: #334155;
        --text: #e2e8f0; --muted: #94a3b8;
        --accent: #10b981; --warn: #f59e0b; --danger: #ef4444;
        --good: #22c55e; --bar-bg: #334155;
      }
      body {
        margin: 0; padding: 32px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                     Roboto, Helvetica, Arial, sans-serif;
        background: var(--bg); color: var(--text); line-height: 1.5;
        max-width: 1100px; margin-inline: auto;
      }
      h1 { font-size: 26px; margin: 0 0 4px 0; }
      h2 { font-size: 18px; margin: 0 0 12px 0; color: var(--muted);
           text-transform: uppercase; letter-spacing: 0.5px; }
      .subtitle { color: var(--muted); font-size: 14px; margin: 0 0 28px 0; }
      .card {
        background: var(--card); border: 1px solid var(--border);
        border-radius: 12px; padding: 24px; margin-bottom: 20px;
      }
      .row { display: flex; gap: 32px; align-items: flex-start; }
      .col { flex: 1; }
      .score-num {
        font-size: 56px; font-weight: 700; color: var(--accent);
        line-height: 1; margin: 8px 0 4px 0;
      }
      .score-bar-bg {
        background: var(--bar-bg); height: 14px; border-radius: 7px;
        overflow: hidden; margin: 8px 0;
      }
      .score-bar-fg {
        height: 100%; background: linear-gradient(90deg, #10b981, #34d399);
        border-radius: 7px;
      }
      .percentile { color: var(--muted); font-size: 13px; }
      .skill-list { list-style: none; padding: 0; margin: 0; }
      .skill-list li {
        padding: 8px 0; border-bottom: 1px solid var(--border);
        display: flex; justify-content: space-between; align-items: center;
      }
      .skill-list li:last-child { border-bottom: none; }
      .check { color: var(--good); margin-right: 10px; }
      .warn  { color: var(--warn); margin-right: 10px; }
      .stars { color: var(--warn); font-size: 13px; letter-spacing: 1px; }
      .imp-val { color: var(--muted); font-size: 12px; margin-left: 8px; }
      .roadmap-bucket { margin-bottom: 16px; padding: 14px;
                        border-left: 3px solid var(--accent); background: #0f172a; }
      .roadmap-bucket .title { font-weight: 600; }
      .roadmap-bucket .label { color: var(--muted); font-size: 13px;
                                margin-bottom: 8px; }
      .roadmap-bucket ul { margin: 6px 0 0 18px; padding: 0; }
      .roadmap-bucket li { padding: 4px 0; }
      .freq-tag {
        font-size: 11px; color: var(--muted);
        background: var(--bar-bg); padding: 2px 6px; border-radius: 4px;
        margin-left: 6px;
      }
      .fsim-grid { display: grid; grid-template-columns: 2fr 1fr 1fr 1fr;
                   gap: 8px 16px; align-items: center; font-size: 13px; }
      .fsim-grid .head { color: var(--muted); font-size: 12px;
                         text-transform: uppercase; letter-spacing: 0.3px;
                         border-bottom: 1px solid var(--border); padding-bottom: 6px; }
      .delta-up { color: var(--good); font-weight: 600; }
      .footer { color: var(--muted); font-size: 12px; margin-top: 24px;
                text-align: center; }
      .meta { color: var(--muted); font-size: 13px; }
    </style>
    """

    # Build skill-gap two-column section
    have_lis = "".join(
        f'<li><span><span class="check">✓</span>{html_escape(s)}</span>'
        f'<span class="stars">{stars_for(imp)}</span></li>'
        for s, imp in skill_gap["have"]
    ) or "<li><i>No overlap (unusual)</i></li>"

    miss_lis = "".join(
        f'<li><span><span class="warn">⚠</span>{html_escape(s)}</span>'
        f'<span class="stars">{stars_for(imp)}<span class="imp-val">{imp:.1f}/5</span></span></li>'
        for s, imp in skill_gap["missing"]
    ) or "<li><i>No missing skills — strong match.</i></li>"

    # Roadmap
    roadmap_blocks = ""
    for bucket in roadmap:
        skill_items = "".join(
            f'<li>{html_escape(s)} '
            f'<span class="freq-tag">appears in {freq}/{TOP_K_JOBS} target jobs</span> '
            f'<span class="stars">{stars_for(imp)}</span></li>'
            for s, imp, freq in bucket["skills"]
        ) or "<li><i>(No further required skills aggregated for this quarter)</i></li>"
        roadmap_blocks += f"""
        <div class="roadmap-bucket">
          <div class="title">{html_escape(bucket["period"])} — {html_escape(bucket["label"])}</div>
          <div class="label">{html_escape(bucket["rationale"])}</div>
          <ul>{skill_items}</ul>
        </div>
        """

    # Forward simulation table
    fsim_rows = ""
    for r in fsim["per_job"]:
        fsim_rows += f"""
        <div>{html_escape(r["job"])}</div>
        <div>{r["before"]*100:.0f}%</div>
        <div class="delta-up">→ {r["after"]*100:.0f}%</div>
        <div class="delta-up">+{(r["after"]-r["before"])*100:.0f} pts</div>
        """

    body = f"""
    <h1>Career Path Dashboard</h1>
    <p class="subtitle">
      Personalized job-fit scoring &amp; skill-gap diagnostic
      &nbsp;·&nbsp; demo for resume #{candidate["resume_id"]} ({html_escape(candidate["category"])})
    </p>

    <!-- SECTION 1: Match score -->
    <div class="card">
      <h2>Section 1 · Top Match</h2>
      <div class="row">
        <div class="col">
          <div class="meta">Top matched job</div>
          <div style="font-weight:600; font-size:18px; margin-top:4px;">
            {html_escape(headline["title"])}
          </div>
          <div class="meta" style="margin-top:4px;">{html_escape(headline["company"])}</div>
          <div class="score-num">{headline["score"]*100:.0f}%</div>
          <div class="score-bar-bg"><div class="score-bar-fg"
               style="width:{headline["score"]*100:.0f}%"></div></div>
          <div class="percentile">
            Top {headline["percentile"]}% of candidates for this role
            &nbsp;·&nbsp; Threshold for "match": {THRESHOLD*100:.0f}%
          </div>
        </div>
        <div class="col">
          <div class="meta">Model confidence framing</div>
          <p style="margin: 8px 0;">
            On comparable gold-set pairs, 3 LLM judges
            ({html_escape("Claude / GPT / Gemini")}) agreed unanimously
            {fsim["confidence_unanimous_pct"]}% of the time and reached majority
            consensus on {fsim["confidence_majority_pct"]}%. The score above
            reflects this calibrated reranker.
          </p>
        </div>
      </div>
    </div>

    <!-- SECTION 2: Skill gap -->
    <div class="card">
      <h2>Section 2 · Skill Gap Report</h2>
      <div class="row">
        <div class="col">
          <div class="meta">Skills you have <strong>({skill_gap["n_have"]}/{skill_gap["n_job"]})</strong></div>
          <ul class="skill-list" style="margin-top:10px;">{have_lis}</ul>
        </div>
        <div class="col">
          <div class="meta">Skills you're missing <strong>({skill_gap["n_missing"]})</strong>
            <span style="color:var(--muted)">— sorted by ESCO importance</span></div>
          <ul class="skill-list" style="margin-top:10px;">{miss_lis}</ul>
        </div>
      </div>
      <div class="meta" style="margin-top:16px;">
        Top two missing skills account for
        <strong style="color:var(--text)">{skill_gap["top2_share_pct"]}%</strong>
        of the total gap importance — focus there.
      </div>
    </div>

    <!-- SECTION 3: Roadmap -->
    <div class="card">
      <h2>Section 3 · 2-Year Skill Roadmap</h2>
      <div class="meta" style="margin-bottom:14px;">
        Aggregated across your top {TOP_K_JOBS} matched jobs, ranked by
        (frequency × importance). Buckets are suggested timelines, not strict
        deadlines.
      </div>
      {roadmap_blocks}
    </div>

    <!-- SECTION 4: Forward simulation -->
    <div class="card">
      <h2>Section 4 · Forward Simulation — what learning these skills does</h2>
      <p class="meta" style="margin-top:0;">
        If you add the Q1–Q2 foundation skills
        (<strong>{html_escape(", ".join(fsim["added_skills"]))}</strong>) to your profile, the
        model re-scores you against your top-{TOP_K_JOBS} jobs as follows:
      </p>
      <div class="fsim-grid">
        <div class="head">Target job</div>
        <div class="head">Before</div>
        <div class="head">After</div>
        <div class="head">Delta</div>
        {fsim_rows}
      </div>
      <p class="meta" style="margin-top:14px;">
        Average match score: <strong style="color:var(--text)">
        {fsim["avg_before"]*100:.0f}% → {fsim["avg_after"]*100:.0f}%</strong>
        (+{(fsim["avg_after"]-fsim["avg_before"])*100:.0f} points).
        Computed by toggling the proposed skills in your `extracted_skills`
        set and re-running the trained model — same logic that produced the
        F1=0.769 reranker result.
      </p>
    </div>

    <p class="footer">
      Prototype generated by build_dashboard_demo.py.
      Backed by the active-learning reranker (LogReg, 600 LLM-labeled pairs, F1=0.769).
      Web frontend, user accounts, and learning-platform integrations are future work.
    </p>
    """

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Career Path Dashboard — Demo</title>
{css}
</head><body>{body}</body></html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(" Dashboard demo builder")
    print("=" * 70)

    # 1. Load model + data
    print("\n[1] loading model + data")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    model  = joblib.load(MODEL_DIR / "logreg.pkl")
    resumes = pd.read_parquet(RESUMES_PARQUET).reset_index(drop=True)
    jobs    = pd.read_parquet(JOBS_PARQUET).reset_index(drop=True)
    pool    = pd.read_parquet(POOL_PARQUET)
    depth_data = json.loads(DEPTH_JSON.read_text())
    importance_fn = get_skill_importance(depth_data, default_importance=2.5)

    # Lookup table (job_id, resume_id) -> feature row
    pair_lookup = pool.set_index(["job_id", "resume_id"])

    # 2. Pick a demo resume (with coherence filter on top-5 matched jobs)
    print("\n[2] picking a demo resume")
    resume_id = pick_demo_resume(resumes, pool, pair_lookup, model, scaler, jobs)
    resume_row = resumes.iloc[resume_id]
    resume_skills = set(parse_skill_field(resume_row["extracted_skills"]))
    print(f"    picked resume_id={resume_id}  category={resume_row['Category']}  "
          f"({len(resume_skills)} skills)")

    # 3. Score this resume against every job in the pool
    print(f"\n[3] scoring resume against all jobs in pool")
    cand_pairs = pool[pool["resume_id"] == resume_id][
        ["job_id"] + ALL_FEATURES
    ].reset_index(drop=True)
    if len(cand_pairs) == 0:
        # Fallback: use the pool's lookup against every job
        cand_pairs = pool[pool["resume_id"] == resume_id]
    Xc = cand_pairs[ALL_FEATURES].values.astype(np.float64)
    cand_pairs["pred_prob"] = model.predict_proba(scaler.transform(Xc))[:, 1]
    cand_pairs = cand_pairs.sort_values("pred_prob", ascending=False).reset_index(drop=True)
    top5 = cand_pairs.head(TOP_K_JOBS).copy()
    print(f"    top-{TOP_K_JOBS} match probabilities: "
          f"{[round(p, 2) for p in top5['pred_prob'].tolist()]}")

    # 4. For each top-K job, compute the skill gap
    print(f"\n[4] computing skill gap for top-{TOP_K_JOBS} jobs")
    top_jobs_info = []
    for _, row in top5.iterrows():
        j = int(row["job_id"])
        job_row = jobs.iloc[j]
        job_skills = set(parse_skill_field(job_row["extracted_skills"]))
        matched = job_skills & resume_skills
        missing = job_skills - resume_skills
        top_jobs_info.append({
            "job_id":    j,
            "title":     str(job_row["position_title"])[:80],
            "company":   str(job_row.get("company_name", ""))[:60],
            "score":     float(row["pred_prob"]),
            "matched":   sorted(matched),
            "missing":   sorted(missing),
            "base_row":  row,
            "job_skills_set": job_skills,
        })

    # Headline = top-1 job
    headline_job = top_jobs_info[0]
    job_score = headline_job["score"]
    # Percentile vs other candidates for this job
    all_for_job = pool[pool["job_id"] == headline_job["job_id"]].copy()
    if len(all_for_job) > 0:
        X_all = all_for_job[ALL_FEATURES].values.astype(np.float64)
        all_for_job["pred_prob"] = model.predict_proba(scaler.transform(X_all))[:, 1]
        better_fraction = float((all_for_job["pred_prob"] > job_score).mean())
        percentile = int(round(better_fraction * 100))
    else:
        percentile = 50
    print(f"    headline job: '{headline_job['title']}' @ "
          f"{headline_job['company']}  score={job_score*100:.0f}%  "
          f"top {percentile}%")

    # Build headline skill gap (for section 2): use the headline job
    # Filter through is_displayable_skill so non-actionable noise doesn't surface.
    have_with_imp = sorted(
        [(s, importance_fn(s)) for s in headline_job["matched"] if is_displayable_skill(s)],
        key=lambda t: -t[1],
    )[:8]
    missing_with_imp = sorted(
        [(s, importance_fn(s)) for s in headline_job["missing"] if is_displayable_skill(s)],
        key=lambda t: -t[1],
    )[:8]

    total_missing_imp = sum(
        importance_fn(s) for s in headline_job["missing"] if is_displayable_skill(s)
    )
    top2_imp = sum(imp for _, imp in missing_with_imp[:2])
    top2_share_pct = int(round(100 * (top2_imp / total_missing_imp))) if total_missing_imp > 0 else 0

    # 5. Aggregate skill gaps across top-5 → roadmap
    # Filter through is_displayable_skill so the roadmap doesn't recommend
    # non-skills like "source" or "job opportunities".
    print("\n[5] aggregating gaps into 2-year roadmap")
    agg = {}
    for j_info in top_jobs_info:
        for s in j_info["missing"]:
            if not is_displayable_skill(s):
                continue
            if s not in agg:
                agg[s] = {"freq": 0, "importance": importance_fn(s)}
            agg[s]["freq"] += 1
    # rank by (frequency × importance)
    ranked = sorted(
        agg.items(),
        key=lambda kv: -(kv[1]["freq"] * kv[1]["importance"]),
    )
    print(f"    aggregated skill list ({len(ranked)} skills):")
    for s, info in ranked[:10]:
        print(f"      {s:<30s}  freq={info['freq']}/{TOP_K_JOBS}  "
              f"imp={info['importance']:.2f}")

    # Bucket into the 4 quarters defined in ROADMAP_BUCKETS
    roadmap = []
    idx = 0
    for (period, label, n_skills_in_bucket) in ROADMAP_BUCKETS:
        bucket_items = []
        while idx < len(ranked) and len(bucket_items) < n_skills_in_bucket:
            s, info = ranked[idx]
            bucket_items.append((s, info["importance"], info["freq"]))
            idx += 1
        if period.startswith("Year 1 · Q1"):
            rationale = "Highest frequency × importance — your biggest gaps."
        elif period.startswith("Year 1 · Q3"):
            rationale = "Builds on the Q1–Q2 foundation; medium-impact gaps."
        elif period.startswith("Year 2 · Q1"):
            rationale = "Advanced topics relevant to a subset of your target jobs."
        else:
            rationale = "Soft skills & portfolio building — differentiation."
        roadmap.append({
            "period":    period,
            "label":     label,
            "rationale": rationale,
            "skills":    bucket_items,
        })

    # 6. Forward simulation: add Q1-Q2 skills, recompute scores
    print("\n[6] forward simulation: adding Q1–Q2 skills, re-scoring top-5 jobs")
    q12_skills = [s for (s, _, _) in roadmap[0]["skills"]]
    print(f"    added skills: {q12_skills}")
    fsim_per_job = []
    for j_info in top_jobs_info:
        new_p, _ = forward_simulate(
            model, scaler, j_info["base_row"],
            resume_skills, j_info["job_skills_set"],
            importance_fn, q12_skills,
        )
        fsim_per_job.append({
            "job":    j_info["title"],
            "before": j_info["score"],
            "after":  new_p,
        })
        print(f"    {j_info['title'][:50]:<52s}  "
              f"{j_info['score']*100:5.0f}% → {new_p*100:5.0f}%  "
              f"({(new_p - j_info['score'])*100:+.0f} pts)")

    avg_before = float(np.mean([r["before"] for r in fsim_per_job]))
    avg_after  = float(np.mean([r["after"]  for r in fsim_per_job]))
    print(f"    avg: {avg_before*100:.0f}% → {avg_after*100:.0f}%  "
          f"(+{(avg_after - avg_before)*100:.0f} pts)")

    # Write forward simulation CSV
    fsim_df = pd.DataFrame([
        {"job_title": r["job"], "before_pct": round(r["before"] * 100, 1),
         "after_pct": round(r["after"] * 100, 1),
         "delta_pct": round((r["after"] - r["before"]) * 100, 1)}
        for r in fsim_per_job
    ])
    fsim_df.to_csv(OUT_DIR / "forward_simulation.csv", index=False)
    print(f"[write] {OUT_DIR / 'forward_simulation.csv'}")

    # 7. Render HTML
    print("\n[7] rendering HTML dashboard")
    demo = {
        "candidate":     {
            "resume_id": resume_id,
            "category":  str(resume_row["Category"]),
        },
        "headline_job":  {
            "title":      headline_job["title"],
            "company":    headline_job["company"],
            "score":      headline_job["score"],
            "percentile": max(1, percentile),
        },
        "skill_gap":     {
            "n_have":         len(headline_job["matched"]),
            "n_missing":      len(headline_job["missing"]),
            "n_job":          len(headline_job["job_skills_set"]),
            "have":           have_with_imp,
            "missing":        missing_with_imp,
            "top2_share_pct": top2_share_pct,
        },
        "roadmap":       roadmap,
        "forward_sim":   {
            "added_skills":              q12_skills,
            "per_job":                   fsim_per_job,
            "avg_before":                avg_before,
            "avg_after":                 avg_after,
            "confidence_unanimous_pct":  71,   # from gold-set inter-LLM agreement
            "confidence_majority_pct":   100,  # majority = unanimous + 2-1 disputes
        },
    }
    html = render_dashboard(demo)
    html_path = OUT_DIR / "index.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"[write] {html_path}")

    # 8. Run metadata
    meta = {
        "resume_id":           resume_id,
        "resume_category":     str(resume_row["Category"]),
        "n_resume_skills":     len(resume_skills),
        "top_jobs":            [
            {"job_id": j["job_id"], "title": j["title"], "score": j["score"]}
            for j in top_jobs_info
        ],
        "q12_added_skills":    q12_skills,
        "avg_score_before":    avg_before,
        "avg_score_after":     avg_after,
        "avg_score_delta":     avg_after - avg_before,
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[write] {OUT_DIR / 'run_metadata.json'}")

    print("\n" + "=" * 70)
    print(f" Done. Open the dashboard in your browser:")
    print(f"   file:///{html_path.resolve().as_posix().replace(' ', '%20')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
