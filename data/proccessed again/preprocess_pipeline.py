"""
Preprocessing pipeline for resumes, job descriptions and O*NET skill data.

Datasets expected in the same directory:
    - training_data.csv              (job descriptions + structured outputs)
    - Resume (1).csv                 (raw resumes: text + HTML)
    - Skills.xlsx                    (O*NET skills, Importance/Level)
    - Skills to Work Activities.xlsx (skill -> work activity mapping)
    - Skills to Work Context.xlsx    (skill -> work context mapping)

Outputs (written to ./processed/):
    - cleaned_resumes.parquet
    - cleaned_jobs.parquet
    - skill_dictionary.json
    - skill_to_activity_map.json
    - skill_to_context_map.json
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
from bs4 import BeautifulSoup

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from tech_skills import (
    TECH_SKILLS,
    TECH_ALIASES,
    AMBIGUOUS_TOKENS,
    AMBIGUOUS_TOKENS_SHORT,
    WEAK_CONTEXT_CUES,
    STRONG_CONTEXT_CUES,
)

# flashtext makes 13k+ phrase preservation tractable (single Aho-Corasick scan
# instead of one regex pass per skill). If missing, fall back to the old regex
# path with a console warning.
try:
    from flashtext import KeywordProcessor   # type: ignore
    _HAVE_FLASHTEXT = True
except Exception:
    _HAVE_FLASHTEXT = False


# ---------------------------------------------------------------------------
# 0. NLTK bootstrap
# ---------------------------------------------------------------------------
def ensure_nltk() -> None:
    for pkg in [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]:
        try:
            nltk.data.find(pkg[0])
        except LookupError:
            nltk.download(pkg[1], quiet=True)


# ---------------------------------------------------------------------------
# 1. Stopword configuration
# ---------------------------------------------------------------------------
KEEP_WORDS = {"with", "using", "experience", "in", "of", "knowledge", "skills", "ability"}
DOMAIN_STOPWORDS = {"responsible", "duties", "including", "various", "etc"}


def build_stopword_set() -> set[str]:
    base = set(stopwords.words("english"))
    base -= KEEP_WORDS
    base |= DOMAIN_STOPWORDS
    return base


# ---------------------------------------------------------------------------
# 2. Text cleaning
# ---------------------------------------------------------------------------
# Keep letters, digits, whitespace, '+' (C++), '#' (C#), '_' (skill phrases)
_ALLOWED_CHARS = re.compile(r"[^a-z0-9+#_\s]")
_MULTI_WS = re.compile(r"\s+")


def strip_html(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    return BeautifulSoup(text, "lxml").get_text(separator=" ")


def build_alias_patterns(aliases: dict[str, str]) -> list[tuple[re.Pattern, str]]:
    """
    Compile regex substitutions for tech skills that don't tokenize cleanly
    (c++, c#, .net, node.js, ci/cd, ...). Apply BEFORE clean_text so the
    troublesome characters never reach the tokenizer.

    Order matters: longest LHS first so 'asp.net' beats '.net'.
    """
    patterns = []
    for canonical in sorted(aliases, key=len, reverse=True):
        # case-insensitive, not anchored on word boundaries because '+' / '#'
        # already disqualify \b. Use a manual lookaround for surrounding non-alphanum.
        regex = re.compile(
            r"(?<![A-Za-z0-9_])" + re.escape(canonical) + r"(?![A-Za-z0-9_])",
            flags=re.IGNORECASE,
        )
        patterns.append((regex, aliases[canonical]))
    return patterns


def apply_aliases(text: str, patterns: list[tuple[re.Pattern, str]]) -> str:
    if not isinstance(text, str) or not text:
        return ""
    for patt, repl in patterns:
        text = patt.sub(repl, text)
    return text


def clean_text(text: str) -> str:
    """Lowercase, strip HTML, drop noise but keep '+' and '#'. Preserves digits."""
    if not isinstance(text, str) or not text:
        return ""
    text = strip_html(text)
    text = text.lower()
    text = _ALLOWED_CHARS.sub(" ", text)
    text = _MULTI_WS.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# 3. Skill phrase preservation
# ---------------------------------------------------------------------------
def build_phrase_patterns(skill_phrases: Iterable[str]) -> list[tuple[re.Pattern, str]]:
    """
    Pre-compile regex patterns that turn 'machine learning' -> 'machine_learning'.
    Longer phrases first so 'deep learning' doesn't get blocked by 'learning'.

    NOTE: at 13k+ phrases this becomes a hot path. The flashtext-backed
    `build_phrase_processor` below is preferred when flashtext is installed;
    this regex implementation remains as a correctness-equivalent fallback.
    """
    cleaned = sorted({p.strip().lower() for p in skill_phrases if isinstance(p, str) and " " in p.strip()},
                     key=len, reverse=True)
    patterns = []
    for phrase in cleaned:
        token = phrase.replace(" ", "_")
        patt = re.compile(r"\b" + re.escape(phrase) + r"\b")
        patterns.append((patt, token))
    return patterns


def build_phrase_processor(skill_phrases: Iterable[str]):
    """
    flashtext-backed multi-keyword scanner. Single Aho-Corasick pass over each
    text replaces all known multi-word skills with their underscore form.
    Linear in input length, independent of vocabulary size after construction.
    """
    if not _HAVE_FLASHTEXT:
        return None
    kp = KeywordProcessor(case_sensitive=False)
    seen = set()
    for phrase in skill_phrases:
        if not isinstance(phrase, str):
            continue
        p = phrase.strip().lower()
        if " " not in p or p in seen:
            continue
        seen.add(p)
        kp.add_keyword(p, p.replace(" ", "_"))
    return kp


def preserve_skill_phrases(text: str, patterns: list[tuple[re.Pattern, str]]) -> str:
    for patt, token in patterns:
        text = patt.sub(token, text)
    return text


def preserve_skill_phrases_kp(text: str, kp) -> str:
    return kp.replace_keywords(text)


def restore_skill_phrases(tokens: list[str]) -> list[str]:
    return [t.replace("_", " ") if "_" in t else t for t in tokens]


# ---------------------------------------------------------------------------
# 3b. Raw-case check for ambiguous skill tokens
# ---------------------------------------------------------------------------
def _ambiguous_in_raw(skill: str, raw_text: str) -> bool:
    """
    Return True if `skill` appears in `raw_text` in a form that looks like
    a real tech mention rather than the homograph English word.

    SHORT tokens (1-2 chars): require capitalization in raw text AND a strong
                              context cue word within ~40 chars. Strict
                              because every false positive blows up overlap.
    LONG  tokens (3+ chars):  capitalization OR any context cue is enough.
    """
    if not isinstance(raw_text, str) or not raw_text:
        return False

    # canonical capitalised form
    cap_token = skill.upper() if len(skill) <= 2 else skill[0].upper() + skill[1:]
    has_cap = bool(re.search(
        r"(?<![A-Za-z0-9_])" + re.escape(cap_token) + r"(?![A-Za-z0-9_])",
        raw_text,
    ))

    # context-cue check on the lowercased text
    lower = raw_text.lower()
    cue_set = STRONG_CONTEXT_CUES if skill in AMBIGUOUS_TOKENS_SHORT else WEAK_CONTEXT_CUES
    has_cue = False
    for m in re.finditer(
        r"(?<![A-Za-z0-9_])" + re.escape(skill) + r"(?![A-Za-z0-9_])",
        lower,
    ):
        window = lower[max(0, m.start() - 40): min(len(lower), m.end() + 40)]
        if any(cue in window for cue in cue_set):
            has_cue = True
            break

    if skill in AMBIGUOUS_TOKENS_SHORT:
        return has_cap and has_cue
    return has_cap or has_cue


# ---------------------------------------------------------------------------
# 4. Tokenization, stopword removal, POS-aware lemmatization
# ---------------------------------------------------------------------------
def _wordnet_pos(tag: str) -> str:
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def tokenize_lemmatize(
    text: str,
    stop_set: set[str],
    lemmatizer: WordNetLemmatizer,
) -> list[str]:
    if not text:
        return []
    tokens = word_tokenize(text)
    # Underscore-joined skill phrases survive word_tokenize as single tokens.
    tagged = nltk.pos_tag(tokens)
    out: list[str] = []
    for tok, tag in tagged:
        if tok in stop_set:
            continue
        # Skip pure punctuation that may have leaked through
        if not re.search(r"[a-z0-9]", tok):
            continue
        if "_" in tok:
            # Skill phrase token — leave as-is, no lemmatization
            out.append(tok)
        else:
            out.append(lemmatizer.lemmatize(tok, _wordnet_pos(tag)))
    return out


# ---------------------------------------------------------------------------
# 5. Resume sectioning (lightweight)
# ---------------------------------------------------------------------------
SECTION_PATTERNS = {
    "skills": re.compile(r"\b(skills|technical skills|core competencies)\b", re.I),
    "experience": re.compile(r"\b(experience|work history|employment|professional experience)\b", re.I),
    "education": re.compile(r"\b(education|academic background|qualifications)\b", re.I),
}


def extract_resume_sections(raw_text: str) -> dict[str, str]:
    """Best-effort section split on the *raw* resume text (before lowercasing)."""
    if not isinstance(raw_text, str) or not raw_text:
        return {"skills": "", "experience": "", "education": ""}

    # Find header positions
    hits = []
    for name, patt in SECTION_PATTERNS.items():
        for m in patt.finditer(raw_text):
            hits.append((m.start(), name))
    hits.sort()

    sections = {"skills": "", "experience": "", "education": ""}
    for i, (start, name) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(raw_text)
        # Only keep the first occurrence per section (resumes often repeat)
        if not sections[name]:
            sections[name] = raw_text[start:end].strip()
    return sections


# ---------------------------------------------------------------------------
# 6. Skill extraction (dictionary lookup)
# ---------------------------------------------------------------------------
def extract_skills(
    tokens: list[str],
    onet_vocab: set[str],
    tech_vocab: set[str],
    alias_reverse: dict[str, str],
    raw_text: str | None = None,
) -> list[dict]:
    """
    Return [{'skill': str, 'source': 'onet'|'tech'}, ...] sorted by skill name.
    Aliased tech tokens (e.g. 'cplusplus') are mapped back to 'c++'.

    If `raw_text` is provided, ambiguous single-/short-token skills (r, go,
    ml, less, swift, ...) are kept only when the raw text shows them with
    capitalisation or in a tech-context window. This kills the false-positive
    flood from English homographs.
    """
    found: dict[str, str] = {}
    for t in tokens:
        candidate = t.replace("_", " ")
        # Reverse alias to canonical (cplusplus -> c++), if applicable
        canonical = alias_reverse.get(candidate, candidate)
        if canonical in onet_vocab:
            found[canonical] = "onet"
        elif canonical in tech_vocab or candidate in tech_vocab:
            found[canonical] = "tech"

    # Ambiguous-token filter (post-extraction, needs raw text)
    if raw_text is not None and found:
        kept: dict[str, str] = {}
        for skill, src in found.items():
            if skill in AMBIGUOUS_TOKENS:
                if _ambiguous_in_raw(skill, raw_text):
                    kept[skill] = src
                # else: drop the false positive
            else:
                kept[skill] = src
        found = kept

    return [{"skill": s, "source": src} for s, src in sorted(found.items())]


# ---------------------------------------------------------------------------
# 7. O*NET skill data processing
# ---------------------------------------------------------------------------
def process_skills_xlsx(path: str) -> tuple[dict, dict]:
    """
    Returns:
        skill_dictionary: {skill_name -> skill_id}
        skill_to_context_map (Importance/Level aggregates per skill, from Skills.xlsx itself):
            {skill_name: {"importance_mean": float, "level_mean": float, "n_records": int}}
    """
    df = pd.read_excel(path)
    df["Element Name"] = df["Element Name"].str.strip().str.lower()

    skill_dictionary = (
        df[["Element Name", "Element ID"]]
        .drop_duplicates()
        .set_index("Element Name")["Element ID"]
        .to_dict()
    )

    # Aggregate Importance / Level scales per skill (across all SOC codes)
    pivot = (
        df.groupby(["Element Name", "Scale ID"])["Data Value"]
        .mean()
        .unstack("Scale ID")
        .rename(columns={"IM": "importance_mean", "LV": "level_mean"})
    )
    counts = df.groupby("Element Name").size().rename("n_records")
    agg = pivot.join(counts).fillna(0.0)

    skill_stats = agg.to_dict(orient="index")
    return skill_dictionary, skill_stats


def process_skill_to_activity(path: str) -> dict[str, list[str]]:
    df = pd.read_excel(path)
    df["Skills Element Name"] = df["Skills Element Name"].str.strip().str.lower()
    df["Work Activities Element Name"] = df["Work Activities Element Name"].str.strip()

    mapping: dict[str, list[str]] = defaultdict(list)
    for skill, act in zip(df["Skills Element Name"], df["Work Activities Element Name"]):
        mapping[skill].append(act)
    return {k: sorted(set(v)) for k, v in mapping.items()}


def process_skill_to_context(path: str) -> dict[str, list[str]]:
    df = pd.read_excel(path)
    df["Skills Element Name"] = df["Skills Element Name"].str.strip().str.lower()
    df["Work Context Element Name"] = df["Work Context Element Name"].str.strip()

    mapping: dict[str, list[str]] = defaultdict(list)
    for skill, ctx in zip(df["Skills Element Name"], df["Work Context Element Name"]):
        mapping[skill].append(ctx)
    return {k: sorted(set(v)) for k, v in mapping.items()}


# ---------------------------------------------------------------------------
# 8. Pipeline orchestration
# ---------------------------------------------------------------------------
class TextPipeline:
    def __init__(
        self,
        skill_phrases: Iterable[str],
        onet_vocab: set[str],
        tech_vocab: set[str],
        aliases: dict[str, str],
    ):
        self.stop_set = build_stopword_set()
        self.lemmatizer = WordNetLemmatizer()
        self.alias_patterns = build_alias_patterns(aliases)
        self.alias_reverse = {v: k for k, v in aliases.items()}
        self.onet_vocab = onet_vocab
        self.tech_vocab = tech_vocab

        # Prefer flashtext for phrase preservation. With 13k+ phrases the
        # regex path is the wall-clock bottleneck (~30x slower).
        self.phrase_kp = build_phrase_processor(skill_phrases)
        if self.phrase_kp is None:
            print("[warn] flashtext not installed -> using slow regex phrase preservation. "
                  "Install with `pip install flashtext` for ~30x speedup.")
            self.phrase_patterns = build_phrase_patterns(skill_phrases)
        else:
            self.phrase_patterns = None

    def process(self, raw: str) -> dict:
        # 1. Apply aliases on RAW text so c++ / c# / node.js survive cleaning
        aliased = apply_aliases(raw, self.alias_patterns)
        cleaned = clean_text(aliased)
        if self.phrase_kp is not None:
            with_phrases = preserve_skill_phrases_kp(cleaned, self.phrase_kp)
        else:
            with_phrases = preserve_skill_phrases(cleaned, self.phrase_patterns)
        tokens = tokenize_lemmatize(with_phrases, self.stop_set, self.lemmatizer)
        # Pass raw text into extract_skills so ambiguous-token filter has
        # access to original capitalisation and surrounding context.
        extracted = extract_skills(
            tokens, self.onet_vocab, self.tech_vocab, self.alias_reverse,
            raw_text=raw,
        )
        readable_tokens = restore_skill_phrases(tokens)
        return {
            "cleaned_text": " ".join(readable_tokens),
            "tokens": tokens,
            "extracted_skills": extracted,
        }


def run(data_dir: str = ".", out_dir: str = "processed", sample: int | None = None) -> None:
    ensure_nltk()
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- O*NET skill data ----
    print("[1/4] Processing O*NET skill files...")
    skill_dictionary, skill_stats = process_skills_xlsx(data_dir / "Skills.xlsx")
    skill_to_activity = process_skill_to_activity(data_dir / "Skills to Work Activities.xlsx")
    skill_to_context_onet = process_skill_to_context(data_dir / "Skills to Work Context.xlsx")

    # Merge importance/level stats into the context map for downstream use
    skill_to_context: dict[str, dict] = {}
    for skill in set(skill_dictionary) | set(skill_to_context_onet):
        skill_to_context[skill] = {
            "context_features": skill_to_context_onet.get(skill, []),
            "importance_mean": skill_stats.get(skill, {}).get("importance_mean", 0.0),
            "level_mean": skill_stats.get(skill, {}).get("level_mean", 0.0),
        }

    with open(out_dir / "skill_dictionary.json", "w", encoding="utf-8") as f:
        json.dump(skill_dictionary, f, indent=2)
    with open(out_dir / "skill_to_activity_map.json", "w", encoding="utf-8") as f:
        json.dump(skill_to_activity, f, indent=2)
    with open(out_dir / "skill_to_context_map.json", "w", encoding="utf-8") as f:
        json.dump(skill_to_context, f, indent=2)

    onet_vocab = set(skill_dictionary.keys())
    tech_vocab = set(s.lower().strip() for s in TECH_SKILLS)

    # ---- ESCO vocabulary (if present) ----------------------------------- #
    esco_path = Path(__file__).resolve().parent / "esco_skills_combined.json"
    esco_vocab: set[str] = set()
    if esco_path.exists():
        try:
            esco_data = json.load(open(esco_path, encoding="utf-8"))
            esco_vocab = set(esco_data.get("skills", []))
            print(f"      ESCO vocabulary loaded: {len(esco_vocab):,} skills "
                  f"from {esco_path.name}")
        except Exception as e:
            print(f"[warn] failed to load ESCO vocabulary ({e}); skipping")
    else:
        print(f"[warn] {esco_path.name} not found -> skipping ESCO. "
              f"Run build_esco_vocab.py first to enable expanded vocab.")

    combined_vocab = onet_vocab | tech_vocab | esco_vocab

    # tech_vocab passed to TextPipeline is the union of TECH_SKILLS and ESCO,
    # because both share the "tech" source tag (no per-source weighting in
    # downstream code distinguishes them today).
    extended_tech_vocab = tech_vocab | esco_vocab

    # Phrase patterns: any multi-word skill from any source
    skill_phrases = [s for s in combined_vocab if " " in s]
    pipeline = TextPipeline(
        skill_phrases=skill_phrases,
        onet_vocab=onet_vocab,
        tech_vocab=extended_tech_vocab,
        aliases=TECH_ALIASES,
    )

    # Persist the merged skill dictionary for downstream consumers
    merged_dict = {**{k: {"id": v, "source": "onet"} for k, v in skill_dictionary.items()}}
    for s in extended_tech_vocab:
        if s not in merged_dict:
            merged_dict[s] = {"id": None, "source": "tech"}
    with open(out_dir / "skill_dictionary_merged.json", "w", encoding="utf-8") as f:
        json.dump(merged_dict, f, indent=2)
    print(f"      merged vocab: {len(onet_vocab)} O*NET + {len(tech_vocab)} tech "
          f"+ {len(esco_vocab)} ESCO = {len(combined_vocab):,} unique skills "
          f"({len(skill_phrases):,} multi-word)")

    # ---- Resumes ----
    print("[2/4] Processing resumes...")
    resume_path = data_dir / "Resume (1).csv"
    resumes = pd.read_csv(resume_path, nrows=sample)
    # Prefer HTML (richer formatting), fall back to plain text
    raw_text = resumes["Resume_html"].fillna("").astype(str)
    raw_text = raw_text.where(raw_text.str.len() > 0, resumes["Resume_str"].fillna("").astype(str))

    sections = raw_text.apply(extract_resume_sections)
    resumes["section_skills"] = sections.apply(lambda d: d["skills"])
    resumes["section_experience"] = sections.apply(lambda d: d["experience"])
    resumes["section_education"] = sections.apply(lambda d: d["education"])

    proc = raw_text.apply(pipeline.process)
    resumes["cleaned_resume"] = proc.apply(lambda d: d["cleaned_text"])
    resumes["tokenized_text"] = proc.apply(lambda d: d["tokens"])
    resumes["extracted_skills"] = proc.apply(lambda d: d["extracted_skills"])

    resumes.to_parquet(out_dir / "cleaned_resumes.parquet", index=False)
    print(f"      wrote {len(resumes)} rows -> cleaned_resumes.parquet")

    # ---- Job descriptions ----
    print("[3/4] Processing job descriptions...")
    jobs = pd.read_csv(data_dir / "training_data.csv", nrows=sample)
    jd_text = jobs["job_description"].fillna("").astype(str)

    proc = jd_text.apply(pipeline.process)
    jobs["cleaned_job_description"] = proc.apply(lambda d: d["cleaned_text"])
    jobs["tokenized_text"] = proc.apply(lambda d: d["tokens"])
    jobs["extracted_skills"] = proc.apply(lambda d: d["extracted_skills"])

    jobs.to_parquet(out_dir / "cleaned_jobs.parquet", index=False)
    print(f"      wrote {len(jobs)} rows -> cleaned_jobs.parquet")

    # ---- Extraction summary (for sanity-checking vocabulary integration) ---
    n_resume_skills = resumes["extracted_skills"].apply(len).sum()
    n_job_skills    = jobs["extracted_skills"].apply(len).sum()
    print()
    print("[summary] skill extraction:")
    print(f"  resumes: {n_resume_skills:,} total skills, "
          f"avg {n_resume_skills / max(1, len(resumes)):.2f} per resume")
    print(f"  jobs:    {n_job_skills:,} total skills, "
          f"avg {n_job_skills / max(1, len(jobs)):.2f} per job")
    if n_resume_skills / max(1, len(resumes)) > 30:
        print("[WARN] avg skills per resume > 30. ESCO false-positive flood likely. "
              "Consider tightening BLOCKLIST or MIN_LEN in build_esco_vocab.py.")

    print("[4/4] Done. Outputs in:", out_dir.resolve())


if __name__ == "__main__":
    import argparse

    # Default paths are resolved relative to THIS script's location, not the
    # caller's CWD. That way `python "data/proccessed again/preprocess_pipeline.py"`
    # from the project root finds the raw CSVs in data/raw/ and writes
    # outputs into data/proccessed again/processed/ without --data-dir.
    SCRIPT_DIR = Path(__file__).resolve().parent
    DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "raw"          # data/raw/
    DEFAULT_OUT_DIR  = SCRIPT_DIR / "processed"           # data/proccessed again/processed/

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--out-dir",  default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--sample", type=int, default=None,
                    help="Process only first N rows (smoke test).")
    args = ap.parse_args()
    print(f"[paths] data_dir = {args.data_dir}")
    print(f"[paths] out_dir  = {args.out_dir}")
    run(args.data_dir, args.out_dir, args.sample)
