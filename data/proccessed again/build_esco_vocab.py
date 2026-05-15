"""
build_esco_vocab.py
===================
Build the merged ESCO + tech-skills vocabulary used by preprocess_pipeline.

Conservative first pass:
  - skillType == 'knowledge' only (3,221 entries) -- domain nouns, not verbs
  - preferredLabel + altLabels + hiddenLabels (newline- or pipe-separated,
    we handle both)
  - lowercase + strip
  - drop entries shorter than 4 chars
  - drop entries on a small English stopword/verb blocklist

Then unions with TECH_SKILLS from tech_skills.py.

Reads:  data/raw/skills_en.csv
Writes: data/proccessed again/esco_skills_combined.json

Run:
    python "data/proccessed again/build_esco_vocab.py"
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

# Make tech_skills.py importable (sits next to this file).
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from tech_skills import TECH_SKILLS  # noqa: E402

PROJECT_ROOT = SCRIPT_DIR.parent.parent
SRC_CSV  = PROJECT_ROOT / "data" / "raw" / "skills_en.csv"
OUT_JSON = SCRIPT_DIR / "esco_skills_combined.json"


# ---------------------------------------------------------------------------
# Filter config
# ---------------------------------------------------------------------------
MIN_LEN = 4

# Words that are too generic to be useful as skill matches. These are common
# English verbs / fillers / nouns that appear in nearly every resume and JD.
# If any of these end up in the vocabulary they will cause skill_overlap to
# look high while measuring nothing.
BLOCKLIST: set[str] = {
    # generic actions / verbs
    "use", "uses", "used", "do", "does", "make", "makes", "made",
    "build", "create", "creates", "manage", "manages", "managed",
    "develop", "develops", "developed", "work", "works", "worked",
    "perform", "performs", "performed", "operate", "operates", "operated",
    "act", "acts", "acted", "aid", "aids", "help", "helps",
    "support", "supports", "assist", "assists", "carry", "ensure",
    "ensures", "provide", "provides", "deliver", "delivers", "complete",
    "completes", "achieve", "achieves", "follow", "follows", "comply",
    "advise", "advises", "recommend", "suggests", "consult", "consults",
    "cooperate", "coordinate", "coordinates", "supervise", "supervises",
    "improve", "improves", "implement", "implements", "establish",
    "evaluate", "evaluates", "communicate", "communicates",
    # generic nouns
    "thing", "things", "stuff", "item", "items", "task", "tasks",
    "job", "jobs", "role", "roles", "person", "people", "user", "users",
    "customer", "customers", "client", "clients", "team", "teams",
    "group", "groups", "issue", "issues", "problem", "problems",
    "matter", "case", "cases", "type", "types", "kind", "kinds",
    "area", "areas", "field", "fields", "company", "organization",
    # English fillers / function words
    "the", "and", "but", "for", "with", "from", "into", "this", "that",
    "those", "these", "such", "some", "many", "more", "most", "other",
    "than", "then", "when", "where", "what", "which", "while",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean(s) -> str:
    if not isinstance(s, str):
        return ""
    return s.strip().lower()


def expand_label_field(field) -> list[str]:
    """ESCO altLabels / hiddenLabels are typically newline-separated, but some
    rows use pipes. Handle both."""
    if not isinstance(field, str) or not field:
        return []
    parts = re.split(r"[\n|]", field)
    return [clean(p) for p in parts if clean(p)]


def keep(label: str) -> bool:
    if not label or len(label) < MIN_LEN:
        return False
    if label in BLOCKLIST:
        return False
    if not re.search(r"[a-z]", label):    # must contain a letter
        return False
    # Drop entries that are mostly digits / punctuation
    alpha_chars = sum(1 for c in label if c.isalpha())
    if alpha_chars / len(label) < 0.5:
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not SRC_CSV.exists():
        print(f"ERROR: source not found: {SRC_CSV}", file=sys.stderr)
        sys.exit(1)

    print(f"[load] {SRC_CSV}")
    df = pd.read_csv(SRC_CSV, encoding="utf-8")
    print(f"       {len(df):,} rows total")

    if "skillType" not in df.columns or "preferredLabel" not in df.columns:
        print("ERROR: source CSV missing skillType or preferredLabel", file=sys.stderr)
        sys.exit(2)

    # Conservative: knowledge-type entries only
    knowledge = df[df["skillType"] == "knowledge"].copy()
    print(f"[filter] skillType=knowledge: {len(knowledge):,} rows")

    skills: set[str] = set()
    n_pref = n_alt = n_hidden = n_dropped = 0

    for _, row in knowledge.iterrows():
        # preferredLabel
        pl = clean(row.get("preferredLabel"))
        if pl:
            if keep(pl):
                skills.add(pl)
                n_pref += 1
            else:
                n_dropped += 1
        # altLabels (synonyms — high recall value)
        for alt in expand_label_field(row.get("altLabels")):
            if keep(alt):
                if alt not in skills:
                    n_alt += 1
                skills.add(alt)
            else:
                n_dropped += 1
        # hiddenLabels (less common synonyms)
        for hid in expand_label_field(row.get("hiddenLabels")):
            if keep(hid):
                if hid not in skills:
                    n_hidden += 1
                skills.add(hid)
            else:
                n_dropped += 1

    print(f"[esco] preferredLabel kept: {n_pref}")
    print(f"[esco] altLabels added:     {n_alt}")
    print(f"[esco] hiddenLabels added:  {n_hidden}")
    print(f"[esco] entries dropped (< {MIN_LEN} chars / blocklist / no alpha): {n_dropped}")
    print(f"[esco] unique skills after filter: {len(skills):,}")

    # Union with TECH_SKILLS (no filter on TECH_SKILLS — they are curated)
    tech_added = 0
    for s in TECH_SKILLS:
        c = clean(s)
        if c and c not in skills:
            skills.add(c)
            tech_added += 1
    print(f"[merge] TECH_SKILLS contributing new entries: {tech_added}")
    print(f"[final] combined vocabulary size: {len(skills):,}")

    out = {
        "skills": sorted(skills),
        "count": len(skills),
        "source_breakdown": {
            "esco_preferred_kept": n_pref,
            "esco_altlabels_added": n_alt,
            "esco_hiddenlabels_added": n_hidden,
            "esco_dropped_by_filter": n_dropped,
            "tech_skills_added": tech_added,
            "total": len(skills),
        },
        "filter_config": {
            "min_length": MIN_LEN,
            "skill_type": "knowledge",
            "blocklist_size": len(BLOCKLIST),
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[write] {OUT_JSON}")


if __name__ == "__main__":
    main()
