"""
build_esco_depths.py
====================
Compute ESCO hierarchy depth for every skill, then convert depth to a
0-5 importance score so weighted_skill_score and avg_missing_skill_importance
no longer collapse to the constant default fallback (2.59).

Inputs:
  data/raw/skills_en.csv                       (URI -> labels)
  data/raw/broaderRelationsSkillPillar_en.csv  (URI -> parent URI)

Output:
  data/proccessed again/esco_skill_depths.json
    {
      "depth_by_label": {"<label_lowercase>": <int depth>, ...},
      "max_depth":       <int>,
      "stats": {...}
    }

Algorithm:
  1. Build the parent-of map (a child URI can have several broaderUris -> DAG).
  2. Find ROOTS: any URI that is a broader of someone but never a child.
     (A node with no parent in the relationships file.)
  3. BFS from each root assigning depth=1, depth=2, ... For DAG-safe handling
     a node's depth is min(depth via any path).
  4. For each ESCO skill (from skills_en.csv) look up its conceptUri's depth,
     and emit one entry per label form (preferredLabel + altLabels +
     hiddenLabels) so downstream lookup works regardless of which surface
     form was extracted.

Run:
    python "data/proccessed again/build_esco_depths.py"
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

SKILLS_CSV    = PROJECT_ROOT / "data" / "raw" / "skills_en.csv"
RELATIONS_CSV = PROJECT_ROOT / "data" / "raw" / "broaderRelationsSkillPillar_en.csv"
OUT_JSON      = SCRIPT_DIR / "esco_skill_depths.json"


def step1_check_inputs() -> None:
    if not SKILLS_CSV.exists():
        print(f"ERROR: missing {SKILLS_CSV}", file=sys.stderr)
        sys.exit(1)
    if not RELATIONS_CSV.exists():
        print()
        print("Missing ESCO hierarchy file. User needs to download "
              "broaderRelationsSkillPillar_en.csv from ESCO and place it in "
              "data/raw/. This file should be in the same ZIP that contained "
              "skills_en.csv.")
        sys.exit(1)


def step2_compute_depths() -> dict[str, int]:
    """Return {conceptUri: depth} for every URI in the relations graph."""
    rel = pd.read_csv(RELATIONS_CSV, encoding="utf-8")
    print(f"[load] {RELATIONS_CSV.name}: {len(rel):,} edges")

    # Build child -> parents and parent -> children adjacencies
    parents:  dict[str, set[str]] = defaultdict(set)
    children: dict[str, set[str]] = defaultdict(set)
    all_nodes: set[str] = set()
    for child_uri, parent_uri in zip(rel["conceptUri"], rel["broaderUri"]):
        if not isinstance(child_uri, str) or not isinstance(parent_uri, str):
            continue
        parents[child_uri].add(parent_uri)
        children[parent_uri].add(child_uri)
        all_nodes.add(child_uri)
        all_nodes.add(parent_uri)

    # Roots: nodes that are someone's parent but never appear as a child.
    roots = {u for u in all_nodes if u not in parents}
    print(f"[graph] nodes: {len(all_nodes):,}  edges: {len(rel):,}  roots: {len(roots):,}")

    # BFS from all roots simultaneously, taking min depth via any path.
    depth: dict[str, int] = {}
    q: deque[tuple[str, int]] = deque()
    for r in roots:
        depth[r] = 1
        q.append((r, 1))

    while q:
        u, d = q.popleft()
        for c in children.get(u, ()):
            new_d = d + 1
            if c not in depth or new_d < depth[c]:
                depth[c] = new_d
                q.append((c, new_d))

    # Sanity: nodes not reached by BFS (cycles or orphans inaccessible from roots).
    missing = all_nodes - depth.keys()
    if missing:
        print(f"[warn] {len(missing)} nodes unreachable from any root "
              "(orphans / cycles); those skills will fall back to default importance")

    if not depth:
        print("ERROR: BFS produced no depths. Check the relations file.", file=sys.stderr)
        sys.exit(2)
    print(f"[depth] max_depth={max(depth.values())}  "
          f"min_depth={min(depth.values())}  mean={sum(depth.values())/len(depth):.2f}")
    return depth


def _split_alt(field) -> list[str]:
    """ESCO alt/hiddenLabels are newline-separated, occasionally pipe."""
    if not isinstance(field, str) or not field:
        return []
    parts = re.split(r"[\n|]", field)
    return [p.strip() for p in parts if p.strip()]


def step3_emit_label_depths(depth_by_uri: dict[str, int]) -> dict[str, int]:
    """For each ESCO concept, attach its depth to every label form (preferred,
    alt, hidden). Lowercase + strip to match the skill_features lookup."""
    skills = pd.read_csv(SKILLS_CSV, encoding="utf-8")
    print(f"[load] {SKILLS_CSV.name}: {len(skills):,} concepts")

    out: dict[str, int] = {}
    n_pref = n_alt = n_hidden = n_no_uri = n_no_depth = 0

    for _, row in skills.iterrows():
        uri = row.get("conceptUri")
        if not isinstance(uri, str) or not uri:
            n_no_uri += 1
            continue
        d = depth_by_uri.get(uri)
        if d is None:
            n_no_depth += 1
            continue

        # preferredLabel
        pl = row.get("preferredLabel")
        if isinstance(pl, str) and pl.strip():
            key = pl.strip().lower()
            # If two different concepts share a label, keep the SHALLOWER
            # depth (more general interpretation -- conservative).
            if key not in out or d < out[key]:
                out[key] = d
            n_pref += 1

        # altLabels (synonyms)
        for alt in _split_alt(row.get("altLabels")):
            key = alt.lower()
            if key not in out or d < out[key]:
                out[key] = d
            n_alt += 1

        # hiddenLabels
        for hid in _split_alt(row.get("hiddenLabels")):
            key = hid.lower()
            if key not in out or d < out[key]:
                out[key] = d
            n_hidden += 1

    print(f"[labels] preferredLabel covered: {n_pref:,}")
    print(f"[labels] altLabels covered:      {n_alt:,}")
    print(f"[labels] hiddenLabels covered:   {n_hidden:,}")
    print(f"[labels] concepts skipped (no URI):       {n_no_uri:,}")
    print(f"[labels] concepts skipped (no depth):     {n_no_depth:,}")
    print(f"[labels] unique label forms with depth:   {len(out):,}")
    return out


def main() -> None:
    step1_check_inputs()
    depth_by_uri = step2_compute_depths()
    depth_by_label = step3_emit_label_depths(depth_by_uri)

    if not depth_by_label:
        print("ERROR: produced no label depths.", file=sys.stderr)
        sys.exit(3)

    # Distribution snapshot
    dist: dict[int, int] = {}
    for d in depth_by_label.values():
        dist[d] = dist.get(d, 0) + 1
    max_depth = max(dist.keys())
    print(f"[dist] depth distribution (label-keyed):")
    for d in sorted(dist):
        bar = "#" * min(50, int(50 * dist[d] / max(dist.values())))
        print(f"  depth {d:>2}: {dist[d]:>6,}  {bar}")

    out = {
        "depth_by_label": depth_by_label,
        "max_depth": max_depth,
        "stats": {
            "unique_labels":    len(depth_by_label),
            "max_depth":        max_depth,
            "min_depth":        min(depth_by_label.values()),
            "mean_depth":       round(sum(depth_by_label.values()) / len(depth_by_label), 3),
            "depth_histogram":  dist,
        },
        "importance_formula": "(depth / max_depth) * 5",
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[write] {OUT_JSON}")


if __name__ == "__main__":
    main()
