"""
wipe_caches.py
==============
Delete the cached SBERT embeddings under processed/embeddings/ so the next
matching_pipeline.py run re-encodes against the fresh cleaned text.

You must run this between preprocess_pipeline.py and matching_pipeline.py
whenever any of the following changed:
  - the skill vocabulary (more phrases preserved as underscore tokens)
  - the resume title source (e.g. fix 4c)
  - the cleaning recipe (HTML stripper, character whitelist, lemmatizer)

Without wiping, matching_pipeline.py's cached_encode() sees matching shape
on the .npy files and reloads them — but the underlying text changed, so
the embeddings are silently stale.

Run:
    python "data/proccessed again/wipe_caches.py"
"""

from __future__ import annotations

from pathlib import Path

EMB_DIR = Path(__file__).resolve().parent / "processed" / "embeddings"


def main() -> None:
    if not EMB_DIR.exists():
        print(f"[skip] {EMB_DIR} does not exist")
        return

    deleted = []
    failed: list[tuple[str, str]] = []
    for npy in sorted(EMB_DIR.glob("*.npy")):
        try:
            npy.unlink()
            deleted.append(npy.name)
        except Exception as e:
            failed.append((npy.name, str(e)))

    if not deleted and not failed:
        print(f"[empty] no .npy files in {EMB_DIR}")
        return

    print(f"[wiped] {len(deleted)} cached embedding files in {EMB_DIR}:")
    for name in deleted:
        print(f"        - {name}")
    if failed:
        print(f"[failed] {len(failed)} files could not be deleted:")
        for name, err in failed:
            print(f"        - {name}: {err}")


if __name__ == "__main__":
    main()
