# Pipeline Engineering Log

Chronological log of every operation performed against the resume / job-description / O*NET preprocessing pipeline. Each entry follows the required schema (Objective → Method → Implementation → Input → Output → Reasoning). New entries are appended to the bottom of the file.

Working directory: `C:\Users\ritus\Downloads\NEW PREPROCCESS DATA`
Source files:
- [preprocess_pipeline.py](preprocess_pipeline.py)
- [tech_skills.py](tech_skills.py)
- Outputs land in `./processed/`

---

## Step 1: Dataset Discovery and Schema Inspection

**Objective:**
Establish ground truth for what columns exist in each of the five inputs before writing any preprocessing code, so the pipeline keys onto real column names rather than assumed ones.

**Method:**
Listed the working directory, then read 2–3 rows from each file via `pandas.read_csv` / `pandas.read_excel` and printed `df.columns` plus a head sample. No transformations were applied.

**Implementation Details:**
- `pandas.read_csv(path, nrows=2)` for the two CSVs.
- `pandas.read_excel(path, nrows=3)` for the three XLSX files (requires `openpyxl`).
- Bash `ls -la` for file inventory and sizing.

**Input:**
- `Resume (1).csv` (~53.7 MB)
- `training_data.csv` (~3.6 MB)
- `Skills.xlsx` (~3.2 MB)
- `Skills to Work Activities.xlsx` (~11 KB)
- `Skills to Work Context.xlsx` (~8.3 KB)

**Output:**
Confirmed schemas:
| File | Columns |
|---|---|
| `training_data.csv` | `company_name`, `job_description`, `position_title`, `description_length`, `model_response` |
| `Resume (1).csv` | `ID`, `Resume_str`, `Resume_html`, `Category` |
| `Skills.xlsx` | `O*NET-SOC Code`, `Title`, `Element ID`, `Element Name`, `Scale ID`, `Scale Name`, `Data Value`, `N`, `Standard Error`, `Lower CI Bound`, `Upper CI Bound`, `Recommend Suppress`, `Not Relevant`, `Date`, `Domain Source` |
| `Skills to Work Activities.xlsx` | `Skills Element ID`, `Skills Element Name`, `Work Activities Element ID`, `Work Activities Element Name` |
| `Skills to Work Context.xlsx` | `Skills Element ID`, `Skills Element Name`, `Work Context Element ID`, `Work Context Element Name` |

**Reasoning:**
Skipping inspection and assuming column names is the single biggest source of pipeline bugs in heterogeneous data work. A 30-second peek prevents 30 minutes of `KeyError` debugging. Reading only `nrows=2–3` keeps the I/O cost negligible on the 54 MB Resume CSV.

---

## Step 2: Environment Bootstrap

**Objective:**
Ensure all third-party libraries required by the pipeline are installed before running any text-processing code.

**Method:**
Probe-then-install: run `python -c "import bs4, lxml, nltk, pandas, pyarrow"`; on `ModuleNotFoundError` execute `pip install` for the missing set.

**Implementation Details:**
- `pip install beautifulsoup4 lxml nltk pandas pyarrow openpyxl`
- `lxml` is selected as the BeautifulSoup parser (faster and stricter than the stdlib `html.parser`, important for the resume HTML which is hand-crafted and frequently malformed).
- `pyarrow` is added because the pipeline serialises outputs as Parquet (smaller, typed, faster to reload than CSV).
- `openpyxl` is required by `pandas.read_excel` for `.xlsx`.

**Input:**
A clean Python environment with `pandas` already available but missing `bs4`, `lxml`, `nltk`.

**Output:**
Installed: `beautifulsoup4-4.14.3`, `lxml-6.1.0`, `nltk-3.9.4`, `soupsieve-2.8.3` (plus already-present `pandas`, `pyarrow`, `openpyxl`).

**Reasoning:**
A probe-import is cheap (microseconds) and avoids a no-op `pip install` on already-satisfied environments. `lxml` over `html.parser` because resumes contain inline-styled markup that `html.parser` is more tolerant of but slower at; we expect ~2k+ HTML resumes so parser speed compounds.

---

## Step 3: NLTK Resource Bootstrap

**Objective:**
Guarantee that all NLTK corpora and tokenizer/tagger models required at runtime are present, without forcing the user to download them manually.

**Method:**
A `try / except LookupError` loop in [`ensure_nltk`](preprocess_pipeline.py:42) probes for each resource and downloads only the missing ones. Called once at the top of `run()`.

**Implementation Details:**
Resources bootstrapped:
- `corpora/stopwords` → `stopwords` (English stopword list)
- `corpora/wordnet` + `corpora/omw-1.4` → WordNet lemmatizer
- `tokenizers/punkt` + `tokenizers/punkt_tab` → `word_tokenize`
- `taggers/averaged_perceptron_tagger` + `taggers/averaged_perceptron_tagger_eng` → `nltk.pos_tag`

`nltk.download(..., quiet=True)` suppresses progress bars in the log.

**Input:**
A possibly-fresh NLTK install with no downloaded data.

**Output:**
All required NLTK resources resolvable at runtime.

**Reasoning:**
NLTK 3.9 split `punkt` into `punkt` + `punkt_tab` and the POS tagger into a generic + `_eng` variant. Listing both forms makes the bootstrap forward- and backward-compatible without pinning to a specific NLTK version. Probing with `nltk.data.find` avoids re-downloading on every invocation.

---

## Step 4: Stopword Set Construction (Domain-Tuned)

**Objective:**
Produce a stopword set that strips genuinely uninformative tokens from resumes and JDs while preserving words that carry signal in the recruiting domain.

**Method:**
Start from NLTK's English stopwords, then **remove** words that are semantically meaningful here, and **add** boilerplate words that are noise in JDs/resumes.

**Implementation Details:**
[`build_stopword_set`](preprocess_pipeline.py:65):
- Base: `nltk.corpus.stopwords.words("english")` (~179 words)
- `KEEP_WORDS = {"with", "using", "experience", "in", "of", "knowledge", "skills", "ability"}` — removed from the stop set.
- `DOMAIN_STOPWORDS = {"responsible", "duties", "including", "various", "etc"}` — added to the stop set.

**Input:**
NLTK English stopword list.

**Output:**
A `set[str]` of domain-tuned stopwords used downstream by `tokenize_lemmatize`.

**Reasoning:**
- "experience", "knowledge", "skills" are *the* signal words in this domain — dropping them via a generic stopword list would erase the hooks we use for downstream skill-gap analysis.
- "with"/"using"/"in"/"of" sit between a verb and its object ("experience **with** Python", "proficiency **in** SQL"); removing them breaks phrase-level co-occurrence features.
- "responsible / duties / including / various / etc" are JD boilerplate that adds noise to TF-IDF and embedding centroids without adding meaning.

---

## Step 5: Text Cleaning (HTML, Casing, Character Filtering)

**Objective:**
Turn arbitrary raw text (HTML resumes, multi-paragraph JDs) into a flat lowercase ASCII-ish string that retains every character that carries skill information.

**Method:**
Three-stage transformation in [`clean_text`](preprocess_pipeline.py:114):
1. **HTML strip** via BeautifulSoup with the `lxml` parser ([`strip_html`](preprocess_pipeline.py:80)). `get_text(separator=" ")` to preserve word boundaries between adjacent tags (`<li>Python</li><li>SQL</li>` → `"Python SQL"`, not `"PythonSQL"`).
2. **Lowercase** via `str.lower()`.
3. **Character whitelist** via the regex `[^a-z0-9+#_\s]` — anything outside the whitelist becomes a space; runs of whitespace are then collapsed.

The whitelist explicitly preserves:
- `+` for `c++`
- `#` for `c#`
- `_` because Step 6 uses underscore-joined skill phrases
- digits because years-of-experience numbers carry signal

**Implementation Details:**
- `BeautifulSoup(text, "lxml").get_text(separator=" ")`
- Pre-compiled regexes `_ALLOWED_CHARS = re.compile(r"[^a-z0-9+#_\s]")` and `_MULTI_WS = re.compile(r"\s+")` for hot-path performance.
- All `None` / non-string / empty inputs short-circuit to `""`.

**Input:**
Raw `Resume_html` (or `Resume_str` fallback), or raw `job_description`.

**Output:**
Lowercase string with HTML stripped, non-skill punctuation removed, single-spaced.

**Reasoning:**
A whitelist regex is safer than a blacklist (`[^...]` can't accidentally let a new exotic char through). Keeping `+ # _` is the *only* way to preserve `c++` / `c#` and to make multi-word skill phrases survive tokenization. Numbers are kept because phrases like `5 years` and `python 3` matter.

---

## Step 6: Tech-Skill Alias Substitution (Pre-Cleaning)

**Objective:**
Preserve skills whose canonical written form contains characters that the cleaning regex *cannot* keep without breaking everything else (e.g. `.`, `/`, hyphens). These would otherwise be silently destroyed by `clean_text` or split apart by the tokenizer.

**Method:**
A pre-cleaning substitution pass [`apply_aliases`](preprocess_pipeline.py:106) replaces canonical multi-character forms with safe single-token aliases *before* `clean_text` runs. The reverse map is held by the `TextPipeline` instance and applied during skill extraction so users see canonical forms in the output.

**Implementation Details:**
- Alias dictionary in [tech_skills.py](tech_skills.py): `TECH_ALIASES`, e.g.
  - `"c++" → "cplusplus"`
  - `"c#" → "csharp"`
  - `".net" → "dotnet"`
  - `"node.js" → "nodejs"`
  - `"ci/cd" → "cicd"`
  - `"a/b testing" → "abtesting"`
  - 17 entries total.
- [`build_alias_patterns`](preprocess_pipeline.py:86) compiles each LHS to a regex with manual lookarounds `(?<![A-Za-z0-9_])X(?![A-Za-z0-9_])` rather than `\b`, because `+` and `#` are non-word characters and `\b` would misfire (e.g. `c++` doesn't have a word boundary on its right).
- Substitutions are sorted **longest LHS first** so `asp.net` matches before `.net`.
- Substitutions are case-insensitive (`re.IGNORECASE`).

**Input:**
Raw text (any case, possibly HTML).

**Output:**
Same text with every canonical alias swapped for its safe alphabetic form. The cleaning regex will now leave these tokens untouched.

**Reasoning:**
We can't add `.` or `/` to the cleaning whitelist — those characters appear in sentence-internal noise (`Mr.`, `and/or`) that we *do* want stripped. Pre-substitution is the cleanest way to give specific high-value tokens immunity without weakening the cleaner. The reverse map ensures users never see `cplusplus` in extraction output — they see `c++`.

---

## Step 7: Multi-Word Skill Phrase Preservation

**Objective:**
Stop the tokenizer from splitting `"machine learning"` into `["machine", "learning"]`, which would destroy phrase-level signal and force downstream embeddings to recover the bigram from context.

**Method:**
After `clean_text`, run [`preserve_skill_phrases`](preprocess_pipeline.py:143) which regex-replaces every known multi-word skill phrase (e.g. `"machine learning"`) with its underscore-joined form (`"machine_learning"`). `word_tokenize` treats this as a single token; downstream code reverses the underscore for human-readable output via [`restore_skill_phrases`](preprocess_pipeline.py:149).

**Implementation Details:**
- [`build_phrase_patterns`](preprocess_pipeline.py:128) consumes the union of all multi-word skills from O*NET and `TECH_SKILLS`.
- Phrases are sorted **by length descending** so `"deep learning"` is replaced before `"learning"`. This avoids the longest-match-loses bug where a substring phrase consumes characters its parent phrase needed.
- Pattern: `r"\b" + re.escape(phrase) + r"\b"` — word boundaries are safe here because cleaning has already stripped punctuation, so the surrounding context is always alphanum or whitespace.

**Input:**
Cleaned lowercase text.

**Output:**
Same text with every known multi-word skill phrase joined by `_`.

**Reasoning:**
This pattern (regex pre-substitution → tokenize → optional un-join) is the simplest way to make multi-word entities survive a word-level tokenizer without dragging in spaCy's `PhraseMatcher` or a full NER pipeline. The trade-off: phrase patterns are a closed list — anything not in the vocabulary won't be preserved as a phrase. Acceptable here because skill vocabularies are curated.

---

## Step 8: Tokenization + POS-Aware Lemmatization

**Objective:**
Convert the cleaned, phrase-preserved string into a list of normalized tokens suitable for embedding lookup, TF-IDF, or skill-vocabulary matching.

**Method:**
[`tokenize_lemmatize`](preprocess_pipeline.py:168):
1. `nltk.word_tokenize` → list of word tokens (underscore-joined phrases pass through intact because no whitespace is inside them).
2. `nltk.pos_tag` over the token list to attach Penn Treebank POS tags.
3. Filter: drop tokens that are in the stopword set, and drop tokens with no alphanumeric character (residual punctuation).
4. Lemmatize: tokens *without* an underscore go through `WordNetLemmatizer.lemmatize(token, pos)` where `pos` is mapped from Penn Treebank to WordNet by `_wordnet_pos`. Tokens *with* an underscore (skill phrases) bypass lemmatization to avoid mangling forms like `data_science` → `datum_science`.

**Implementation Details:**
- POS map in [`_wordnet_pos`](preprocess_pipeline.py:156): `J*→ADJ`, `V*→VERB`, `N*→NOUN`, `R*→ADV`, default `NOUN`.
- WordNet lemmatizer is instantiated once on the `TextPipeline` to avoid re-loading the corpus per row.
- Underscore tokens are detected by `if "_" in tok` and re-emitted unchanged.

**Input:**
Cleaned text with multi-word skills underscore-joined.

**Output:**
`list[str]` of lemmatized tokens; multi-word skills retain their underscore form for downstream vocabulary lookup.

**Reasoning:**
- POS-aware lemmatization is materially better than the noun-only default: `running` (verb) → `run`, but the noun-default would leave it as `running`. Resumes are verb-heavy ("managed", "designed", "implemented") so getting verb POS right matters.
- Bypassing lemmatization for underscore phrases is a deliberate trade-off: we lose the (rare) ability to lemmatize inside a phrase, but we gain bullet-proof phrase identity for the vocabulary lookup in Step 11.

---

## Step 9: Resume Section Extraction (Lightweight)

**Objective:**
Provide best-effort structured access to the three sections recruiters care about most — Skills, Experience, Education — without committing to a full resume-parsing model.

**Method:**
[`extract_resume_sections`](preprocess_pipeline.py:203) runs case-insensitive regex header patterns over the **raw** resume text (pre-cleaning, to keep header capitalisation cues), records the byte offset of each header hit, and slices `[start, next_start)` for each section.

**Implementation Details:**
- `SECTION_PATTERNS`:
  - `skills` ← `r"\b(skills|technical skills|core competencies)\b"`
  - `experience` ← `r"\b(experience|work history|employment|professional experience)\b"`
  - `education` ← `r"\b(education|academic background|qualifications)\b"`
- All headers across all sections are merged and sorted by start position.
- Only the **first** occurrence of each section is retained (resumes often repeat headers in PDFs/HTML duplicates).
- All three keys are always present in the output dict, defaulting to `""`.

**Input:**
Raw resume text (HTML- or plain-text-derived, prior to cleaning).

**Output:**
`dict` with keys `skills`, `experience`, `education`. Three new dataframe columns: `section_skills`, `section_experience`, `section_education`.

**Reasoning:**
A regex-header heuristic captures ~70–80% of well-formatted resumes for free. The trade-off is recall on creatively-formatted resumes (e.g. headers as images, or non-English headers). Acceptable here because the cleaned full text is also retained — sections are an *additional* feature, not a replacement.

---

## Step 10: O*NET Skill Vocabulary Construction

**Objective:**
Convert the three O*NET spreadsheets into three machine-friendly JSON artefacts — a name→ID dictionary, a skill→activities mapping, and a skill→context mapping enriched with importance/level scores.

**Method:**

[`process_skills_xlsx`](preprocess_pipeline.py:253):
- Lowercase + strip whitespace on `Element Name`.
- `skill_dictionary = {Element Name: Element ID}` after de-dup.
- Aggregate `Data Value` per `(Element Name, Scale ID)` with `groupby().mean()`, then `unstack` the `Scale ID` axis to produce one row per skill with two columns:
  - `IM` → `importance_mean`
  - `LV` → `level_mean`
- Join in the per-skill record count `n_records`.
- Mean is taken across all `O*NET-SOC` codes — i.e. the average importance/level a skill has across every occupation that requires it.

[`process_skill_to_activity`](preprocess_pipeline.py:284):
- Lowercase + strip skill name.
- Build `defaultdict(list)` of `skill → activities`, then de-dup with `sorted(set(...))` for deterministic output.

[`process_skill_to_context`](preprocess_pipeline.py:295):
- Same shape as the activity processor.

In `run()`, the context map is enriched with the importance/level means from the skills processor, producing a per-skill object: `{context_features: [...], importance_mean: float, level_mean: float}`.

**Implementation Details:**
- `pandas.read_excel` (openpyxl backend).
- `groupby().mean().unstack()` for the wide-form pivot.
- `json.dump(..., indent=2)` for human-readable artefacts.

**Input:**
- `Skills.xlsx` — 35 distinct O*NET skills × ~870 SOC codes × 2 scales (Importance, Level)
- `Skills to Work Activities.xlsx` — 32 skills mapped to ~15 activities each
- `Skills to Work Context.xlsx` — 35 skills mapped to context features

**Output:**
- `processed/skill_dictionary.json` — `{skill_name: element_id}` (35 entries)
- `processed/skill_to_activity_map.json` — `{skill_name: [activities]}`
- `processed/skill_to_context_map.json` — `{skill_name: {context_features, importance_mean, level_mean}}`

**Reasoning:**
O*NET ships skills denormalised across thousands of rows (one per SOC × Scale). A single-row-per-skill aggregation is the right shape for downstream lookup and for joining into JD/resume features. Mean-across-SOC is a defensible default; if a downstream task is occupation-specific, the raw `Skills.xlsx` can still be re-aggregated by SOC.

---

## Step 11: Skill Extraction (Two-Source Vocabulary Lookup)

**Objective:**
For every resume and JD, return the structured list of skills that appear in the text, tagged by which vocabulary they came from (O*NET vs. tech).

**Method:**
[`extract_skills`](preprocess_pipeline.py:227) iterates the lemmatized token list and:
1. Replaces underscore with space (`machine_learning` → `machine learning`) so the candidate matches the canonical form stored in the vocabularies.
2. Reverse-aliases tech-alias forms (`cplusplus` → `c++`) using the alias map populated at pipeline construction.
3. Looks up the candidate in the O*NET vocab first (priority), falls back to the tech vocab.
4. Returns a list of `{"skill": str, "source": "onet"|"tech"}` records, sorted by skill name with duplicates collapsed.

**Implementation Details:**
- Vocab membership is `set` lookup → O(1) per token.
- O*NET wins ties because O*NET names are more strictly defined; a skill present in both vocabs (e.g. "programming") is reported as `onet`.
- Reverse alias map is built once per pipeline as `{v: k for k, v in TECH_ALIASES.items()}`.

**Input:**
Lemmatized token list from Step 8.

**Output:**
`list[dict]` written to the `extracted_skills` column of both output dataframes.

**Reasoning:**
Tagging the source lets downstream code weight or filter by vocabulary — useful when O*NET-style soft skills shouldn't compete with hard tech skills in the same ranker. Vocabulary lookup is cheaper and more transparent than learned NER for a known closed set; the precision/recall ceiling is the vocabulary itself, which is in the user's hands.

---

## Step 12: Tech Skill Vocabulary (Curated)

**Objective:**
Layer technical skills (Python, TensorFlow, Kubernetes, …) on top of O*NET, which only ships 35 generic competencies and would otherwise produce near-empty `extracted_skills` for tech resumes.

**Method:**
A curated, hand-grouped vocabulary in [tech_skills.py](tech_skills.py) consisting of 10 thematic blocks (`_LANGUAGES`, `_WEB_FRAMEWORKS`, `_DATA_ML`, `_BIG_DATA`, `_DATABASES`, `_CLOUD`, `_DEVOPS`, `_TOOLS`, `_MOBILE`, `_CONCEPTS`) unioned into `TECH_SKILLS: set[str]`. A separate `TECH_ALIASES` dict (used by Step 6) handles canonical forms that contain non-tokenizer-friendly characters.

**Implementation Details:**
- All entries are lowercase, single-space-delimited (e.g. `"node js"`, not `"Node.js"` — that goes through `TECH_ALIASES`).
- Aliases' RHS values (`cplusplus`, `csharp`, …) are also unioned into `TECH_SKILLS` so that the post-substitution token can be matched directly.
- The pipeline merges this set with the O*NET vocab to form `combined_vocab` (390 unique skills total: 35 O*NET + 355 tech) used by Step 7's phrase patterns and Step 11's lookup.

**Input:**
Hard-coded set literals in [tech_skills.py](tech_skills.py).

**Output:**
- `TECH_SKILLS` exposed as a Python `set[str]`.
- `TECH_ALIASES` exposed as a Python `dict[str, str]`.
- Persisted as `processed/skill_dictionary_merged.json` with shape `{skill_name: {"id": str|null, "source": "onet"|"tech"}}`.

**Reasoning:**
A curated dictionary is the fastest path from "no tech extraction" to "decent tech extraction" without training data. The trade-off versus a model-based extractor (spaCy NER, fine-tuned transformer) is poor handling of single-word generic terms (`go`, `r`, `swift`, `less`, `express` collide with English) — this is a known precision issue and is documented in Step 13 as a follow-up.

---

## Step 13: Pipeline Orchestration and Output Materialization

**Objective:**
Tie the per-row transformation steps into a single dataset-level pipeline that reads the five inputs, applies the transformations, and writes deterministic, schema-stable outputs.

**Method:**
[`TextPipeline`](preprocess_pipeline.py:309) bundles the per-row state (stopword set, lemmatizer, alias patterns, phrase patterns, vocabularies) and exposes a single `process(raw: str) -> dict` method. [`run`](preprocess_pipeline.py:340) orchestrates:
1. Bootstrap NLTK resources.
2. Process O*NET → write three JSON artefacts + merged dictionary.
3. Apply `pipeline.process` to `Resume_html` (falling back to `Resume_str` when HTML is empty), attach section columns, write `cleaned_resumes.parquet`.
4. Apply `pipeline.process` to `job_description`, write `cleaned_jobs.parquet`.

**Implementation Details:**
- `pipeline.process` is invoked via `pandas.Series.apply`. Vectorised regex would be marginally faster but at the cost of harder per-row debugging; current throughput (~10 rows / second on the full HTML resume CSV) is acceptable.
- Resume HTML is preferred over plain text because `Resume_html` retains formatting cues (headers, lists) that `strip_html` then converts into clean spaced text. `Resume_str` is the fallback when `Resume_html` is empty/null.
- Output format is Parquet (typed, columnar, ~5–10× smaller than CSV for this content); the `pyarrow` engine is the default.
- A `--sample N` CLI flag short-circuits to the first N rows for smoke testing.

**Input:**
All five source files in the working directory.

**Output (written to `processed/`):**
- `cleaned_resumes.parquet` — adds `cleaned_resume`, `tokenized_text`, `extracted_skills`, `section_skills`, `section_experience`, `section_education`.
- `cleaned_jobs.parquet` — adds `cleaned_job_description`, `tokenized_text`, `extracted_skills`.
- `skill_dictionary.json`, `skill_to_activity_map.json`, `skill_to_context_map.json`, `skill_dictionary_merged.json`.

**Reasoning:**
A class-level pipeline lets us pay regex compilation cost once and amortise it across thousands of rows. Keeping the orchestrator in `run()` (rather than inlining at module level) makes the pipeline importable and unit-testable. Parquet over CSV because the `tokenized_text` and `extracted_skills` columns are nested — Parquet preserves them natively; CSV would force JSON-encoded strings.

---

## Step 14: End-to-End Smoke Test (sample = 5, sample = 10)

**Objective:**
Verify the full pipeline executes against real data and that outputs are non-empty and structurally correct, before committing to a full-corpus run on 50+ MB of resumes.

**Method:**
- `python preprocess_pipeline.py --sample 5` (initial wiring).
- `python preprocess_pipeline.py --sample 10` (after tech-skill layer added).
- Loaded both Parquet outputs back via `pandas.read_parquet` and printed `extracted_skills` per row.
- Ran a synthetic-input regression test through `TextPipeline.process` directly:
  > `"I know C++, C#, Node.js, .NET, machine learning and CI/CD pipelines on AWS."`
- Inspected JSON artefacts for size and shape.

**Input:**
First 5 / first 10 rows of each source file.

**Output:**
- All three JSON files written, shapes confirmed: 35 O*NET skills, 32 activity entries, 35 context entries.
- Synthetic input round-tripped to: `["machine learning", ".net", "aws", "c#", "c++", "ci/cd", "node.js"]` — confirming alias substitution and reverse mapping work end-to-end.
- Pipeline reported: `merged vocab: 35 O*NET + 355 tech = 390 unique skills`.
- **Known issue surfaced:** single-word ambiguous tech tokens produce false positives (`r`, `go`, `less`, `express`, `swift`, `storm`, `segment`, `sketch`). Logged as a precision/recall trade-off; three remediation paths drafted (context-window match, raw-case match, spaCy `PhraseMatcher`) — pending user decision before implementation.

**Reasoning:**
Smoke tests on a sample catch wiring bugs (KeyError, schema mismatch) at near-zero cost; running the full 27k-row resume corpus first would have wasted minutes per failure. The synthetic input is the fastest way to check that the most fragile transformation (alias substitution for `+`/`#`/`.`/`/` characters) survives the full chain.

---

## Step 15: Full-Corpus Preprocessing Run

**Objective:**
Materialize the cleaned outputs against the entire input corpus so downstream stages have real data to operate on (the 10-row smoke tests in Step 14 only verified wiring).

**Method:**
Ran `python preprocess_pipeline.py` (no `--sample` flag) end-to-end with all transformations defined in Steps 4–13.

**Implementation Details:**
- Background bash invocation; tee'd to `preprocess_full.log` for retention.
- No code changes — the same `TextPipeline` class is amortised across the full corpus.

**Input:**
- `Resume (1).csv` — 2484 rows
- `training_data.csv` — 853 rows
- O*NET Skills + Skills→Activities + Skills→Context

**Output:**
- `processed/cleaned_resumes.parquet` — 2484 rows × 10 columns
- `processed/cleaned_jobs.parquet` — 853 rows × 8 columns
- `processed/skill_dictionary.json`, `processed/skill_dictionary_merged.json`, `processed/skill_to_activity_map.json`, `processed/skill_to_context_map.json`

**Reasoning:**
Full-corpus materialization is needed before the matching pipeline (Step 16) can build embeddings. Running it once and caching as Parquet is the right shape: all subsequent steps re-read cheaply rather than re-doing HTML stripping / lemmatization on every iteration.

---

## Step 16: SBERT Embedding Generation (Cached)

**Objective:**
Produce dense semantic representations for resumes, job descriptions, resume experience sections, resume titles, and job titles — once — so they can be reused by Stage 1 retrieval and Stage 2 feature extraction without re-encoding.

**Method:**
[`encode_texts`](matching_pipeline.py:138) calls `SentenceTransformer.encode` with batching (`batch_size=64`), `convert_to_numpy=True`, and `normalize_embeddings=True`. The normalization is critical: it makes cosine similarity equivalent to a single matrix multiply (`A @ B.T`) downstream.

[`cached_encode`](matching_pipeline.py:255) wraps `encode_texts` with disk-backed caching: `processed/embeddings/{name}.npy` is checked first; if shape matches, the embedding is loaded instead of recomputed. Cache is bypassed when `--sample` is used (sample mode shouldn't poison the canonical cache).

**Implementation Details:**
- Library: `sentence-transformers` (HuggingFace bi-encoder).
- Model: `all-MiniLM-L6-v2` (384-d, ~22M params, ~3 ms / sentence on CPU).
- Storage: `np.float32` `.npy` files. Resume embedding matrix is 2484 × 384 × 4 B ≈ 3.8 MB.
- Five embedding tensors are produced and cached:
  | Name | Source | Shape |
  |---|---|---|
  | `resume_emb` | `cleaned_resume` | (2484, 384) |
  | `job_emb` | `cleaned_job_description` | (853, 384) |
  | `resume_exp_emb` | `section_experience` | (2484, 384) |
  | `resume_title_emb` | `Category` | (2484, 384) |
  | `job_title_emb` | `position_title` | (853, 384) |

**Input:**
Five `list[str]` columns from the cleaned Parquet outputs.

**Output:**
Five `(N, 384)` `numpy.float32` arrays, persisted under `processed/embeddings/`.

**Reasoning:**
- `all-MiniLM-L6-v2` chosen over larger models (e.g. `mpnet`) because the corpus is small (3.3k texts), so the speedup matters more than the marginal quality gain. Easy to swap by changing the `SBERT_MODEL` constant.
- Pre-normalization moves the cost of `1/||x||` from per-pair to per-vector — a 2.7M× saving when computing the full job × resume similarity matrix.
- File-level caching pays off the second time the script runs (e.g. when only feature-extraction parameters change). Sample-mode cache bypass prevents a 10-row run from corrupting the full 2484-row cache.

---

## Step 17: Top-K Retrieval (Stage 1)

**Objective:**
For each of the 853 jobs, identify the K=50 most similar resumes by cosine similarity — reducing 853 × 2484 = 2.12 M candidate pairs to 853 × 50 = 42,650 pairs (a **50× reduction**).

**Method:**
[`topk_retrieval`](matching_pipeline.py:148):
1. Compute the full similarity matrix `sims = job_emb @ resume_emb.T` — shape `(853, 2484)`. Both inputs are pre-normalized in Step 16, so the dot product is exact cosine similarity.
2. For each row, use `np.argpartition(-sims, k-1, axis=1)[:, :k]` to pull the top-K indices in O(n) per row instead of O(n log n) for a full sort.
3. Sort just those K with `np.argsort` to produce ranked output.

**Implementation Details:**
- Pure NumPy — no FAISS / hnswlib dependency, because the matrix is only 2.12 M floats (~8.5 MB) and an exact computation is cheaper than ANN setup at this scale.
- Memory: 2.12 M × 4 B = 8.5 MB peak for `sims`; the partition / argsort intermediates are bounded by `(n_jobs × k) ≈ 43k` ints = 170 KB.
- Numerical: cosine values are in `[-1, 1]`; we negate before partitioning because NumPy returns smallest-K, not largest-K.

**Input:**
- `job_emb`: `(853, 384)`, normalized.
- `resume_emb`: `(2484, 384)`, normalized.
- `k = 50`.

**Output:**
- `topk_idx`: `(853, 50)` resume indices, descending similarity.
- `topk_scores`: `(853, 50)` cosine values aligned to `topk_idx`.

**Reasoning:**
At this scale, exact retrieval is the right call:
- Avoids approximate-recall errors that ANN structures introduce — irrelevant in a 2.5k-document index.
- One matmul is a single BLAS call, which CPython + NumPy hand off to OpenBLAS / MKL: faster wall-clock than building a FAISS index for ≤10k vectors.
- argpartition over argsort matters when K << N: at K=50, N=2484, partition is ≈ 50× cheaper than full sort. Sorting only the K survivors restores ranked order at negligible cost.

---

## Step 18: Reduced Pair Set Construction

**Objective:**
Materialize the (job_id, resume_id) pair index that all Stage-2 feature extractors will operate on, so each subsequent feature is a vectorized array op rather than nested loops over the cosine matrix.

**Method:**
- `pairs_job = np.repeat(np.arange(n_jobs), k)` — flattens the row axis: `[0,0,...,0, 1,1,...,1, ...]`.
- `pairs_res = topk_idx.reshape(-1)` — flattens the column axis aligned to `pairs_job`.
- `pairs_sbert = topk_scores.reshape(-1)` — pre-computed Stage-1 similarity becomes the first feature, no recomputation.

**Implementation Details:**
Pure NumPy reshape; O(1) memory beyond the existing arrays (views, not copies, where possible).

**Input:**
`topk_idx`, `topk_scores` from Step 17.

**Output:**
Three aligned 1-D arrays of length 42,650: `pairs_job`, `pairs_res`, `pairs_sbert`.

**Reasoning:**
A flat index lets every subsequent feature be expressed as a single fancy-index op (`feature[pairs_job]`, `feature[pairs_res]`) rather than per-pair Python loops. This is the critical performance hinge — moving the inner loop into NumPy.

---

## Step 19: TF-IDF Pair Similarity (Stage 2 — Lexical Channel)

**Objective:**
Add a lexical-overlap signal that complements SBERT's semantic channel. SBERT can score a resume that talks "about" the same topic as the JD even when no shared vocabulary exists; TF-IDF catches the converse — exact term overlap that semantic embedding can sometimes blur.

**Method:**
[`tfidf_pair_similarities`](matching_pipeline.py:181):
1. Fit a single `TfidfVectorizer` on the concatenated `resumes + jobs` corpus (so the vocabulary and IDF are jointly defined).
2. Transform each side separately into sparse matrices `R` and `J`.
3. Row-normalize both with `sklearn.preprocessing.normalize` so cosine = element-wise product summed over axis 1.
4. Iterate the 42,650 pairs in batches of 4096: `J[j_idx].multiply(R[r_idx]).sum(axis=1)` — sparse element-wise product followed by row-sum, never materializing a dense `(N_pair × vocab)` matrix.

**Implementation Details:**
- Library: `scikit-learn` (`TfidfVectorizer`, `normalize`).
- Parameters: `max_features=20_000`, `ngram_range=(1, 2)`, `min_df=2`, `sublinear_tf=True`.
- Sparse-product trick: `csr.multiply(csr).sum(axis=1)` runs in O(nnz_intersection), not O(vocab).

**Input:**
`resume_text`, `job_text`, `pairs_job`, `pairs_res`.

**Output:**
`tfidf_sims`: `np.float32` array of shape `(42650,)` — cosine similarity in `[0, 1]`.

**Reasoning:**
- Bigrams (`ngram_range=(1,2)`) capture compound terms ("data science", "machine learning") that SBERT already handles via embedding proximity but TF-IDF only catches when present verbatim. The two channels disagree most where the labels live — useful signal for downstream rankers.
- `sublinear_tf=True` (i.e. `log(1 + tf)`) damps the influence of repeated terms, important on resumes that repeat a skill in every bullet point.
- `min_df=2` drops hapaxes (typos, names) that would otherwise inflate dimensionality without information.
- Batch-of-pairs sparse multiply observed at ≈ 50 k pairs/sec on this corpus, dominated by argpartition rather than the multiply.

---

## Step 20: Skill Overlap, Weighted Match, and Missing-Skill Features

**Objective:**
Produce skill-level features that are explainable (which skills are missing? which are present?) and weighted by O*NET-derived importance so generic skills don't dominate over critical ones.

**Method:**
[`skill_features`](matching_pipeline.py:204):
- `skill_overlap = |R ∩ J| / |J|`  (resume coverage of the job's required skills).
- `weighted_skill_score = Σ importance(R ∩ J) / Σ importance(J)` — same shape as overlap but each skill is weighted by the O*NET importance score (mean over all SOC codes).
- `num_missing_skills = |J − R|` — count of skills the job requires that the resume lacks.
- `avg_missing_skill_importance = mean(importance(J − R))` — weighted version of the gap.

[`build_importance_map`](matching_pipeline.py:248):
- O*NET skills get `importance_mean` from `skill_to_context_map.json` (range 0–5).
- Tech-vocab skills (no O*NET importance) fall back to `default_importance = mean(O*NET importances)` — ≈ 2.5. This prevents tech skills from being silently zero-weighted (which would eliminate them from `weighted_skill_score`) without letting them dominate.

**Implementation Details:**
- Skill sets are precomputed once per resume / job; pair iteration only does set intersection / difference.
- `extracted_skills` columns may arrive as `list[dict]`, `numpy.ndarray`, or string-encoded; [`parse_skill_field`](matching_pipeline.py:35) normalizes all three.

**Input:**
- `job_skill_sets`, `resume_skill_sets` (from Step 11 of preprocessing).
- `pairs_job`, `pairs_res`.
- `importance_map` keyed by canonical lowercase skill name.

**Output:**
Four `np.ndarray`s aligned to the pair index:
- `skill_overlap` (`float32`, `[0, 1]`)
- `weighted_skill_score` (`float32`, `[0, 1]`)
- `num_missing_skills` (`int32`)
- `avg_missing_skill_importance` (`float32`, `[0, 5]`)

**Reasoning:**
Two parallel measurements (`skill_overlap` and `weighted_skill_score`) provide both an explainable raw count and an importance-aware version. Downstream rankers can use either or both; missing-skill features are what makes the output **explainable** — they tell the candidate what to learn next.

---

## Step 21: Experience Features

**Objective:**
Capture three orthogonal experience signals: how much experience the candidate has, how that compares to the JD's required experience, and how semantically close the candidate's experience section is to the job description.

**Method:**

**Years-of-experience** ([`extract_years`](matching_pipeline.py:71)):
- Two regex patterns: `r"(\d+(?:\.\d+)?)\s*\+?\s*(?:to\s*\d+\s*)?years?\b"` and `r"(\d+(?:\.\d+)?)\s*yrs?\b"`.
- Multiple matches → take the maximum (resumes mention many durations; JDs typically state the minimum).
- Cap at 50 to reject misparses (e.g. `"1995 years"` from a date string).
- Extracted on **raw** text (`Resume_str`, `job_description`) because dates are stripped by cleaning — the YoE pattern needs `"5 years"` to be intact.

**Experience gap:** `job_yoe - resume_yoe` per pair. Negative = candidate over-qualified; positive = under-qualified.

**Experience relevance:** Cosine similarity between `resume_exp_emb[r]` (Step 16, encoded from `section_experience`) and `job_emb[j]`. Captures whether the candidate's experience section is *about* the same domain as the JD, independent of years.

**Implementation Details:**
- All three features are computed by fancy-indexing into precomputed arrays — no per-pair Python loops.
- `experience_gap` is `float32`; it can be negative (and frequently is, because JD YoE often parses as 0 — see "Known Issues" below).

**Input:**
- `resume_yoe` (`(2484,)`), `job_yoe` (`(853,)`).
- `resume_exp_emb` (`(2484, 384)`), `job_emb` (`(853, 384)`).
- `pairs_job`, `pairs_res`.

**Output:**
- `years_of_experience` (`float32`)
- `experience_gap` (`float32`)
- `experience_relevance_score` (`float32`, `[-1, 1]`)

**Reasoning:**
A single-feature view of "experience" misses two failure modes:
1. **Years match, domain doesn't** — 10 years of accounting ≠ 10 years of ML. `experience_relevance_score` covers this.
2. **Domain matches, years don't** — junior ML engineer applying for staff role. `experience_gap` covers this.

Three features let the downstream model trade them off rather than collapsing them into a single arbitrary scalar.

---

## Step 22: Title Similarity

**Objective:**
Provide a fast, high-signal feature that captures whether the candidate's most-recent role and the job's posted title are semantically similar (e.g. "Senior Backend Engineer" vs. "Software Engineer III").

**Method:**
Cosine similarity between `title_resume_emb[r]` and `title_job_emb[j]` for each pair. Both are SBERT embeddings of short title strings (Step 16).

**Implementation Details:**
- Resume "title" is taken from the `Category` column (e.g. `INFORMATION-TECHNOLOGY`, `HR`, `TEACHER`) — this dataset doesn't expose a parsed job title field.
- Job title is `position_title` from `training_data.csv`.
- Vectorized: `(title_resume_emb[pairs_res] * title_job_emb[pairs_job]).sum(axis=1)`.

**Input:**
- `title_resume_emb` (`(2484, 384)`), `title_job_emb` (`(853, 384)`).

**Output:**
- `title_similarity` (`float32`, `[-1, 1]`).

**Reasoning:**
Title similarity is the most concentrated signal in the feature set — short text, high lexical overlap when matched. It's also the cheapest feature to compute (one matmul). The `Category` proxy is a known approximation; if a downstream task wants a more granular resume title, parsing the first line of `cleaned_resume` or the `section_experience` header would be a drop-in replacement.

---

## Step 23: Education Match

**Objective:**
Boolean indicator of whether the candidate's highest degree level meets or exceeds what the job requires.

**Method:**
- [`extract_degree_level`](matching_pipeline.py:90) maps regex hits to ordinal levels: `PhD = 3`, `Masters = 2`, `Bachelors = 1`, none = 0.
- Resume education is extracted from `section_education` (Step 9 produces this).
- Job education is extracted from raw `job_description` (degree mentions are usually flat sentences, not in a structured section).
- Match logic: `(job_edu == 0) | (resume_edu >= job_edu)`. The first clause is critical — if the JD states no degree requirement, the candidate can't fail education match.

**Implementation Details:**
- Degree regexes cover common abbreviations: `Ph.D` / `PhD` / `doctorate`, `M.S.` / `M.A.` / `MBA` / `M.Tech`, `B.S.` / `B.A.` / `B.Tech` / `Bachelor's`.
- Output is `int8` (0 or 1) — 1 byte per pair vs. 4 for `int32`.

**Input:**
- `resume_edu` (`(2484,)`, ordinal 0–3), `job_edu` (`(853,)`, ordinal 0–3).

**Output:**
- `education_match` (`int8`, 0 or 1).

**Reasoning:**
Education is asymmetric: under-qualifying disqualifies, over-qualifying doesn't. A simple `>=` check encodes this. The "no requirement = always match" rule prevents penalizing candidates against JDs that simply don't mention education — which is most of them.

---

## Step 24: Output Materialization (Pair Feature Table)

**Objective:**
Persist the 42,650-row pair feature table in a format that's ready for ranking, model training, and explainability tooling.

**Method:**
Assemble all per-pair feature arrays into a `pandas.DataFrame` keyed by `(job_id, resume_id)` and write to Parquet.

**Implementation Details:**
- All numeric features are `float32` / `int32` / `int8` to minimize disk footprint.
- Output path: `processed/pair_features.parquet`.
- Final schema: 13 columns matching the spec exactly:
  ```
  job_id, resume_id,
  embedding_similarity, tfidf_similarity,
  skill_overlap, weighted_skill_score,
  num_missing_skills, avg_missing_skill_importance,
  years_of_experience, experience_gap, experience_relevance_score,
  title_similarity, education_match
  ```

**Input:**
All per-pair NumPy arrays from Steps 17–23.

**Output:**
- `processed/pair_features.parquet` — 42,650 rows × 13 columns, ~ 2.5 MB on disk.

**Run-time summary statistics (full corpus):**

| Feature | Mean | Std | Min | Max |
|---|---|---|---|---|
| embedding_similarity | 0.655 | 0.091 | 0.130 | 0.890 |
| tfidf_similarity | 0.080 | 0.032 | 0.000 | 0.317 |
| skill_overlap | 0.129 | 0.282 | 0.000 | 1.000 |
| weighted_skill_score | 0.127 | 0.281 | 0.000 | 1.000 |
| num_missing_skills | 1.43 | 2.53 | 0 | 27 |
| avg_missing_skill_importance | 1.53 | 1.36 | 0.00 | 3.60 |
| years_of_experience | 4.59 | 7.56 | 0 | 50 |
| experience_gap | -4.59 | 7.56 | -50 | 0 |
| experience_relevance_score | 0.276 | 0.116 | -0.052 | 0.723 |
| title_similarity | 0.280 | 0.155 | -0.102 | 1.000 |
| education_match | 0.763 | 0.425 | 0 | 1 |

**Reasoning:**
Parquet preserves dtypes (no `int8` → `int64` promotion) and is ~5× smaller than CSV. The DataFrame is the right boundary between the offline pipeline and the model-training / serving stages — re-loading is one `pd.read_parquet` call.

---

## Known Issues After Full Run

1. **`experience_gap` is always ≤ 0.** Mean is `-4.59` and the *max* is exactly 0. This indicates JD-side YoE extraction is failing systematically — most JDs in `training_data.csv` either don't state a YoE requirement explicitly or state it in a phrasing the regex doesn't catch (e.g. "5+ years required" lives in a structured `model_response` column, not the free-text JD). Resume-side YoE works (mean 4.6, std 7.6, hits 50 cap).

   *Remediation options:*
   - Parse `model_response` for YoE (it appears to be structured LLM output).
   - Add patterns for "minimum X years", "X+ years experience", "X-Y years".

2. **Single-token skill false positives** (carried forward from Step 14). Skills like `r`, `go`, `less`, `express`, `swift` collide with English. Same remediation paths apply.

3. **Resume "title" is the `Category` column** — coarse (e.g. all SWE-adjacent resumes are `INFORMATION-TECHNOLOGY`). A more granular title would lift `title_similarity`'s separating power.

---

## Next Steps

1. **Address JD-side YoE extraction** (likely the highest-impact fix).
2. **Train a learned ranker** on `pair_features.parquet`. Without ground-truth labels, candidates: (a) self-supervised (synthetic positives via LLM judging), (b) graded relevance from `model_response`, (c) reciprocal-rank-fusion of feature-weighted heuristics.
3. **Add a hard-negative mining loop** — currently `topk_retrieval` only returns positives; for ranker training we'll also need bottom-K or random-negative samples.
4. **Skill-gap report API** — the data is already in place; surface it as `for_each_resume(top_k_jobs) → missing_skills` so users can see what to learn.
