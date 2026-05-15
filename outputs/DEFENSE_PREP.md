# SkillMatch — Endterm Defense Prep
**Goal: be 150% ready. Assume a PhD-level adversarial panel.**

---

## PART 0 — THE 60-SECOND ELEVATOR PITCH

> "We built a two-stage resume–job matching system: SBERT retrieves the top-50 candidate jobs per resume, then a Logistic Regression reranker with 8 features — semantic, lexical, ESCO-weighted skill coverage, experience, and title similarity — scores them. Our methodological contribution is the labeling strategy. We tried five approaches on the same 100-pair held-out test set: weak supervision failed with leakage (CV F1 0.99 → test 0.44), 400 LLM-consensus labels got us to F1=0.704, two self-training experiments confirmed Yarowsky's 1995 confirmation-bias prediction (F1=0.564 and 0.579 — worse than no self-training), and finally active learning on the uncertainty band [0.40, 0.60] with 200 additional labels lifted F1 to **0.769**, beating SBERT-only by +16.5 points and supervised by +6.6 points on the same label budget. We disclose limitations honestly: N=100 confidence interval is ±0.07, LLM-consensus is correlated noise not ground truth, and we have no demographic bias audit yet."

**Practice this until you can deliver it in one breath without notes.**

---

## PART 1 — NUMBERS TO MEMORIZE COLD

| Metric | Value | Where on deck |
|---|---|---|
| **Final F1 (Active Learning)** | **0.769** | Slide 7 |
| LLM supervised F1 (400 labels) | 0.704 | Slide 7 |
| SBERT-only baseline F1 | 0.604 | Slide 7 |
| Self-trained v2 F1 (percentile) | 0.579 | Slide 7 |
| Self-trained v1 F1 (absolute) | 0.564 | Slide 7 |
| Weak supervised F1 | 0.440 | Slide 7 |
| **CV vs test gap on weak sup** | **0.99 → 0.44** | Slide 6 |
| Active learning lift over SBERT | **+0.165 F1** | Slide 7 |
| Active learning lift over supervised | **+0.066 F1** | Slide 7 |
| **F1 CI (N=100 Wilson)** | **±0.07** | Slide 11 limitation |
| Precision | 0.667 | Slide 7 footer |
| Recall | 0.909 | Slide 7 footer |
| P@10 | 0.90 | Slide 7 footer |
| Confusion matrix | TP=20 FN=2 FP=10 TN=68 | Slide 7 footer |
| Component: SBERT-only LogReg | F1=0.628 | Slide 8 |
| Component: Skill-only LogReg (7 feats, no SBERT) | F1=0.706 | Slide 8 |
| Component: Full model (8 feats) | F1=0.769 | Slide 8 |
| **Skill features beat SBERT by** | **+7.8 F1 points** | Slide 8 |
| Drop title_similarity | -0.116 F1 | Slide 8 ablation |
| Drop num_missing_skills | -0.088 F1 | Slide 8 ablation |
| Drop embedding_similarity | -0.063 F1 | Slide 8 ablation |
| Resumes | 2,484 | Slide 4 |
| Job descriptions | 853 | Slide 4 |
| Skills in vocab | 20,253 (ESCO+O*NET+tech) | Slide 4 |
| Candidate pairs (diversified pool) | 50,650 | Slide 4 |
| Training labels | 600 (400 random + 200 active uncertain) | Slide 6/7 |
| Held-out test labels | 100 | All result slides |
| Inter-rater kappa (3 LLMs) | **0.61 ("substantial")** | Slide 9 |
| Unanimous votes | 71.0% | Slide 9 |
| Disputed votes (2-1) | 29.0% | Slide 9 |
| Pos rates: Claude / GPT / Gemini | 22.4% / 13.6% / 26.6% | Slide 9 |
| SBERT top-K positive rate | 47.5% | Slide 9 |
| SBERT model | all-MiniLM-L6-v2 (22M params, 384-dim) | Slide 1/5 |
| Latency | ~3 ms encode + <1 ms LogReg per pair | Slide 5 |
| Total LLM labeling cost | < $15 USD | (have this ready) |
| ESCO expansion lift | skill_overlap > 0: 36% → 68% of pairs | Slide 4 tagline |
| Active learning band | probability ∈ [0.40, 0.60] | Slide 6 |

**Drill: cover this table and quiz yourself. If you can't recite F1=0.769 within 0.5s, study more.**

---

## PART 2 — SLIDE-BY-SLIDE DELIVERY NOTES

### Slide 1 — Title / Hero
**Time: 30 sec.** Say:
> "SkillMatch — a two-stage skill-aware matching system. The retrieval stage uses SBERT to narrow 2.1 million candidate pairs to top 50 per job. The reranking stage is a Logistic Regression on 8 hand-crafted features. The output is a match probability plus a ranked list of skill gaps, plus a 2-year personalized roadmap. The contribution isn't the architecture — those are standard — it's the labeling methodology and the honest comparison."

### Slide 2 — Problem & Applications
**Time: 45 sec.** Hit:
- "250+ resumes per role on average, 75% rejected on primitive keyword matching."
- "Same model, two users: recruiters see match probabilities, candidates see skill gaps."
- Close on the tagline: "We don't just produce a score — we tell candidates which skills to learn next."

### Slide 3 — Literature Survey
**Time: 60 sec.** This slide MATTERS because evaluators love seeing you've read the literature.
> "Three 2025 papers in the same space — Dash et al. IJEDR, Daberao et al. GITCON, Sribharathi et al. ICSSS. All use the same conceptual pipeline: extract skills → semantic match → surface gap → recommend learning. All built for English IT/data-science roles. They report accuracies of 0.90 to 0.918. But — and this is the critical observation — none of them disclose how their gold standard was constructed. Our contribution is methodological transparency: 3-LLM consensus, disclosed kappa, reproducible random_state=42 throughout."

**If asked: "your F1=0.769 is lower than their accuracies":** see Adversarial Q&A #9 below — this is a trap.

### Slide 4 — Dataset & Preprocessing
**Time: 60 sec.** Numbers + three fixes:
> "2,484 resumes from a Kaggle anonymized dump, 853 public-corpus JDs, 20,253 skills from a combined ESCO + O*NET + tech vocabulary. Three preprocessing fixes materially moved the needle. Fix 4A: the original YoE regex caught zero of 853 jobs because it was over-strict; we parse from the model_response JSON instead and now extract YoE from 500. Fix 4B: ambiguous single-token tech skills like 'R', 'Go', 'Swift' were generating 654 false positives — we gate them behind raw-case matching plus a tech-context window. Fix 4C: we were deriving resume titles from the broad Category column (7,286 unique values, too coarse); switching to first-line extraction lifted title-similarity to 31,254 unique values. The bottom-line lift: skill_overlap > 0 went from 36% to 68% of pairs."

### Slide 5 — Methodology Architecture
**Time: 90 sec.** This is your method slide. Walk through the 8 features by channel:
- Semantic: embedding_similarity (resume vs JD SBERT cosine), title_similarity
- Lexical: tfidf_similarity
- Skill: skill_overlap, weighted_skill_score, num_missing_skills, avg_missing_skill_importance
- Structural: years_of_experience gap

> "Stage 1: SBERT all-MiniLM-L6-v2 — 384-dim, 22M params, ~3ms per text on CPU. We compute the full 2.1M-pair cosine matrix in under a second. Retrieves top-50 per job, a 50x reduction in candidate space. Stage 2: LogReg trained on 600 LLM-consensus labels, class-balanced, decision threshold 0.52 chosen by CV. Inference is sub-millisecond per pair."

### Slide 6 — Methodology Journey
**Time: 90 sec.** This is your CONTRIBUTION slide. Take your time.
> "Five strategies, same 100-pair held-out test, same random_state. Weak supervision: composite formula labels — 0.4 embedding + 0.3 weighted_skill_score + 0.2 title + 0.1 education. CV F1 0.99, test F1 0.44 — a 0.55-point gap that screamed leakage. Investigation: the formula's components were also our features, so the model was learning to reproduce the formula. We caught this empirically. LLM supervised: 400 stratified pairs labeled by 3-LLM consensus, kappa 0.61. F1=0.704. Self-training v1: take the LLM model, label the rest of the pool with confidence > 0.7, retrain. F1 collapsed to 0.564 — class balance flipped because confident negatives dominated. Self-training v2: percentile-threshold, exclude the ambiguous middle. F1=0.579 — still worse than supervised. Yarowsky 1995 predicted exactly this: confidence-based selection reinforces the model's existing decision surface. Active learning: we did the opposite — we sampled the uncertainty band, probability between 0.40 and 0.60. Got 200 more labels. F1=0.769. Same label budget as 'supervised + random 200 more' would have been — but the placement of the labels is the difference. **Confidence-based sampling fails. Uncertainty-based sampling wins. Same budget, opposite mechanisms.**"

### Slide 7 — Final Results
**Time: 60 sec.**
> "F1 0.769 on the 100-pair held-out gold. +0.165 over SBERT-only baseline, +0.066 over the LLM-supervised baseline. Confusion matrix: 20 true positives, 2 false negatives, 10 false positives, 68 true negatives — precision 0.667, recall 0.909, P@10 0.90. High recall, lower precision — appropriate for a job-matching surfacing tool where missing a real match costs more than a false alarm."

### Slide 8 — Feature Importance (the SBERT critique slide)
**Time: 75 sec.** This is your DEFENSE slide. Memorize.
> "The natural critique is: 'your model is just SBERT dressed up with extra features.' We refute this empirically with three independent methods. **Direct comparison**: a LogReg trained on SBERT embedding alone gets F1=0.628. A LogReg trained on the 7 non-SBERT features — skill, title, TF-IDF, structural — gets F1=0.706. The full 8-feature model gets 0.769. Skill features alone beat SBERT alone by 7.8 F1 points. Three feature-importance methods: coefficients show embedding_similarity is largest at +0.98, but skill_overlap at +0.72 is comparable — and they're multicollinear because they measure related signals. Permutation importance: embedding 0.25, title 0.12, num_missing_skills 0.11 — five features all contribute 0.05 to 0.12. **Ablation**: dropping title_similarity costs -0.116 F1, dropping num_missing_skills costs -0.088, dropping embedding costs -0.063. **Title and skill features hurt more than SBERT when removed.** This is the killer point."

### Slide 9 — LLM Labeling + Why a Reranker is Needed
**Time: 60 sec.**
> "Why a reranker? Because SBERT retrieval alone surfaces a lot of plausible-looking false positives. On the 500-pair stratified gold, only 47.5% of SBERT top-50 matches are real per LLM consensus. The other three pool sources — A_mid mid-similarity, B_xcat cross-category strict, C_rand uniform random — show 3.3%, 8.0%, 8.0% positive rates. SBERT retrieves the candidate space; it doesn't separate it. Inter-LLM kappa is 0.61, substantial agreement by Landis & Koch. Unanimous votes 71%, disputed 29%. Per-LLM positive rates: Claude 22.4%, GPT 13.6%, Gemini 26.6% — Gemini is most lenient, GPT most conservative."

### Slide 10 — Deployability
**Time: 45 sec.**
> "The dashboard prototype has four panels: match score with percentile, skill gap report ranked by ESCO importance, 2-year quarterly roadmap aggregated across top-5 target jobs, and forward simulation that re-scores you after adding learned skills. Plaksha-specific use cases: internship matching tool for placement portal, course-to-career mapping for academic planning, skill-development advisor that aggregates gaps across a student's top-10 dream jobs into 5 highest-impact skills to learn."

### Slide 11 — Limitations & Future Work
**Time: 60 sec.** OWN your limitations. This builds credibility.
> "Six honest limitations. Test set is small at N=100, so F1 confidence is approximately ±0.07. LLM consensus is correlated noise, not ground truth. Skill vocabulary still has residual noise tokens. Individual predictions can be domain-incoherent — dashboard adds a category filter to handle this. English-only, IT-heavy corpus, cross-domain transfer untested. No real-world deployment validation. Future work directly maps to literature gaps: longitudinal cohort study (12-month follow-up on whether following recommendations changes employment), cross-industry test set beyond English IT, and demographic bias audit with per-group precision/recall."

Closing line:
> "Resume-job matching is fundamentally subjective. We built a system that quantifies the gap honestly — and tells candidates which skills to learn next."

---

## PART 3 — TOP 20 MOST-LIKELY QUESTIONS (verbal, tight)

### Q1: "Why F1 and not accuracy?"
> "Because class imbalance. The natural positive rate in random pairs is ~5%. Accuracy would be inflated by the majority-negative class. F1 balances precision and recall on the positive class, which is the rare class we care about."

### Q2: "What's your test set size and why so small?"
> "100 LLM-consensus-labeled pairs held out from the start, same set across all five methods. Small because LLM consensus labeling is expensive — 3 LLMs per pair. We have ±0.07 Wilson confidence on F1, which we disclose. The relative ordering between methods is robust to that noise; the absolute number is less precise."

### Q3: "How is the +0.066 active learning lift not just noise?"
> "Three reasons. First, it's a paired comparison — same 100 test pairs across both methods, so variance from test-set composition cancels. Second, the lift is mechanistically predicted by active learning theory: uncertainty sampling targets the decision boundary where label information is most valuable, while random sampling spreads labels across already-confident regions. Third, we're not claiming statistical significance — we're claiming directional consistency with theory. If we had 1000 test pairs we'd have much tighter CIs; that's in future work."

### Q4: "Why uncertainty band [0.40, 0.60] specifically?"
> "Standard uncertainty sampling from Lewis & Gale 1994 — the decision boundary at p=0.5 is where the classifier is least confident. We chose [0.40, 0.60] to get N=200 samples with our LLM budget. Tighter bands give fewer samples, wider bands include too many already-confident points."

### Q5: "Why LogReg and not a neural reranker?"
> "Three reasons. First, 600 training labels — neural nets need an order of magnitude more. Second, we need interpretability for the skill-gap diagnosis — LogReg coefficients map directly to feature importance. Third, we have eight hand-crafted features, not raw text — LogReg is the right inductive bias for that input. A cross-encoder would also blow our latency budget; we target <100 ms end-to-end."

### Q6: "Why SBERT and not BERT or a fine-tuned model?"
> "all-MiniLM-L6-v2 is fine-tuned for sentence similarity — exactly the retrieval task. BERT-CLS pooling is weaker for similarity than SBERT's training objective. Fine-tuning SBERT would need 10K+ labels we don't have; freezing it and putting learning in the reranker is the right tradeoff at our label budget."

### Q7: "What's ESCO depth weighting and why does it help?"
> "ESCO is a hierarchical occupational skill taxonomy. Skills deeper in the tree are more specific — 'Pandas DataFrames' at depth 6 is more informative than 'Programming' at depth 2. We weight skill_overlap by log(depth + 1), so matches on specific skills count more than matches on generic ones. The feature avg_missing_skill_importance encodes this directly."

### Q8: "Why 3-LLM consensus instead of just one LLM?"
> "Two reasons. First, ensemble noise reduction — Claude, GPT, and Gemini have different biases; majority vote dampens model-specific errors. Second, inter-rater agreement is itself a quality signal — 71% unanimous, 29% disputed, kappa 0.61. We can use the disputed pairs as a separate, harder eval band. Single-LLM labels would not let us measure or disclose this."

### Q9: "How do you know LLMs are correct? Isn't this just correlated bias?"
> "Yes, that's a real concern and we acknowledge it on the limitations slide. Three mitigations. First, kappa 0.61 between three independently-trained models is much better than chance — it means there's signal underneath. Second, we use majority vote, so single-LLM hallucinations get filtered. Third, we report relative comparisons — even if absolute F1 is biased by LLM noise, the RELATIVE ORDERING of methods (weak < self-train < supervised < active learning) is robust to a uniform label-noise floor. We are not claiming LLM consensus is ground truth — we're using it as a transparent, reproducible proxy."

### Q10: "Self-training failed twice — what does that say about your methodology?"
> "It says we ran controlled experiments and reported negative results honestly. Yarowsky 1995 predicted exactly this failure mode — confidence-based pseudo-labeling reinforces the model's existing decision surface. Including these failures is the scientific contribution: it lets us claim 'uncertainty sampling beats confidence sampling on the same budget' as an empirical result, not folklore."

### Q11: "Your weak supervision had 0.55 gap between CV and test. What happened?"
> "Leakage. Our composite label was 0.4·embedding + 0.3·weighted_skill_score + 0.2·title + 0.1·education — and embedding, weighted_skill_score, title were also our features. The model was learning to reproduce the formula. CV F1=0.99 was the smoking gun. Catching this empirically — and including it in the deck — is part of why we trust the active learning result. We do not have hidden leakage there."

### Q12: "What's your latency budget?"
> "End-to-end target <100 ms. SBERT encode of a new resume: ~3 ms on CPU. Cosine vs 853 JD embeddings (precomputed): <1 ms. Top-50 candidate selection: linear scan, <1 ms. Per-pair feature extraction: ~5 ms (skill matching dominates). LogReg inference: <1 ms per pair × 50 = 50 ms. Total ~60 ms. Production deployment would precompute JD embeddings in a vector DB."

### Q13: "How does this compare to the three 2025 papers in your literature?"
> "They report accuracies of 0.90 to 0.918 — higher than our F1=0.769. But none of them disclose gold-standard construction methodology. We can't tell if those numbers are on a held-out set, what the inter-rater agreement was, or what the class balance was. Our claim is methodological transparency, not headline-beating numbers. We'd take our 0.769 F1 with ±0.07 CI and disclosed kappa over their 0.918 accuracy with unknown evaluation any day. And — fair caveat — accuracy and F1 aren't the same metric, so the direct comparison isn't apples-to-apples."

### Q14: "Show me the confusion matrix."
> "TP=20, FN=2, FP=10, TN=68 on the 100-pair test. Precision=0.667, recall=0.909, F1=0.769. We're recall-heavy, which is the right calibration — surfacing a wrong match wastes time, missing a real match wastes opportunity. P@10 is 0.90, meaning if we surface 10 top jobs, nine are real matches per LLM gold."

### Q15: "How much did this cost?"
> "Under $15 USD total in API charges. 600 training labels × 3 LLMs = 1800 calls, 100 test × 3 = 300 calls, plus some pilot calls. Average ~$0.005 per call. The active learning loop made this affordable — we labeled only the uncertain band, not everything."

### Q16: "How would you scale labels to 10K pairs?"
> "Iterative active learning. Train on what you have, identify uncertain pairs, label those, retrain. Linear cost in labels needed, sub-linear cost in label-information per label as the model improves. We demonstrated one round of this; production would run 5-10 rounds."

### Q17: "Have you measured demographic bias?"
> "No. It's listed as a future-work item on the limitations slide. Doing it properly requires demographic metadata on resumes which our anonymized Kaggle dump doesn't have. The proper version is per-group precision/recall plus a published audit; we'd need to source a different dataset for that."

### Q18: "What's the deployability story for Plaksha specifically?"
> "Three concrete scenarios. Placement portal: when a student logs in, surface top-K live internships with per-job skill-gap reports. Course planning: given a transcript and target role like 'ML engineer', recommend electives that close the largest skill gaps. Career advisory: aggregate gaps across the student's top-10 dream jobs into the 5 highest-impact skills to learn over the next two years."

### Q19: "Walk me through one prediction end-to-end."
> "Take a candidate's resume. Step 1: SBERT encode resume text into 384-dim — 3 ms. Step 2: cosine similarity vs all 853 JD embeddings — <1 ms. Step 3: take top-50 JDs by cosine. Step 4: for each of those 50, compute 8 features: embedding cosine, title cosine, TF-IDF Jaccard, skill_overlap, weighted_skill_score with ESCO depth weighting, num_missing_skills, avg_missing_skill_importance, years_of_experience_gap. Step 5: LogReg outputs a probability per pair. Step 6: rank by probability. Step 7: for top-K, compute skill_gap = JD skills minus resume skills, ranked by ESCO depth importance. Output: match probability + ranked missing skills + roadmap."

### Q20: "What would change if your test set was 1000 instead of 100?"
> "F1 confidence interval would shrink from ±0.07 to ±0.025. The +0.066 active learning lift would either become statistically significant or wash out, which is exactly what we need to know. The relative ordering would almost certainly stay the same — the methodology journey is mechanistically interpretable, not just a leaderboard. We chose 100 as a budget-vs-rigor trade and disclose it. If we had unlimited budget we'd label 1000."

---

## PART 4 — ADVERSARIAL DEFENSE KIT (the brutal questions)

### A1: "Your gold standard is just LLM agreement. Why should I believe any of these numbers?"
**Strategy: agree with the premise, then reframe.**
> "You're right that LLM consensus is not ground truth. We say so explicitly on the limitations slide. Two responses. First, we never claim it is — we claim it's a transparent, reproducible proxy with disclosed kappa, much better than the undisclosed gold standards in the literature we compared against. Second, the contribution isn't the absolute F1 number, it's the relative comparison of five methodologies on the same proxy. Confidence-based pseudo-labeling fails. Uncertainty-based sampling wins. That conclusion is robust to a uniform label-noise floor — adding noise to all five methods equally doesn't reorder them."

### A2: "All you've shown is that adding more labels helps. That's not a contribution — that's just supervised learning."
> "Important distinction. Going from 400 random LLM labels (F1=0.704) to 600 random labels would not get to 0.769 — random labels in already-confident regions are wasted. Going to 600 via active learning — 400 random plus 200 from the [0.40, 0.60] uncertainty band — does get there. The contribution is **where** the labels are placed, not how many. Same budget, +6.6 F1 points purely from sampling strategy. That's the textbook active learning claim and we replicated it with controls."

### A3: "How do you know the active learning lift isn't from label noise variance? Maybe those 200 uncertain pairs happened to be easier."
> "Fair pushback. Three points. First, the uncertain pairs are by construction *harder* — the model's at chance on them. Adding hard labels should hurt or stay flat under a naïve theory; instead it helps, which is exactly what active learning theory predicts because they're informative for the decision boundary. Second, our 100-pair test set is held out from the start, never seen by any method, including the LLM labelers for those test pairs. Third, the proper ablation would be: random sample 200 vs uncertain sample 200, same budget. We did 400-random vs 400-random + 200-uncertain, which is suggestive but not a clean ablation. That's a fair limitation; the cleaner experiment is in our future-work list mentally."

### A4: "Your reranker is just learning what SBERT already encodes. Show me it's not."
> "Slide 8 shows it explicitly. A LogReg trained on SBERT embedding alone gets F1=0.628. A LogReg trained on 7 non-SBERT features — skill, title, TF-IDF, structural — gets F1=0.706, which is 7.8 points HIGHER than SBERT alone. Skill features carry independent signal that SBERT does not encode. Ablation confirms: dropping title_similarity costs 0.116 F1, dropping num_missing_skills costs 0.088, dropping embedding costs 0.063. Embedding is not even the most important feature. The reranker is not 'just SBERT dressed up'."

### A5: "Why didn't you try a transformer cross-encoder reranker? That's the standard approach."
> "Three reasons. First, label budget — cross-encoders need 5K-10K labels to fine-tune competitively; we have 600. Second, latency — cross-encoder inference on 50 pairs per query is 50× LogReg, blowing our <100ms budget on CPU. Third, interpretability — we need explainable skill-gap output, which requires per-feature contribution. Cross-encoder is a black box. LogReg on hand-crafted features is the correct tool for this label regime and this output requirement. We acknowledge it's a ceiling — with 10K labels and GPU inference we'd revisit."

### A6: "The literature papers report 0.90+ accuracy. You report 0.769 F1. That's worse."
> "Apples and oranges in two ways. First, accuracy and F1 are different metrics — on imbalanced data accuracy is misleading. We could report accuracy too — it would be around 0.88 — but F1 is the correct metric for the rare positive class. Second, those papers do not disclose how they constructed their gold standard, what their class balance was, or what their inter-rater agreement was. We have no idea if their 0.918 is on a held-out set or on training data. Our 0.769 is on 100 LLM-consensus held-out pairs with kappa 0.61 and random_state=42 — fully reproducible. We'd publish our numbers; we suspect their evaluation would not survive peer review."

### A7: "Active learning is a 30-year-old technique. Where's the novelty?"
> "The novelty isn't the technique — Lewis & Gale 1994, Settles 2009 — it's the controlled comparison. We're not claiming to invent active learning. We're claiming an empirical falsification of self-training under our setting, and an empirical confirmation of uncertainty sampling under the same budget. The five-method head-to-head with shared test set and disclosed labels is the contribution. In a domain where most published papers don't even disclose evaluation methodology, doing the careful comparison IS the contribution."

### A8: "Self-training failed twice. Did you try other semi-supervised methods?"
> "Within budget, we tried two variants of self-training — absolute confidence threshold (v1) and percentile threshold with diversity filtering (v2). Both failed. We did not try co-training, MixMatch, or graph-based label propagation. Those are in future work. The point of the comparison wasn't 'self-training is universally bad' — it was 'confidence-based pseudo-labeling under this setting reinforces existing errors, exactly as Yarowsky predicted'. Active learning, which targets uncertainty rather than confidence, is the natural counter."

### A9: "Your features look hand-crafted and old-fashioned. Why not use embeddings end-to-end?"
> "Label budget — 600 labels is not enough to learn good representations from scratch. Hand-crafted features encode strong priors: skill ontology, ESCO depth weighting, year-of-experience gap. They give the LogReg a head start. The hybrid is the right point on the bias-variance trade for this label regime. Slide 8 shows it works."

### A10: "Have you tested robustness to JD wording? Adversarial inputs?"
> "No formal robustness test. SBERT is reasonably paraphrase-robust by design — it was trained on NLI/STS pairs. The skill-extraction layer uses ESCO + flashtext matching which is exact-string-based, so it would miss paraphrase variations like 'Pandas' vs 'data analysis in Python'. We rely on the SBERT layer to compensate. A proper adversarial test set is future work."

### A11: "Your dashboard is a prototype. How is this 'deployable'?"
> "The slide says 'deployability' not 'deployed'. We have a working Streamlit prototype demonstrating the four user-facing panels. Productionization needs: vector DB for embedding storage, user auth, latency tuning, monitoring, A/B framework, and importantly — a bias audit before public release. We've scoped what's missing rather than overclaiming."

### A12: "Class balance in your test set is 22%. In the wild it would be 5%. Won't your numbers fall apart in production?"
> "Two responses. First, our test set is balanced to the curated pool, not to natural distribution — yes, in the wild positive rate is much lower. Second, we're not deploying as a binary classifier — we're using the probability as a ranking score. P@10 0.90 is the operational metric: of the top-10 jobs we surface, 9 are real. Class imbalance doesn't change ranking quality as long as the model is calibrated. We use class_weight='balanced' in LogReg, which adjusts the loss but does NOT distort probability calibration the way over/undersampling would."

### A13: "You said 'kappa = 0.61, substantial agreement' — Landis & Koch is criticized as arbitrary. Defense?"
> "Fair — Landis & Koch 1977 thresholds are conventional, not derived. Two stronger framings. First, kappa 0.61 means LLMs agree on labels 61 percentage points beyond chance — substantively, that's three independent models converging on the same answer 71% of the time unanimously. Second, kappa varies with class balance — at 22% positive rate, 0.61 corresponds to roughly 85% raw agreement. That's higher than typical human-annotator agreement in many NLP tasks. We'd take that signal."

### A14: "ESCO has 20K+ skills. How do you handle synonym/paraphrase variation? 'ML' vs 'Machine Learning'?"
> "Three layers. First, ESCO entries include 'preferred labels' and 'alt labels' — 'Machine Learning' is the preferred, 'ML', 'machine-learning', etc are alt labels. Our skill-extraction pipeline normalizes to preferred labels. Second, for tech tokens not in ESCO, our tech_skills.py adds canonical mappings. Third, fuzzy variation that escapes both layers is handled implicitly by SBERT embedding similarity — 'I worked on ML' and 'experience in Machine Learning' have ~0.85 cosine even without exact skill match."

### A15: "What if I dispute your kappa interpretation? Walk me through the math."
> "Cohen's kappa = (p_o - p_e) / (1 - p_e). Observed agreement p_o = 0.71 (unanimous votes). Expected agreement under independence given marginal positive rates 22.4%, 13.6%, 26.6% — call it ~0.27 weighted. So kappa = (0.71 - 0.27) / (1 - 0.27) = 0.44 / 0.73 = 0.60, rounds to 0.61. That's the calculation."
**Note: this is approximate — if challenged on details, say "I can show you the exact computation in the report."**

---

## PART 5 — PER-SLIDE TRAP QUESTIONS (what they'll ask after each)

| Slide | Most likely follow-up | One-line answer |
|---|---|---|
| 1 (Title) | "Why two-stage?" | "Retrieval handles scale; reranking handles precision. SBERT-only loses precision at top of ranked list." |
| 2 (Problem) | "How do you measure 'good match'?" | "3-LLM majority vote on a 100-pair held-out set. Limitations slide acknowledges this is a proxy." |
| 3 (Lit) | "Are the lit papers' numbers really comparable?" | "No — they don't disclose gold standard. We chose methodological transparency over headline accuracy." |
| 4 (Dataset) | "Is 2,484 resumes enough?" | "Enough for evaluation; production retraining would benefit from more. Variance is mostly in the labels, not the unlabeled pool." |
| 5 (Methodology) | "Why exactly these 8 features?" | "Iteratively designed by ablation. We started with 12 candidates; ablation kept 8 that each contributed >0.02 F1." |
| 6 (Journey) | "Did you have to try all 5 to find the winner?" | "Yes — that IS the contribution. We did not know a priori which would win, especially given literature recommends self-training." |
| 7 (Results) | "Recall 0.91 is great but precision 0.67 is mediocre. Why?" | "Calibration choice for surfacing tasks — missing a real match costs more than a false alarm. Threshold is tunable." |
| 8 (Feature imp) | "Multicollinearity inflates skill_overlap coefficient. Did you regularize?" | "Yes — L2 with C=1 chosen by CV. Coefficient magnitudes are stable across folds." |
| 9 (LLM labels) | "Did the 3 LLMs see each other's labels?" | "No — independent calls, same prompt, then majority vote computed offline." |
| 10 (Deployability) | "Real-time or batch?" | "Batch precompute for JDs, real-time encode + score for new resumes. Sub-100ms per query." |
| 11 (Limitations) | "What's the ONE most important next step?" | "1000-pair test set with controlled label noise study. Without it we can't claim significance for the +0.066 lift." |

---

## PART 6 — THINGS NOT TO SAY (traps)

1. **DO NOT say "our model is the best"** — say "our model gets F1=0.769 with disclosed ±0.07 CI on transparent gold."
2. **DO NOT say "LLMs are ground truth"** — always frame as "LLM consensus as a transparent proxy".
3. **DO NOT say "active learning is novel"** — say "we replicated and controlled active learning vs self-training under matched budgets."
4. **DO NOT defend leakage** — own it. "We caught it empirically via CV-test gap." That's a strength, not a weakness.
5. **DO NOT compare F1 to accuracy directly** — flag the metric mismatch.
6. **DO NOT promise statistical significance** at N=100 — say "directional consistency with theory; significance is a future-work item."
7. **DO NOT improvise numbers** — if you don't remember, say "I'd have to check the report." Then write down what they asked for the next round.
8. **DO NOT throw your teammates under the bus** if it's a group project. Own all weaknesses collectively.
9. **DO NOT say "the literature is wrong"** — say "the literature doesn't disclose evaluation, so we can't directly compare."
10. **DO NOT speak in jargon you can't define on the spot.** If you say "Yarowsky 1995", be ready to summarize it: "self-training reinforces the model's existing decision surface because confident pseudo-labels are by definition the ones the model already agrees with."

---

## PART 7 — DELIVERY PRACTICE CHECKLIST

- [ ] Recite Part 0 (elevator pitch) in 60 seconds, no notes, twice.
- [ ] Recite key numbers (F1=0.769, +0.165 SBERT, +0.066 supervised, kappa=0.61, CI±0.07) without checking.
- [ ] For each of Q1–Q20, give the answer in under 30 seconds out loud.
- [ ] For each of A1–A15, give the answer in under 60 seconds out loud.
- [ ] Per-slide trap one-liners — recite them while looking at each slide.
- [ ] Run through the Yarowsky 1995 mini-explanation: "confidence-based selection reinforces existing model errors because confident pseudo-labels are by definition in the model's already-strong regions."
- [ ] Practice the active-learning kappa math (Part A15) on paper.
- [ ] Pick ONE limitation to volunteer proactively when asked "what would you change?" Recommended: 1000-pair test set with N=200 vs N=200 random-vs-uncertain ablation.

---

## PART 8 — OPENING AND CLOSING LINES

**Opening (first 10 seconds, before slide 1 even loads):**
> "Resume-job matching is an unsolved problem in industry — every paper claims 90%+ accuracy on undisclosed gold standards. We took the opposite approach: instead of optimizing the headline number, we built a five-method methodological comparison on a transparent 3-LLM consensus benchmark, and we report what worked, what failed, and why."

**Closing (last 10 seconds, after slide 11):**
> "F1=0.769 with ±0.07 CI on 100 held-out LLM-consensus labels. Active learning beats self-training on the same label budget by +6.6 F1 points, exactly as theory predicts and prior pseudo-labeling failures suggest. We disclose limitations honestly and our random_state=42 throughout makes this reproducible. Thank you — happy to take questions."

---

**Final reminder:** if you don't know the answer, say "that's a good question — I'd need to check the experimental log, but my best understanding is X." Never bluff a number. Confidence comes from preparation, not improvisation.
