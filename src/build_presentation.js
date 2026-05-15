// build_presentation.js
// =====================
// Generates the endterm PPTX deck matching the reference design language
// (EmpowerMe-style: light slate background, electric-blue accent, pill labels,
// dark navy headlines with selective blue emphasis, card-based layouts).
//
// Output: outputs/presentation/SkillMatch_Final_Presentation.pptx
//
// Run:
//   node src/build_presentation.js

const path = require("path");
const pptxgen = require("pptxgenjs");

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------
const ROOT = path.resolve(__dirname, "..");
const CHARTS = path.join(ROOT, "outputs", "presentation");
const OUT = path.join(CHARTS, "SkillMatch_Final_Presentation.pptx");

// ---------------------------------------------------------------------------
// Design tokens (extracted from reference deck)
// ---------------------------------------------------------------------------
const BG          = "F8FAFC";   // background, light slate-50
const NAVY        = "0F172A";   // headings, slate-900
const BODY        = "475569";   // body text, slate-600
const MUTED       = "94A3B8";   // captions, slate-400
const BLUE        = "2563EB";   // primary accent, blue-600
const BLUE_LIGHT  = "DBEAFE";   // pill bg, blue-100
const BLUE_BORDER = "BFDBFE";   // pill border, blue-200
const CARD_BG     = "FFFFFF";   // card surface
const CARD_BORDER = "E2E8F0";   // card outline, slate-200
const GREEN       = "10B981";   // success / winner emerald
const RED         = "EF4444";   // failure red

// Deck is LAYOUT_WIDE: 13.333" x 7.5"
const SW = 13.333;
const SH = 7.5;

const FONT_HEAD = "Arial Black";
const FONT_BODY = "Arial";

// ---------------------------------------------------------------------------
// Initialize
// ---------------------------------------------------------------------------
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title  = "SkillMatch — Two-Stage Skill-Aware Reranker";
pres.author = "MLPR Project Team";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Setup a slide with the brand background and small brand mark in header
function newSlide(opts = {}) {
  const s = pres.addSlide();
  s.background = { color: BG };
  return s;
}

// Pill-shaped section label (uppercase, blue, letter-spaced, light blue bg + border)
function addPill(slide, text, x, y, w = 3.5) {
  const h = 0.42;
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h,
    fill: { color: BLUE_LIGHT },
    line: { color: BLUE_BORDER, width: 1 },
    rectRadius: h / 2,
  });
  slide.addText(text, {
    x, y, w, h,
    fontFace: FONT_BODY, fontSize: 11, color: BLUE,
    bold: true, charSpacing: 4,
    align: "center", valign: "middle",
    margin: 0,
  });
}

// Mini brand mark (top of slide)
function addBrandHeader(slide) {
  // small pill at top center: "SkillMatch"
  const w = 2.0, h = 0.45, x = (SW - w) / 2, y = 0.18;
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h,
    fill: { color: BLUE_LIGHT },
    line: { color: BLUE_BORDER, width: 1 },
    rectRadius: 0.1,
  });
  slide.addText([
    { text: "S", options: { color: BLUE, bold: true } },
    { text: "killMatch", options: { color: NAVY, bold: true } },
  ], {
    x, y, w, h,
    fontFace: FONT_HEAD, fontSize: 18,
    align: "center", valign: "middle", margin: 0,
  });
}

// Section header with leading blue dot
function addSectionHeader(slide, text, x, y, w = 7) {
  // blue dot
  slide.addShape(pres.shapes.OVAL, {
    x, y: y + 0.22, w: 0.18, h: 0.18,
    fill: { color: BLUE }, line: { color: BLUE },
  });
  slide.addText(text, {
    x: x + 0.32, y, w: w - 0.32, h: 0.6,
    fontFace: FONT_HEAD, fontSize: 24, bold: true,
    color: NAVY, align: "left", valign: "middle", margin: 0,
  });
}

// Card with a small uppercase blue tag, bold title, body text
function addCard(slide, x, y, w, h, opts) {
  // outer card
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h,
    fill: { color: CARD_BG },
    line: { color: CARD_BORDER, width: 1 },
    rectRadius: 0.08,
  });
  // tag (uppercase blue)
  if (opts.tag) {
    slide.addText(opts.tag, {
      x: x + 0.25, y: y + 0.18, w: w - 0.5, h: 0.3,
      fontFace: FONT_BODY, fontSize: 10, bold: true,
      color: BLUE, charSpacing: 3,
      align: "left", valign: "top", margin: 0,
    });
  }
  // title
  if (opts.title) {
    slide.addText(opts.title, {
      x: x + 0.25, y: y + (opts.tag ? 0.5 : 0.2),
      w: w - 0.5, h: opts.titleH || 0.45,
      fontFace: FONT_HEAD, fontSize: opts.titleSize || 16, bold: true,
      color: NAVY, align: "left", valign: "top", margin: 0,
    });
  }
  // body — bodyY computed so it sits just below title (uses titleH if small)
  if (opts.body) {
    const titleH = opts.titleH || 0.45;
    const titleY = y + (opts.tag ? 0.5 : 0.2);
    const bodyY  = titleY + titleH + 0.05;
    slide.addText(opts.body, {
      x: x + 0.25, y: bodyY,
      w: w - 0.5, h: h - (bodyY - y) - 0.12,
      fontFace: FONT_BODY, fontSize: opts.bodySize || 12,
      color: BODY, align: "left", valign: "top", margin: 0,
    });
  }
}

// Footer (brand - title - date - URL)
function addFooter(slide, page) {
  slide.addText([
    { text: "SkillMatch", options: { bold: true, color: NAVY } },
    { text: "    -    ",  options: { color: MUTED } },
    { text: "Endterm Project Evaluation",  options: { color: BODY } },
    { text: "    -    ",  options: { color: MUTED } },
    { text: "MLPR 2026",   options: { color: BODY } },
    { text: "    -    ",  options: { color: MUTED } },
    { text: `${page} / 11`, options: { color: MUTED } },
  ], {
    x: 0.5, y: SH - 0.4, w: SW - 1, h: 0.3,
    fontFace: FONT_BODY, fontSize: 9, align: "center", margin: 0,
  });
}

// Italic emphasized closing line, like reference deck
function addItalicTagline(slide, text, x, y, w, opts = {}) {
  slide.addText(text, {
    x, y, w, h: 0.4,
    fontFace: FONT_BODY, fontSize: opts.fontSize || 14,
    italic: true, color: opts.color || BODY,
    bold: !!opts.bold,
    align: "center", valign: "middle", margin: 0,
  });
}

// Connector line between two points (straight)
function addLine(slide, x1, y1, x2, y2, color = NAVY, width = 1.2) {
  const w = Math.abs(x2 - x1);
  const h = Math.abs(y2 - y1);
  slide.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w, h,
    line: { color, width },
    flipH: x2 < x1, flipV: y2 < y1,
  });
}

// Small "tag" pill (used for source labels on lit-review etc.)
function addSmallPill(slide, text, x, y, w = 1.4) {
  const h = 0.34;
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h,
    fill: { color: CARD_BG },
    line: { color: CARD_BORDER, width: 1 },
    rectRadius: h / 2,
  });
  slide.addText(text, {
    x, y, w, h,
    fontFace: FONT_BODY, fontSize: 10, color: NAVY, bold: true,
    charSpacing: 2, align: "center", valign: "middle", margin: 0,
  });
}

// ===========================================================================
// SLIDE 1 — Title / Hero
// ===========================================================================
{
  const s = newSlide();
  addBrandHeader(s);

  // Section pill
  addPill(s, "TWO-STAGE SKILL-AWARE MATCHING SYSTEM", (SW - 5.2) / 2, 0.95, 5.2);

  // Hero headline (two lines with blue emphasis word at end)
  s.addText([
    { text: "Match candidates to jobs", options: { color: NAVY, breakLine: true } },
    { text: "they're actually ", options: { color: NAVY } },
    { text: "qualified", options: { color: BLUE } },
    { text: " for.", options: { color: NAVY } },
  ], {
    x: 0.6, y: 1.7, w: SW - 1.2, h: 1.8,
    fontFace: FONT_HEAD, fontSize: 48, bold: true,
    align: "center", valign: "top", margin: 0,
  });

  // Subtitle
  s.addText(
    "A two-stage retrieval-and-reranking system that scores resume–job pairs using semantic, lexical, and ESCO-grounded skill features.",
    {
      x: 1.2, y: 3.5, w: SW - 2.4, h: 0.5,
      fontFace: FONT_BODY, fontSize: 15, color: BODY,
      align: "center", valign: "middle", margin: 0,
    },
  );

  // "Built for" line
  s.addText([
    { text: "Built for ", options: { color: BODY } },
    { text: "job seekers", options: { color: BLUE, bold: true } },
    { text: ", ", options: { color: BODY } },
    { text: "career counselors", options: { color: BLUE, bold: true } },
    { text: ", and ", options: { color: BODY } },
    { text: "placement teams", options: { color: BLUE, bold: true } },
    { text: " — anywhere a quantitative skill-gap diagnosis matters.", options: { color: BODY } },
  ], {
    x: 1.2, y: 4.05, w: SW - 2.4, h: 0.45,
    fontFace: FONT_BODY, fontSize: 13,
    align: "center", valign: "middle", margin: 0,
  });

  // Architecture diagram: 3 cards with connecting top bar
  const cy = 5.0;
  const ch = 1.2;
  const cw = 3.6;
  const gap = 0.35;
  const totalW = 3 * cw + 2 * gap;
  const cx0 = (SW - totalW) / 2;

  // Top header bar (blue) — "SkillMatch Pipeline" SPANNING ALL 3 CARDS
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: cx0, y: cy - 0.5, w: totalW, h: 0.42,
    fill: { color: BLUE }, line: { color: BLUE },
    rectRadius: 0.1,
  });
  s.addText("SkillMatch Pipeline", {
    x: cx0, y: cy - 0.5, w: totalW, h: 0.42,
    fontFace: FONT_HEAD, fontSize: 16, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0,
  });

  // (tick marks removed — header bar already visually connects to cards)

  // 3 architecture cards
  const cards = [
    { tag: "STAGE 1 — RETRIEVAL", title: "SBERT", body: "Retrieves top-50 candidate pairs per job from 2.1M possible pairs (all-MiniLM-L6-v2)." },
    { tag: "STAGE 2 — RERANKING", title: "LogReg + 8 features", body: "Semantic, lexical, skill-coverage, ESCO-weighted importance, experience, title similarity." },
    { tag: "OUTPUT", title: "Skill Gap Diagnosis", body: "Match probability + ranked missing skills + 2-year personalised roadmap." },
  ];
  cards.forEach((c, i) => {
    addCard(s, cx0 + i * (cw + gap), cy, cw, ch, {
      tag: c.tag, title: c.title, body: c.body,
      titleSize: 18, bodySize: 11,
    });
  });

  addFooter(s, 1);
}

// ===========================================================================
// SLIDE 2 — Problem Statement & Applications
// ===========================================================================
{
  const s = newSlide();
  addBrandHeader(s);

  addSectionHeader(s, "Problem Statement & Applications", 0.6, 0.95);

  // Two-column layout
  // Left: problem statement
  s.addText("The problem", {
    x: 0.6, y: 1.75, w: 6.0, h: 0.45,
    fontFace: FONT_HEAD, fontSize: 18, bold: true, color: NAVY, margin: 0,
  });
  s.addText([
    { text: "Given a resume and a job description, predict whether the candidate is a good match.", options: { color: BODY, breakLine: true, breakAfter: true } },
    { text: "\n", options: { breakLine: true } },
    { text: "Recruiters screen 250+ resumes per role on average and reject 75% on primitive keyword matching. Job seekers see hundreds of postings with no systematic way to know which they're qualified for or what skills to develop.", options: { color: BODY, breakLine: true } },
    { text: "\n", options: { breakLine: true } },
    { text: "Same model, two users: identify good matches at the recruiter end; quantify skill gaps at the candidate end.", options: { color: NAVY, italic: true } },
  ], {
    x: 0.6, y: 2.2, w: 6.0, h: 3.0,
    fontFace: FONT_BODY, fontSize: 13, valign: "top", margin: 0,
    paraSpaceAfter: 10,
  });

  // Right: 3 application cards stacked
  const rx = 7.0, rw = 5.7;
  s.addText("Applications", {
    x: rx, y: 1.75, w: rw, h: 0.45,
    fontFace: FONT_HEAD, fontSize: 18, bold: true, color: NAVY, margin: 0,
  });

  const apps = [
    { tag: "FOR JOB SEEKERS", title: "Personalised job-fit scoring", body: "Rank live postings by match probability. Surface skill gaps with importance weights." },
    { tag: "FOR PLAKSHA PLACEMENT",  title: "Internship & course mapping", body: "Match students to internships; recommend electives by aligning transcripts with target-role skill graphs." },
    { tag: "FOR CAREER COUNSELING",  title: "2-year skill roadmap", body: "Aggregate gaps across target roles → quarterly learning plan with forward-simulation of match probability lift." },
  ];
  let ay = 2.15;
  const ah = 1.18;
  apps.forEach((a) => {
    addCard(s, rx, ay, rw, ah, {
      tag: a.tag, title: a.title, body: a.body,
      titleSize: 15, bodySize: 11, titleH: 0.32,
    });
    ay += ah + 0.22;
  });

  // Italic emphasis at bottom
  addItalicTagline(
    s,
    "We don't just produce a score — we tell candidates which skills to learn next.",
    0.6, SH - 0.95, SW - 1.2,
    { bold: true, color: BLUE, fontSize: 14 },
  );

  addFooter(s, 2);
}

// ===========================================================================
// SLIDE 3 — Literature Survey
// ===========================================================================
{
  const s = newSlide();
  addBrandHeader(s);

  addSectionHeader(s, "Literature Survey — three 2025 papers", 0.6, 0.95, 12);

  // Three paper cards (top row)
  const papers = [
    {
      tag: "P1 — DASH et al., IJEDR 2025",
      title: "AI Skill Gap Analyzer",
      body: "Full-stack web app. BERT/RoBERTa NER + ESCO/O*NET ontologies. Reports 0.90 accuracy + 92% user satisfaction. Has course-recommendation engine.",
    },
    {
      tag: "P2 — DABERAO et al., GITCON 2025",
      title: "ResumeInsight",
      body: "Indian campus recruitment. spaCy + Levenshtein, RF/XGB/ANN classifier. K-Means cluster labels (a methodological weakness flagged in our review).",
    },
    {
      tag: "P3 — SRIBHARATHI et al., ICSSS 2025",
      title: "Scopira AI Career Platform",
      body: "TF-IDF + BERT, 4-module system with career-path generator. 0.918 accuracy claimed; 200+ user pilot. Outperforms LinkedIn Skills Match.",
    },
  ];
  const px = 0.6;
  const pw = (SW - 1.2 - 0.4) / 3;
  papers.forEach((p, i) => {
    addCard(s, px + i * (pw + 0.2), 1.75, pw, 1.85, {
      tag: p.tag, title: p.title, body: p.body,
      titleSize: 15, bodySize: 11, titleH: 0.42,
    });
  });

  // What unites them + what divides them (two side-by-side blocks)
  const by = 3.85, bh = 2.2;
  const bw = (SW - 1.2 - 0.3) / 2;

  // Common pipeline card (light)
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.6, y: by, w: bw, h: bh,
    fill: { color: BLUE_LIGHT }, line: { color: BLUE_BORDER, width: 1 },
    rectRadius: 0.08,
  });
  s.addText("WHAT UNITES THEM", {
    x: 0.85, y: by + 0.2, w: bw - 0.5, h: 0.3,
    fontFace: FONT_BODY, fontSize: 11, bold: true, color: BLUE,
    charSpacing: 3, margin: 0,
  });
  s.addText([
    { text: "Same conceptual pipeline: ", options: { bold: true, color: NAVY } },
    { text: "extract skills → semantic match → surface gap → recommend learning.", options: { color: BODY } },
    { text: "\n\n", options: { breakLine: true } },
    { text: "All built for English IT/data-science roles. All evaluate on internal gold standards that are never disclosed.", options: { color: BODY } },
  ], {
    x: 0.85, y: by + 0.55, w: bw - 0.5, h: bh - 0.7,
    fontFace: FONT_BODY, fontSize: 12, valign: "top", margin: 0,
  });

  // Where we differ — matched border weight for symmetry
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.6 + bw + 0.3, y: by, w: bw, h: bh,
    fill: { color: CARD_BG }, line: { color: GREEN, width: 1 },
    rectRadius: 0.08,
  });
  s.addText("WHERE OUR WORK ADVANCES THE FIELD", {
    x: 0.6 + bw + 0.55, y: by + 0.2, w: bw - 0.5, h: 0.3,
    fontFace: FONT_BODY, fontSize: 11, bold: true, color: GREEN,
    charSpacing: 3, margin: 0,
  });
  s.addText([
    { text: "•  3-LLM consensus labels (Claude+GPT+Gemini, κ=0.61)", options: { color: NAVY, breakLine: true } },
    { text: "•  Empirical leakage detection (CV 0.99 → test 0.44)", options: { color: NAVY, breakLine: true } },
    { text: "•  Head-to-head: weak / supervised / self-training / active learning", options: { color: NAVY, breakLine: true } },
    { text: "•  Inter-rater agreement disclosed; reproducible", options: { color: NAVY } },
  ], {
    x: 0.85 + bw + 0.3, y: by + 0.55, w: bw - 0.5, h: bh - 0.7,
    fontFace: FONT_BODY, fontSize: 12, valign: "top", margin: 0,
    paraSpaceAfter: 6,
  });

  addFooter(s, 3);
}

// ===========================================================================
// SLIDE 4 — Dataset & Preprocessing
// ===========================================================================
{
  const s = newSlide();
  addBrandHeader(s);

  addSectionHeader(s, "Dataset & Preprocessing", 0.6, 0.95);

  // 4 large stat callouts in a row
  const stats = [
    { val: "2,484", label: "Resumes\n(Kaggle, anonymized)" },
    { val: "853",   label: "Job descriptions\n(public corpus)" },
    { val: "20,253", label: "Skills in vocabulary\n(ESCO + O*NET + tech)" },
    { val: "50,650", label: "Candidate pairs\n(diversified pool)" },
  ];
  const sx = 0.6, sw = (SW - 1.2 - 3 * 0.25) / 4, sy = 1.75, sh = 1.4;
  stats.forEach((st, i) => {
    const x = sx + i * (sw + 0.25);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: sy, w: sw, h: sh,
      fill: { color: CARD_BG }, line: { color: CARD_BORDER, width: 1 },
      rectRadius: 0.08,
    });
    s.addText(st.val, {
      x, y: sy + 0.18, w: sw, h: 0.7,
      fontFace: FONT_HEAD, fontSize: 36, bold: true, color: BLUE,
      align: "center", valign: "middle", margin: 0,
    });
    s.addText(st.label, {
      x: x + 0.15, y: sy + 0.85, w: sw - 0.3, h: 0.5,
      fontFace: FONT_BODY, fontSize: 11, color: BODY,
      align: "center", valign: "top", margin: 0,
    });
  });

  // Three preprocessing fixes
  const fy = 3.45;
  s.addText("Three preprocessing fixes that materially improved data quality", {
    x: 0.6, y: fy, w: SW - 1.2, h: 0.4,
    fontFace: FONT_HEAD, fontSize: 16, bold: true, color: NAVY, margin: 0,
  });

  const fixes = [
    { tag: "FIX 4A — JD YEARS-OF-EXPERIENCE", title: "Parse from model_response JSON", body: "Original regex caught 0/853. After fix: 500/853 jobs have non-zero YoE." },
    { tag: "FIX 4B — AMBIGUOUS TECH TOKENS",  title: "Raw-case + context check", body: "Tokens like R, Go, Swift gated behind capitalization OR tech-context window. 654 false positives removed." },
    { tag: "FIX 4C — RESUME TITLES",         title: "First-line instead of Category", body: "Was using broad Category column (7,286 unique). First-line extraction lifted unique title-similarity values to 31,254." },
  ];
  const fx = 0.6, fw = (SW - 1.2 - 0.4) / 3, fxy = 3.95, fxh = 1.85;
  fixes.forEach((f, i) => {
    addCard(s, fx + i * (fw + 0.2), fxy, fw, fxh, {
      tag: f.tag, title: f.title, body: f.body,
      titleSize: 14, bodySize: 11, titleH: 0.42,
    });
  });

  // ESCO callout at bottom
  addItalicTagline(
    s,
    "ESCO vocabulary expansion lifted skill_overlap > 0 from 36% to 68% of pairs.",
    0.6, SH - 0.95, SW - 1.2,
    { bold: true, color: BLUE, fontSize: 13 },
  );

  addFooter(s, 4);
}

// ===========================================================================
// SLIDE 5 — Methodology Architecture (the two-stage)
// ===========================================================================
{
  const s = newSlide();
  addBrandHeader(s);

  addSectionHeader(s, "Methodology — Two-Stage Architecture", 0.6, 0.95, 12);

  // Stage 1 + Stage 2 boxes with arrow between
  const sy = 1.75;
  const sh = 2.35;
  const w1 = 5.0;
  const w2 = 5.5;
  const ax = 1.0; const bx = ax + w1 + 0.7;

  // Stage 1
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: ax, y: sy, w: w1, h: sh,
    fill: { color: CARD_BG }, line: { color: BLUE_BORDER, width: 1.5 },
    rectRadius: 0.1,
  });
  s.addText("STAGE 1 — RETRIEVAL", {
    x: ax + 0.3, y: sy + 0.2, w: w1 - 0.6, h: 0.3,
    fontFace: FONT_BODY, fontSize: 11, bold: true, color: BLUE, charSpacing: 3, margin: 0,
  });
  s.addText("SBERT semantic similarity", {
    x: ax + 0.3, y: sy + 0.55, w: w1 - 0.6, h: 0.5,
    fontFace: FONT_HEAD, fontSize: 22, bold: true, color: NAVY, margin: 0,
  });
  s.addText([
    { text: "• all-MiniLM-L6-v2 (384-dim, ~22M params)", options: { color: BODY, breakLine: true } },
    { text: "• 2.12M possible pairs → top 50 per job", options: { color: BODY, breakLine: true } },
    { text: "• ~3ms per text on CPU; full matrix <1s", options: { color: BODY, breakLine: true } },
    { text: "• 50× reduction in candidate space", options: { color: BODY } },
  ], {
    x: ax + 0.3, y: sy + 1.15, w: w1 - 0.6, h: sh - 1.3,
    fontFace: FONT_BODY, fontSize: 13, valign: "top", margin: 0, paraSpaceAfter: 4,
  });

  // Arrow
  s.addShape(pres.shapes.RIGHT_TRIANGLE, {
    x: ax + w1 + 0.1, y: sy + sh / 2 - 0.18, w: 0.5, h: 0.36,
    fill: { color: BLUE }, line: { color: BLUE }, rotate: 0,
  });

  // Stage 2
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: bx, y: sy, w: w2, h: sh,
    fill: { color: CARD_BG }, line: { color: GREEN, width: 1.8 },
    rectRadius: 0.1,
  });
  s.addText("STAGE 2 — RERANKING", {
    x: bx + 0.3, y: sy + 0.2, w: w2 - 0.6, h: 0.3,
    fontFace: FONT_BODY, fontSize: 11, bold: true, color: GREEN, charSpacing: 3, margin: 0,
  });
  s.addText("Logistic Regression on 8 features", {
    x: bx + 0.3, y: sy + 0.55, w: w2 - 0.6, h: 0.5,
    fontFace: FONT_HEAD, fontSize: 22, bold: true, color: NAVY, margin: 0,
  });
  s.addText([
    { text: "• Trained on 600 LLM-consensus labels", options: { color: BODY, breakLine: true } },
    { text: "• Class-balanced, CV threshold = 0.52", options: { color: BODY, breakLine: true } },
    { text: "• Inference <1ms per pair", options: { color: BODY, breakLine: true } },
    { text: "• Outputs match probability + skill gap", options: { color: BODY } },
  ], {
    x: bx + 0.3, y: sy + 1.15, w: w2 - 0.6, h: sh - 1.3,
    fontFace: FONT_BODY, fontSize: 13, valign: "top", margin: 0, paraSpaceAfter: 4,
  });

  // The 8 features (4 columns of 2)
  const fy = 4.35;
  s.addText("The 8 features (4 channels):", {
    x: 0.6, y: fy, w: SW - 1.2, h: 0.32,
    fontFace: FONT_HEAD, fontSize: 14, bold: true, color: NAVY, margin: 0,
  });

  const featureGroups = [
    { label: "SEMANTIC", items: ["embedding_similarity", "title_similarity"] },
    { label: "LEXICAL",   items: ["tfidf_similarity"] },
    { label: "SKILL",     items: ["skill_overlap", "weighted_skill_score", "num_missing_skills", "avg_missing_skill_importance"] },
    { label: "STRUCTURAL", items: ["years_of_experience"] },
  ];
  const gx = 0.6, gw = (SW - 1.2 - 0.45) / 4, gy = 4.78, gh = 1.65;
  featureGroups.forEach((g, i) => {
    const x = gx + i * (gw + 0.15);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: gy, w: gw, h: gh,
      fill: { color: BLUE_LIGHT }, line: { color: BLUE_BORDER, width: 1 },
      rectRadius: 0.08,
    });
    s.addText(g.label, {
      x: x + 0.15, y: gy + 0.15, w: gw - 0.3, h: 0.3,
      fontFace: FONT_BODY, fontSize: 10, bold: true, color: BLUE, charSpacing: 3, margin: 0,
    });
    const items = g.items.map((it, ix) => ({
      text: "• " + it,
      options: { color: NAVY, breakLine: ix < g.items.length - 1 },
    }));
    s.addText(items, {
      x: x + 0.15, y: gy + 0.5, w: gw - 0.3, h: gh - 0.55,
      fontFace: "Courier New", fontSize: 10.5, valign: "top", margin: 0, paraSpaceAfter: 2,
    });
  });

  addFooter(s, 5);
}

// ===========================================================================
// SLIDE 6 — Methodology Journey (5 methods comparison)
// ===========================================================================
{
  const s = newSlide();
  addBrandHeader(s);

  addSectionHeader(s, "Methodology Journey — five strategies tested", 0.6, 0.95, 12);

  // Chart on left (the methodology journey bar chart)
  s.addImage({
    path: path.join(CHARTS, "04_methodology_journey.png"),
    x: 0.5, y: 1.7, w: 7.8, h: 3.6,
  });

  // 5 small method cards on the right summarizing each
  const rx = 8.5, rw = 4.3, ry0 = 1.7, rh = 0.82;
  const methods = [
    { tag: "1 — WEAK SUPERVISION", title: "Formula labels", body: "F1=0.44 · CV-test gap 0.55 (leakage)", color: RED },
    { tag: "2 — LLM SUPERVISED",   title: "400 LLM labels", body: "F1=0.704 · +10 over SBERT baseline", color: BLUE },
    { tag: "3 — SELF-TRAIN v1",    title: "Absolute threshold", body: "F1=0.564 · class balance flipped", color: RED },
    { tag: "4 — SELF-TRAIN v2",    title: "Percentile + diversified", body: "F1=0.579 · hard cases excluded", color: RED },
    { tag: "5 — ACTIVE LEARNING",  title: "Uncertainty-band sampling", body: "F1=0.769 · +6.5 over supervised", color: GREEN },
  ];
  methods.forEach((m, i) => {
    const y = ry0 + i * (rh + 0.12);
    s.addShape(pres.shapes.RECTANGLE, {
      x: rx, y, w: 0.1, h: rh,
      fill: { color: m.color }, line: { color: m.color },
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: rx + 0.1, y, w: rw - 0.1, h: rh,
      fill: { color: CARD_BG }, line: { color: CARD_BORDER, width: 1 },
    });
    s.addText(m.tag, {
      x: rx + 0.25, y: y + 0.07, w: rw - 0.35, h: 0.22,
      fontFace: FONT_BODY, fontSize: 9, bold: true, color: m.color, charSpacing: 2, margin: 0,
    });
    s.addText(m.title, {
      x: rx + 0.25, y: y + 0.28, w: rw - 0.35, h: 0.26,
      fontFace: FONT_HEAD, fontSize: 13, bold: true, color: NAVY, margin: 0,
    });
    s.addText(m.body, {
      x: rx + 0.25, y: y + 0.55, w: rw - 0.35, h: 0.24,
      fontFace: FONT_BODY, fontSize: 10.5, color: BODY, margin: 0,
    });
  });

  // Closing italic insight
  addItalicTagline(
    s,
    "Confidence-based selection (self-training) fails. Uncertainty-based selection (active learning) wins. Same label budget — opposite mechanisms.",
    0.6, SH - 0.95, SW - 1.2,
    { bold: true, color: NAVY, fontSize: 13 },
  );

  addFooter(s, 6);
}

// ===========================================================================
// SLIDE 7 — Final Results
// ===========================================================================
{
  const s = newSlide();
  addBrandHeader(s);

  addSectionHeader(s, "Final Results — F1 = 0.769 on 100-pair LLM gold standard", 0.6, 0.95, 12);

  // Big result chart on left
  s.addImage({
    path: path.join(CHARTS, "02_main_performance.png"),
    x: 0.4, y: 1.7, w: 7.5, h: 3.7,
  });

  // Right column: headline metric box + delta cards
  const rx = 8.2, rw = 4.7;

  // Headline F1 callout
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: rx, y: 1.7, w: rw, h: 1.6,
    fill: { color: GREEN }, line: { color: GREEN }, rectRadius: 0.1,
  });
  s.addText("FINAL MODEL F1", {
    x: rx + 0.2, y: 1.8, w: rw - 0.4, h: 0.3,
    fontFace: FONT_BODY, fontSize: 11, bold: true, color: "FFFFFF", charSpacing: 4, margin: 0,
  });
  s.addText("0.769", {
    x: rx + 0.2, y: 2.1, w: rw - 0.4, h: 1.0,
    fontFace: FONT_HEAD, fontSize: 72, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0,
  });

  // Lift cards
  const liftCards = [
    { label: "VS SBERT-ONLY BASELINE",   val: "+0.165 F1", body: "Active learning beats semantic-similarity-only by 16.5 percentage points." },
    { label: "VS SUPERVISED BASELINE",   val: "+0.066 F1", body: "200 uncertainty-band labels added 6.6 F1 points on top of 400-label baseline." },
  ];
  let ly = 3.5;
  liftCards.forEach((l) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: rx, y: ly, w: rw, h: 0.8,
      fill: { color: CARD_BG }, line: { color: CARD_BORDER, width: 1 }, rectRadius: 0.08,
    });
    s.addText(l.label, {
      x: rx + 0.2, y: ly + 0.1, w: rw * 0.55, h: 0.25,
      fontFace: FONT_BODY, fontSize: 9, bold: true, color: BLUE, charSpacing: 3, margin: 0,
    });
    s.addText(l.body, {
      x: rx + 0.2, y: ly + 0.35, w: rw * 0.55, h: 0.4,
      fontFace: FONT_BODY, fontSize: 10, color: BODY, margin: 0,
    });
    s.addText(l.val, {
      x: rx + rw * 0.55 + 0.1, y: ly + 0.1, w: rw * 0.4, h: 0.7,
      fontFace: FONT_HEAD, fontSize: 22, bold: true, color: GREEN,
      align: "right", valign: "middle", margin: 0,
    });
    ly += 0.92;
  });

  // Confusion matrix mini callout
  s.addText(
    "Test confusion matrix:  TP=20, FN=2, FP=10, TN=68  ·  Precision=0.667, Recall=0.909, P@10=0.90",
    {
      x: 0.6, y: SH - 0.9, w: SW - 1.2, h: 0.4,
      fontFace: FONT_BODY, fontSize: 11, italic: true, color: BODY,
      align: "center", valign: "middle", margin: 0,
    },
  );

  addFooter(s, 7);
}

// ===========================================================================
// SLIDE 8 — Feature Importance (refuting the SBERT-only critique)
// ===========================================================================
{
  const s = newSlide();
  addBrandHeader(s);

  addSectionHeader(s, "Feature Importance — Do skill features add real signal?", 0.6, 0.95, 12);

  // Top right: SBERT vs Skill comparison chart
  s.addImage({
    path: path.join(CHARTS, "06_sbert_vs_skill.png"),
    x: 7.6, y: 1.55, w: 5.3, h: 3.0,
  });

  // Left: the question framed + answer (more height so text doesn't collide)
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.6, y: 1.55, w: 6.7, h: 1.95,
    fill: { color: BLUE_LIGHT }, line: { color: BLUE_BORDER, width: 1 }, rectRadius: 0.08,
  });
  s.addText('THE CRITIQUE TO ADDRESS', {
    x: 0.85, y: 1.68, w: 6.2, h: 0.28,
    fontFace: FONT_BODY, fontSize: 10, bold: true, color: BLUE, charSpacing: 3, margin: 0,
  });
  s.addText('"Your model is just SBERT dressed up with extra features."', {
    x: 0.85, y: 1.97, w: 6.2, h: 0.32,
    fontFace: FONT_BODY, fontSize: 13, italic: true, color: NAVY, valign: "top", margin: 0,
  });
  s.addText(
    "Empirically refuted: a LogReg trained on the 7 non-SBERT features achieves F1=0.706 — beating an SBERT-only LogReg (F1=0.628) by 7.8 points.",
    {
      x: 0.85, y: 2.45, w: 6.2, h: 0.95,
      fontFace: FONT_BODY, fontSize: 13, color: NAVY, valign: "top", margin: 0,
    },
  );

  // Three method panels summary — pushed down clear of the critique box
  s.addText("Three independent feature-importance methods", {
    x: 0.6, y: 4.75, w: 12.0, h: 0.3,
    fontFace: FONT_HEAD, fontSize: 14, bold: true, color: NAVY, margin: 0,
  });
  const methods3 = [
    { tag: "COEFFICIENTS", body: "embedding_similarity = +0.98 (largest). skill_overlap = +0.72. Multicollinearity between skill_overlap & weighted_skill_score." },
    { tag: "PERMUTATION",   body: "embedding=0.25, title=0.12, num_missing_skills=0.11. SBERT strongest, but 5 features all contribute 0.05–0.12." },
    { tag: "ABLATION",      body: "Drop title_similarity: −0.116 F1. Drop num_missing_skills: −0.088. Drop embedding: −0.063. Title hurts more than SBERT." },
  ];
  methods3.forEach((m, i) => {
    const totalW = SW - 1.2;
    const cardW = (totalW - 2 * 0.2) / 3;
    const x = 0.6 + i * (cardW + 0.2);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: 5.15, w: cardW, h: 1.35,
      fill: { color: CARD_BG }, line: { color: CARD_BORDER, width: 1 }, rectRadius: 0.06,
    });
    s.addText(m.tag, {
      x: x + 0.15, y: 5.25, w: cardW - 0.3, h: 0.26,
      fontFace: FONT_BODY, fontSize: 10, bold: true, color: BLUE, charSpacing: 3, margin: 0,
    });
    s.addText(m.body, {
      x: x + 0.15, y: 5.55, w: cardW - 0.3, h: 0.9,
      fontFace: FONT_BODY, fontSize: 10.5, color: BODY, valign: "top", margin: 0,
    });
  });

  // Closing italic
  addItalicTagline(
    s,
    "Skill features alone outperform SBERT alone by +7.8 F1. The senior's hypothesis is refuted with concrete numbers.",
    0.6, SH - 0.95, SW - 1.2,
    { bold: true, color: BLUE, fontSize: 13 },
  );

  addFooter(s, 8);
}

// ===========================================================================
// SLIDE 9 — LLM Labeling + Why a Reranker is Needed
// ===========================================================================
{
  const s = newSlide();
  addBrandHeader(s);

  addSectionHeader(s, "LLM Labeling & Why a Reranker is Needed", 0.6, 0.95, 12);

  // Left chart: per-source positive rates
  s.addImage({
    path: path.join(CHARTS, "08_per_source_breakdown.png"),
    x: 0.4, y: 1.7, w: 7.6, h: 3.6,
  });

  // Right: LLM agreement summary
  const rx = 8.2, rw = 4.7;

  // Headline finding card
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: rx, y: 1.7, w: rw, h: 1.55,
    fill: { color: BLUE }, line: { color: BLUE }, rectRadius: 0.1,
  });
  s.addText("47.5%", {
    x: rx + 0.2, y: 1.8, w: rw - 0.4, h: 0.9,
    fontFace: FONT_HEAD, fontSize: 56, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0,
  });
  s.addText(
    "of SBERT top-50 retrievals are real matches per LLM consensus — the rest are plausible-looking false positives.",
    {
      x: rx + 0.2, y: 2.75, w: rw - 0.4, h: 0.5,
      fontFace: FONT_BODY, fontSize: 11, color: "FFFFFF",
      align: "center", valign: "top", margin: 0,
    },
  );

  // 3-LLM agreement stats
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: rx, y: 3.45, w: rw, h: 1.95,
    fill: { color: CARD_BG }, line: { color: CARD_BORDER, width: 1 }, rectRadius: 0.08,
  });
  s.addText("3-LLM CONSENSUS — 500-PAIR GOLD", {
    x: rx + 0.2, y: 3.55, w: rw - 0.4, h: 0.3,
    fontFace: FONT_BODY, fontSize: 10, bold: true, color: BLUE, charSpacing: 3, margin: 0,
  });
  s.addText([
    { text: "Unanimous (3-0):  ", options: { color: NAVY } },
    { text: "71.0%", options: { color: NAVY, bold: true, breakLine: true } },
    { text: "Disputed (2-1):  ",  options: { color: NAVY } },
    { text: "29.0%", options: { color: NAVY, bold: true, breakLine: true } },
    { text: "Mean pairwise:    ", options: { color: NAVY } },
    { text: "80.7%", options: { color: NAVY, bold: true, breakLine: true } },
    { text: "Chance-adjusted κ:", options: { color: NAVY } },
    { text: " 0.61",   options: { color: NAVY, bold: true, breakLine: true } },
    { text: "Claude pos rate:  ", options: { color: BODY } },
    { text: "22.4%", options: { color: BODY, bold: true, breakLine: true } },
    { text: "GPT pos rate:     ", options: { color: BODY } },
    { text: "13.6%", options: { color: BODY, bold: true, breakLine: true } },
    { text: "Gemini pos rate:  ", options: { color: BODY } },
    { text: "26.6%", options: { color: BODY, bold: true } },
  ], {
    x: rx + 0.2, y: 3.85, w: rw - 0.4, h: 1.5,
    fontFace: "Courier New", fontSize: 11, valign: "top", margin: 0, paraSpaceAfter: 1,
  });

  addItalicTagline(
    s,
    "SBERT is great at retrieval — but loses precision at the top of the ranked list. A reranker fills that gap.",
    0.6, SH - 0.95, SW - 1.2,
    { bold: true, color: NAVY, fontSize: 13 },
  );

  addFooter(s, 9);
}

// ===========================================================================
// SLIDE 10 — Deployability & Plaksha Use Cases
// ===========================================================================
{
  const s = newSlide();
  addBrandHeader(s);

  addSectionHeader(s, "Deployability — Dashboard & Plaksha Use Cases", 0.6, 0.95, 12);

  // Dashboard mock callout (left)
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.6, y: 1.7, w: 6.2, h: 3.85,
    fill: { color: CARD_BG }, line: { color: CARD_BORDER, width: 1 }, rectRadius: 0.1,
  });
  s.addText("CAREER PATH DASHBOARD (PROTOTYPE)", {
    x: 0.8, y: 1.85, w: 5.8, h: 0.3,
    fontFace: FONT_BODY, fontSize: 10, bold: true, color: BLUE, charSpacing: 3, margin: 0,
  });

  const dashSections = [
    { num: "1", title: "Match score",     body: "Probability + percentile rank vs other candidates for that role" },
    { num: "2", title: "Skill gap report", body: "Skills you have + missing skills ranked by ESCO importance" },
    { num: "3", title: "2-year roadmap",  body: "Quarterly skill plan aggregated across top-5 target jobs" },
    { num: "4", title: "Forward simulation", body: "Re-scores you after adding learned skills (real model prediction)" },
  ];
  dashSections.forEach((d, i) => {
    const x = 0.8, y = 2.25 + i * 0.78, w = 5.8;
    // numbered circle
    s.addShape(pres.shapes.OVAL, {
      x, y: y + 0.07, w: 0.4, h: 0.4,
      fill: { color: BLUE }, line: { color: BLUE },
    });
    s.addText(d.num, {
      x, y: y + 0.07, w: 0.4, h: 0.4,
      fontFace: FONT_HEAD, fontSize: 14, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    s.addText(d.title, {
      x: x + 0.55, y: y + 0.05, w: w - 0.55, h: 0.3,
      fontFace: FONT_HEAD, fontSize: 14, bold: true, color: NAVY, margin: 0,
    });
    s.addText(d.body, {
      x: x + 0.55, y: y + 0.35, w: w - 0.55, h: 0.4,
      fontFace: FONT_BODY, fontSize: 11, color: BODY, margin: 0,
    });
  });

  // Right column: Plaksha use cases
  const rx = 7.2, rw = 5.7, ry = 1.7;
  s.addText("Plaksha-specific scenarios", {
    x: rx, y: ry, w: rw, h: 0.35,
    fontFace: FONT_HEAD, fontSize: 16, bold: true, color: NAVY, margin: 0,
  });

  const useCases = [
    { tag: "PLACEMENT PORTAL",   title: "Internship matching tool", body: "When a student logs in, surface top-K live internship matches with per-job skill-gap reports." },
    { tag: "COURSE PLANNING",    title: "Course-to-career mapping", body: "Given transcript + target role (e.g. ML engineer), recommend electives next semester." },
    { tag: "CAREER ADVISORY",    title: "Skill-development advisor", body: "Aggregate gaps across student's top-10 dream jobs → 5 highest-impact skills to learn." },
  ];
  let uy = 2.05;
  const uh = 1.38;
  useCases.forEach((u) => {
    addCard(s, rx, uy, rw, uh, {
      tag: u.tag, title: u.title, body: u.body,
      titleSize: 14, bodySize: 11, titleH: 0.32,
    });
    uy += uh + 0.18;
  });

  addFooter(s, 10);
}

// ===========================================================================
// SLIDE 11 — Limitations + Future Work + Closing
// ===========================================================================
{
  const s = newSlide();
  addBrandHeader(s);

  addSectionHeader(s, "Limitations & Future Work", 0.6, 0.95, 12);

  // Limitations (left column)
  s.addText("Known limitations (disclosed honestly)", {
    x: 0.6, y: 1.7, w: 6.0, h: 0.35,
    fontFace: FONT_HEAD, fontSize: 14, bold: true, color: NAVY, margin: 0,
  });

  const limits = [
    "Test set is small (N=100); F1 confidence ≈ ±0.07",
    "LLM consensus is correlated noise, not ground truth",
    "Skill vocabulary still contains residual noise tokens",
    "Individual predictions can be domain-incoherent (dashboard adds category filter)",
    "English-only, IT-heavy corpus; cross-domain transfer untested",
    "No real-world deployment validation",
  ];
  s.addText(limits.map((t, i) => ({
    text: t,
    options: { color: BODY, bullet: { code: "25CF" }, breakLine: i < limits.length - 1 },
  })), {
    x: 0.6, y: 2.1, w: 6.0, h: 2.8,
    fontFace: FONT_BODY, fontSize: 12, valign: "top", margin: 0, paraSpaceAfter: 6,
  });

  // Future work (right column)
  s.addText("Future work (tied to literature-survey gaps)", {
    x: 7.0, y: 1.7, w: 5.8, h: 0.35,
    fontFace: FONT_HEAD, fontSize: 14, bold: true, color: NAVY, margin: 0,
  });

  const futureCards = [
    { tag: "LONGITUDINAL VALIDATION", title: "Cohort study", body: "12-month follow-up: does following recommendations actually change employment?" },
    { tag: "DOMAIN TRANSFER",         title: "Cross-industry test set", body: "Healthcare, legal, creative — beyond English IT." },
    { tag: "BIAS AUDIT",              title: "Demographic disaggregation", body: "Per-group precision/recall; published audit." },
  ];
  let fy = 2.05;
  const fh = 1.35;
  futureCards.forEach((f) => {
    addCard(s, 7.0, fy, 5.8, fh, {
      tag: f.tag, title: f.title, body: f.body,
      titleSize: 13, bodySize: 11, titleH: 0.3,
    });
    fy += fh + 0.18;
  });

  // Closing tagline at the bottom (like reference deck's italic line)
  s.addText([
    { text: "Resume–job matching is fundamentally subjective.",
      options: { color: NAVY, italic: true, breakLine: true } },
    { text: "We built a system that quantifies the gap honestly — and tells candidates which skills to learn next.",
      options: { color: BLUE, italic: true, bold: true } },
  ], {
    x: 0.6, y: SH - 1.25, w: SW - 1.2, h: 0.75,
    fontFace: FONT_BODY, fontSize: 15, align: "center", valign: "middle", margin: 0,
  });

  addFooter(s, 11);
}

// ---------------------------------------------------------------------------
// Save
// ---------------------------------------------------------------------------
pres.writeFile({ fileName: OUT }).then(function(p) {
  console.log("[done] wrote " + p);
}).catch(function(err) {
  console.error("ERROR:", err);
  process.exit(1);
});
