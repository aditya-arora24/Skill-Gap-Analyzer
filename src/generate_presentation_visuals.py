"""
generate_presentation_visuals.py
=================================
Build the full set of slide-ready charts and tables for the endterm
presentation. All numbers in this file are the LOCKED-IN final results
from the run logs — no re-computation, no risk of values shifting.

Output directory: outputs/presentation/

Charts produced (each as a 1920×1080 or similar PNG at 200 dpi):
  01_lit_review_table.png            literature comparison: P1/P2/P3 vs ours
  02_main_performance.png            F1 across all 6 approaches (the headline)
  03_full_metrics_table.png          comparison table with all 8 metrics
  04_methodology_journey.png         F1 progression through the 5 methodologies
  05_feature_importance_3panel.png   coefficients / permutation / ablation
  06_sbert_vs_skill.png              the senior's challenge, refuted
  07_esco_impact.png                 before/after metrics from ESCO expansion
  08_per_source_breakdown.png        positive rate by pool source
  09_inter_llm_agreement.png         3-LLM consensus statistics
  10_confusion_matrix.png            confusion matrix for the final model
  11_pk_comparison.png               Precision@K across approaches

Markdown tables produced (for copy-paste into slides):
  tables/lit_review_comparison.md
  tables/performance_metrics.md
  tables/feature_importance.md
  tables/dataset_stats.md
  tables/esco_impact.md

Run:
    python "src/generate_presentation_visuals.py"
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR     = PROJECT_ROOT / "outputs" / "presentation"
TABLES_DIR  = OUT_DIR / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style: clean, slide-friendly, high contrast
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        13,
    "axes.titleweight": "bold",
    "axes.titlesize":   16,
    "axes.labelsize":   13,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
    "legend.frameon":   False,
    "figure.dpi":       150,
    "savefig.dpi":      200,
    "savefig.bbox":     "tight",
    "savefig.facecolor": "white",
})

# Color palette (consistent across all charts)
COLOR_WINNER     = "#10b981"   # emerald green — active learning / winner
COLOR_SUPERVISED = "#3b82f6"   # blue — strong baseline
COLOR_NEUTRAL    = "#6366f1"   # indigo — other approaches
COLOR_WEAK       = "#94a3b8"   # slate — baseline / weak
COLOR_FAIL       = "#ef4444"   # red — self-training failures
COLOR_ACCENT     = "#f59e0b"   # amber — accent / highlight
COLOR_DARK       = "#1f2937"
COLOR_LIGHT_BG   = "#f9fafb"

# ---------------------------------------------------------------------------
# Locked-in results data (from run logs)
# ---------------------------------------------------------------------------
APPROACHES = [
    # (name, F1, accuracy, precision, recall, P@5, P@10, P@20, color, kind)
    ("SBERT only",              0.604, 0.79, 0.516, 0.727, 1.00, 0.70, 0.65, COLOR_WEAK,    "baseline"),
    ("Weak supervised",         0.440, 0.72, 0.393, 0.500, 0.40, 0.60, 0.40, COLOR_FAIL,    "weak"),
    ("Self-trained v1",         0.564, 0.83, 0.647, 0.500, 1.00, 0.80, 0.70, COLOR_FAIL,    "selftrain"),
    ("Self-trained v2",         0.579, 0.84, 0.688, 0.500, 1.00, 0.90, 0.70, COLOR_FAIL,    "selftrain"),
    ("LLM supervised",          0.704, 0.84, 0.594, 0.864, 1.00, 0.80, 0.70, COLOR_SUPERVISED, "strong"),
    ("Active learning (final)", 0.769, 0.88, 0.667, 0.909, 1.00, 0.90, 0.70, COLOR_WINNER,  "winner"),
]

# Feature importance (from feature_analysis/run_metadata.json)
FEATURES = [
    "embedding_similarity",
    "skill_overlap",
    "num_missing_skills",
    "weighted_skill_score",
    "tfidf_similarity",
    "title_similarity",
    "avg_missing_skill_importance",
    "years_of_experience",
]
COEFS = {
    "embedding_similarity":         0.979,
    "skill_overlap":                0.716,
    "num_missing_skills":          -0.699,
    "weighted_skill_score":        -0.697,
    "tfidf_similarity":             0.560,
    "title_similarity":             0.497,
    "avg_missing_skill_importance":-0.279,
    "years_of_experience":          0.004,
}
PERM = {
    "embedding_similarity":         0.250,
    "title_similarity":             0.118,
    "num_missing_skills":           0.105,
    "tfidf_similarity":             0.085,
    "skill_overlap":                0.071,
    "weighted_skill_score":         0.053,
    "avg_missing_skill_importance": 0.011,
    "years_of_experience":          0.000,
}
ABLATION_DROP = {
    "title_similarity":             0.116,
    "num_missing_skills":           0.088,
    "embedding_similarity":         0.063,
    "tfidf_similarity":             0.035,
    "years_of_experience":          0.000,
    "avg_missing_skill_importance":-0.006,
    "weighted_skill_score":        -0.015,
    "skill_overlap":               -0.047,
}

# SBERT vs Skill comparison
SBERT_ONLY_F1   = 0.628
SKILL_ONLY_F1   = 0.706
FULL_MODEL_F1   = 0.769

# ESCO vocabulary impact (before / after expansion)
ESCO_IMPACT = [
    # (metric, before, after, unit)
    ("skill_overlap > 0",        36.0,    68.0,    "%"),
    ("years_of_experience > 0",  39.0,    48.0,    "%"),
    ("title_similarity unique", 7286,    31254,   "values"),
    ("avg skills per resume",     3.4,    13.2,    "skills"),
    ("avg skills per job",        1.7,     5.96,   "skills"),
]

# Per-source breakdown (positive rates in 500-pair gold, inter-LLM unanimity)
PER_SOURCE = [
    # (source, n, pos_rate_%, unanimous_%, color)
    ("topK",   200, 47.5, 46.5, "#3b82f6"),
    ("A_mid",  150,  3.3, 91.3, "#10b981"),
    ("B_xcat", 100,  8.0, 82.0, "#f59e0b"),
    ("C_rand",  50,  8.0, 86.0, "#a855f7"),
]

# Inter-LLM agreement (500-pair gold standard)
LLM_AGREEMENT = {
    "unanimous":      71.0,
    "disputed":       29.0,
    "claude_gpt":     77.2,
    "claude_gemini":  89.4,
    "gpt_gemini":     75.4,
    "mean_pairwise":  80.7,
    "kappa":          0.61,
    "rate_claude":    22.4,
    "rate_gpt":       13.6,
    "rate_gemini":    26.6,
    "rate_majority":  22.4,
}

# Confusion matrix for active-learning model (final)
# Test set: 100 pairs (22 pos, 78 neg). Acc=0.88, P=0.667, R=0.909.
# TP=20, FN=2, FP=10, TN=68.
CONFUSION = {
    "TP": 20, "FN": 2,
    "FP": 10, "TN": 68,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_fig(name: str):
    path = OUT_DIR / name
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[write] {path}")


def add_value_labels(ax, bars, fmt="{:.3f}", padding=0.005, fontsize=10):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + padding,
                fmt.format(h), ha="center", va="bottom",
                fontsize=fontsize, fontweight="bold")


# ===========================================================================
# 01 — Literature comparison table (rendered as matplotlib image)
# ===========================================================================
def chart_01_lit_review_table():
    rows = [
        # (dim, P1, P2, P3, ours)
        ("Skill extraction",
         "BERT/RoBERTa NER",
         "spaCy + Levenshtein",
         "TF-IDF + BERT",
         "ESCO vocab (20k) + flashtext"),
        ("Skill taxonomy",
         "ESCO + O*NET",
         "None",
         "None",
         "ESCO + O*NET + depth weights"),
        ("Match scoring",
         "Sentence-BERT cosine",
         "RandomForest on 7 features",
         "BERT cosine + classifier",
         "LogReg on 8 features"),
        ("Label source",
         "Undisclosed",
         "K-Means clusters (flagged)",
         "Undisclosed",
         "3-LLM majority vote"),
        ("Inter-rater agreement",
         "Not reported",
         "N/A",
         "Not reported",
         "71% unanimous, κ=0.61"),
        ("Reported metric",
         "0.90 accuracy (RoBERTa)",
         "0.72 accuracy (vs K-Means)",
         "0.918 accuracy",
         "0.769 F1 (LLM gold)"),
        ("Methodology compared",
         "BERT vs RoBERTa",
         "RF/XGB/ANN",
         "BERT vs TF-IDF",
         "5 methodologies on same test"),
        ("Leakage detection",
         "Not addressed",
         "Not addressed",
         "Not addressed",
         "CV 0.99 → test 0.44"),
        ("Self-training tested",
         "No", "No", "No",
         "Yes — 2 variants, both fail"),
        ("Active learning tested",
         "No", "No", "No",
         "Yes — +6.5 F1 lift"),
        ("Production deployment",
         "Docker/K8s stack",
         "Not demonstrated",
         "200+ user pilot",
         "Prototype dashboard"),
    ]

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")

    col_labels = ["", "P1 — Dash et al.\n(IJEDR 2025)",
                   "P2 — Daberao et al.\n(IEEE GITCON 2025)",
                   "P3 — Sribharathi et al.\n(IEEE ICSSS 2025)",
                   "OUR PROJECT"]
    cell_text  = [[d, p1, p2, p3, ours] for (d, p1, p2, p3, ours) in rows]

    table = ax.table(
        cellText=cell_text, colLabels=col_labels,
        cellLoc="left", loc="center",
        colWidths=[0.18, 0.20, 0.20, 0.20, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 2.0)

    # Style: header row dark, body alternating, our column highlighted
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(weight="bold", color="white")
        elif c == 0:
            cell.set_facecolor("#f1f5f9")
            cell.set_text_props(weight="bold")
        elif c == 4:
            cell.set_facecolor("#dcfce7")  # light green for our column
            if r > 0:
                cell.set_text_props(weight="bold", color="#065f46")
        elif r % 2 == 0:
            cell.set_facecolor("#f9fafb")
        else:
            cell.set_facecolor("white")

    ax.set_title("Literature Comparison — Our Project vs. Three 2025 Papers",
                 fontsize=18, pad=20)
    fig.text(0.5, 0.02,
             "Green column: dimensions where our methodology advances the field. "
             "Source: P1 (Dash et al., IJEDR 2025), P2 (Daberao et al., IEEE GITCON 2025), "
             "P3 (Sribharathi et al., IEEE ICSSS 2025).",
             ha="center", fontsize=9, style="italic", color="#6b7280")
    save_fig("01_lit_review_table.png")


# ===========================================================================
# 02 — Main performance comparison (the headline chart)
# ===========================================================================
def chart_02_main_performance():
    names  = [a[0] for a in APPROACHES]
    f1s    = [a[1] for a in APPROACHES]
    colors = [a[8] for a in APPROACHES]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    bars = ax.bar(names, f1s, color=colors, edgecolor="white", linewidth=2)
    add_value_labels(ax, bars, fmt="{:.3f}", padding=0.012, fontsize=12)

    # Highlight the winner
    bars[-1].set_edgecolor(COLOR_DARK)
    bars[-1].set_linewidth(3)

    # Reference lines for the baselines
    ax.axhline(0.604, color=COLOR_WEAK, linestyle=":", alpha=0.5, linewidth=1.5)
    ax.text(0.02, 0.612, "SBERT-only baseline (0.604)", color=COLOR_WEAK,
            transform=ax.get_yaxis_transform(), fontsize=9, va="bottom")

    ax.set_ylabel("F1 Score on held-out 100-pair gold standard", fontsize=14)
    ax.set_title("Final Performance — F1 Across All Six Approaches", fontsize=18, pad=15)
    ax.set_ylim(0, 0.92)
    ax.tick_params(axis="x", rotation=8)

    # Annotation: the active learning lift
    ax.annotate(
        "+0.165 over SBERT\n+0.066 over supervised",
        xy=(5, 0.769), xytext=(4.0, 0.86),
        ha="center", fontsize=11, fontweight="bold", color=COLOR_DARK,
        arrowprops=dict(arrowstyle="->", color=COLOR_DARK, lw=1.5),
    )

    fig.text(0.5, -0.01,
             "Same 100-pair held-out test set across all approaches. "
             "Labels: 3-LLM consensus (Claude + GPT + Gemini majority vote). "
             "random_state = 42 throughout.",
             ha="center", fontsize=9, style="italic", color="#6b7280")
    save_fig("02_main_performance.png")


# ===========================================================================
# 03 — Full metrics table
# ===========================================================================
def chart_03_full_metrics_table():
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.axis("off")

    col_labels = ["Approach", "Accuracy", "Precision", "Recall", "F1",
                   "P@5", "P@10", "P@20"]
    cell_text = []
    for (name, f1, acc, prec, rec, p5, p10, p20, _, _) in APPROACHES:
        cell_text.append([name, f"{acc:.2f}", f"{prec:.3f}",
                          f"{rec:.3f}", f"{f1:.3f}",
                          f"{p5:.2f}", f"{p10:.2f}", f"{p20:.2f}"])

    table = ax.table(
        cellText=cell_text, colLabels=col_labels,
        cellLoc="center", loc="center",
        colWidths=[0.26, 0.105, 0.105, 0.105, 0.105, 0.085, 0.085, 0.085],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.1)

    # Find winner row index
    winner_row = max(range(len(APPROACHES)), key=lambda i: APPROACHES[i][1]) + 1

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(weight="bold", color="white")
        elif r == winner_row:
            cell.set_facecolor("#dcfce7")
            cell.set_text_props(weight="bold", color="#065f46")
        elif c == 0:
            cell.set_facecolor("#f1f5f9")
            cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f9fafb")
        else:
            cell.set_facecolor("white")

    ax.set_title("Performance Metrics — All Six Approaches on Same 100-Pair Test",
                 fontsize=17, pad=15)
    fig.text(0.5, 0.02,
             "Green row: final model. Bold-formatted P@K columns measure top-K ranking "
             "quality — the production-relevant metric for a reranker.",
             ha="center", fontsize=9, style="italic", color="#6b7280")
    save_fig("03_full_metrics_table.png")


# ===========================================================================
# 04 — Methodology journey (the story of why we ended up at active learning)
# ===========================================================================
def chart_04_methodology_journey():
    # Order: chronological / methodological story
    steps = [
        ("SBERT only\n(baseline)",        0.604, COLOR_WEAK),
        ("Weak supervised\n(formula)",    0.440, COLOR_FAIL),
        ("LLM supervised\n(400 labels)",  0.704, COLOR_SUPERVISED),
        ("Self-trained v1\n(absolute)",   0.564, COLOR_FAIL),
        ("Self-trained v2\n(percentile)", 0.579, COLOR_FAIL),
        ("Active learning\n(uncertainty)",0.769, COLOR_WINNER),
    ]
    names  = [s[0] for s in steps]
    f1s    = [s[1] for s in steps]
    colors = [s[2] for s in steps]

    fig, ax = plt.subplots(figsize=(14, 6.5))
    x = np.arange(len(steps))
    bars = ax.bar(x, f1s, color=colors, edgecolor="white", linewidth=2, width=0.65)
    add_value_labels(ax, bars, fmt="{:.3f}", padding=0.012, fontsize=12)

    # Connecting line showing the journey
    ax.plot(x, f1s, color="#1f2937", linewidth=2, alpha=0.4, zorder=0,
            marker="o", markersize=8, markerfacecolor="white",
            markeredgecolor="#1f2937", markeredgewidth=2)

    # Highlight the winning bar
    bars[-1].set_edgecolor(COLOR_DARK)
    bars[-1].set_linewidth(3)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("F1 on held-out gold", fontsize=14)
    ax.set_title("The Methodology Journey — From Weak Supervision to Active Learning",
                 fontsize=17, pad=15)
    ax.set_ylim(0, 0.96)

    # Annotations sit above bars (not inside them), with offset to avoid overlap
    # with the numeric value labels added by add_value_labels above.
    annotations = [
        (1, "leakage detected\n(CV 0.99 → test 0.44)", COLOR_FAIL,    0.79),
        (2, "external labels\nfix leakage",           COLOR_SUPERVISED, 0.82),
        (4, "confidence-based\nselection fails",      COLOR_FAIL,    0.69),
        (5, "uncertainty-based\nwins",                COLOR_WINNER,  0.90),
    ]
    for x_pos, text, color, y_pos in annotations:
        ax.text(x_pos, y_pos, text, ha="center", fontsize=9,
                color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           edgecolor=color, linewidth=1, alpha=0.95))

    save_fig("04_methodology_journey.png")


# ===========================================================================
# 05 — Feature importance: 3-panel chart
# ===========================================================================
def chart_05_feature_importance():
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # Panel 1: signed coefficients
    ax = axes[0]
    sorted_feats = sorted(COEFS.keys(), key=lambda f: COEFS[f])
    vals = [COEFS[f] for f in sorted_feats]
    colors = ["#ef4444" if v < 0 else "#10b981" for v in vals]
    bars = ax.barh(sorted_feats, vals, color=colors, edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(v + (0.03 if v >= 0 else -0.03), i, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=11, fontweight="bold")
    ax.axvline(0, color="#1f2937", linewidth=1)
    ax.set_xlim(-1.0, 1.3)
    ax.set_title("(1) Scaled LogReg coefficients\n+ = pushes toward match", fontsize=14)
    ax.set_xlabel("Coefficient")
    ax.grid(axis="x", alpha=0.3)

    # Panel 2: permutation importance
    ax = axes[1]
    sorted_feats = sorted(PERM.keys(), key=lambda f: PERM[f])
    vals = [PERM[f] for f in sorted_feats]
    bars = ax.barh(sorted_feats, vals, color="#3b82f6", edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center",
                fontsize=11, fontweight="bold")
    ax.set_xlim(0, 0.30)
    ax.set_title("(2) Permutation importance\nF1 drop when shuffled in test", fontsize=14)
    ax.set_xlabel("F1 drop")
    ax.grid(axis="x", alpha=0.3)
    ax.set_yticklabels([])

    # Panel 3: ablation (drop one feature, retrain)
    ax = axes[2]
    sorted_feats = sorted(ABLATION_DROP.keys(), key=lambda f: ABLATION_DROP[f])
    vals = [ABLATION_DROP[f] for f in sorted_feats]
    colors = ["#10b981" if v > 0 else ("#94a3b8" if v == 0 else "#f59e0b") for v in vals]
    bars = ax.barh(sorted_feats, vals, color=colors, edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(v + (0.003 if v >= 0 else -0.003), i, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=11, fontweight="bold")
    ax.axvline(0, color="#1f2937", linewidth=1)
    ax.set_xlim(-0.06, 0.15)
    ax.set_title("(3) Leave-one-feature-out\nF1 drop when retrained without", fontsize=14)
    ax.set_xlabel("F1 drop (positive = feature matters)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_yticklabels([])

    fig.suptitle("Feature Importance — Three Independent Methods on the Final Model",
                 fontsize=18, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02,
             "Multicollinearity note: skill_overlap and weighted_skill_score have near-equal "
             "opposite-sign coefficients — the model uses both jointly but individual "
             "importances are masked by their correlation.",
             ha="center", fontsize=10, style="italic", color="#6b7280")
    save_fig("05_feature_importance_3panel.png")


# ===========================================================================
# 06 — SBERT-only vs Skill-only vs Full
# ===========================================================================
def chart_06_sbert_vs_skill():
    labels = ["SBERT only\n(1 feature)", "Skill only\n(7 features\nno SBERT)",
              "Full model\n(all 8 features)"]
    vals = [SBERT_ONLY_F1, SKILL_ONLY_F1, FULL_MODEL_F1]
    colors = [COLOR_WEAK, COLOR_NEUTRAL, COLOR_WINNER]

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    bars = ax.bar(labels, vals, color=colors, edgecolor="white",
                  linewidth=2, width=0.55)
    add_value_labels(ax, bars, fmt="{:.3f}", padding=0.015, fontsize=14)
    bars[-1].set_edgecolor(COLOR_DARK)
    bars[-1].set_linewidth(3)

    ax.set_ylabel("F1 on held-out gold", fontsize=14)
    ax.set_title("Does Skill Beat SBERT? — Direct Comparison",
                 fontsize=18, pad=15)
    ax.set_ylim(0, 0.98)

    # Add the lift arrows ABOVE the bars (so they don't collide with x-tick text)
    arrow_y_start = 0.88
    ax.annotate("", xy=(0.95, arrow_y_start), xytext=(0.05, arrow_y_start - 0.04),
                arrowprops=dict(arrowstyle="->", color="#1f2937", lw=2))
    ax.text(0.5, arrow_y_start + 0.025,
            f"+{SKILL_ONLY_F1 - SBERT_ONLY_F1:.3f} F1",
            ha="center", fontsize=12, fontweight="bold", color="#1f2937")

    ax.annotate("", xy=(1.95, arrow_y_start + 0.02), xytext=(1.05, arrow_y_start - 0.02),
                arrowprops=dict(arrowstyle="->", color="#1f2937", lw=2))
    ax.text(1.5, arrow_y_start + 0.045,
            f"+{FULL_MODEL_F1 - SKILL_ONLY_F1:.3f} F1",
            ha="center", fontsize=12, fontweight="bold", color="#1f2937")

    fig.text(0.5, -0.01,
             "All three trained as LogReg on the same 600 LLM-labeled pairs, "
             "evaluated on the same 100-pair test set.\n"
             "Skill features alone (F1=0.706) outperform SBERT alone (F1=0.628) "
             "by 7.8 points — empirical refutation of the 'just SBERT' critique.",
             ha="center", fontsize=10, style="italic", color="#6b7280")
    save_fig("06_sbert_vs_skill.png")


# ===========================================================================
# 07 — ESCO vocabulary impact
# ===========================================================================
def chart_07_esco_impact():
    fig, ax = plt.subplots(figsize=(13, 6.5))

    metrics = [m[0] for m in ESCO_IMPACT]
    before = [m[1] for m in ESCO_IMPACT]
    after  = [m[2] for m in ESCO_IMPACT]

    x = np.arange(len(metrics))
    w = 0.35

    # Normalize: percent metrics on left axis, count metrics on right
    pct_idx = [0, 1]
    cnt_idx = [2, 3, 4]

    # Plot percent metrics
    bars1 = ax.bar(x[pct_idx] - w/2, [before[i] for i in pct_idx], w,
                    label="Before ESCO", color=COLOR_WEAK, edgecolor="white")
    bars2 = ax.bar(x[pct_idx] + w/2, [after[i]  for i in pct_idx], w,
                    label="After ESCO",  color=COLOR_WINNER, edgecolor="white")
    for b, v in zip(bars1, [before[i] for i in pct_idx]):
        ax.text(b.get_x() + b.get_width()/2, v + 2, f"{v:.0f}%",
                ha="center", fontsize=11, fontweight="bold")
    for b, v in zip(bars2, [after[i] for i in pct_idx]):
        ax.text(b.get_x() + b.get_width()/2, v + 2, f"{v:.0f}%",
                ha="center", fontsize=11, fontweight="bold", color=COLOR_DARK)

    ax.set_ylabel("Value (% for left bars, scaled for right)", fontsize=13)
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in ESCO_IMPACT], fontsize=10)
    ax.tick_params(axis="x", rotation=12)
    ax.set_title("ESCO Vocabulary Integration — Before vs After Impact",
                 fontsize=17, pad=15)
    ax.legend(loc="upper right", fontsize=11)

    # Plot count metrics on twin axis
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    bars3 = ax2.bar(x[cnt_idx] - w/2, [before[i] for i in cnt_idx], w,
                     color=COLOR_WEAK, edgecolor="white")
    bars4 = ax2.bar(x[cnt_idx] + w/2, [after[i]  for i in cnt_idx], w,
                     color=COLOR_WINNER, edgecolor="white")
    for b, v in zip(bars3, [before[i] for i in cnt_idx]):
        ax2.text(b.get_x() + b.get_width()/2,
                  v + (max([after[i] for i in cnt_idx]) * 0.02),
                  f"{v:,.1f}" if v < 1000 else f"{v:,.0f}",
                  ha="center", fontsize=10, fontweight="bold")
    for b, v in zip(bars4, [after[i] for i in cnt_idx]):
        ax2.text(b.get_x() + b.get_width()/2,
                  v + (max([after[i] for i in cnt_idx]) * 0.02),
                  f"{v:,.1f}" if v < 1000 else f"{v:,.0f}",
                  ha="center", fontsize=10, fontweight="bold", color=COLOR_DARK)
    ax2.set_ylim(0, max([after[i] for i in cnt_idx]) * 1.15)
    ax2.set_ylabel("Count / unique values", fontsize=13)

    fig.text(0.5, -0.04,
             "Vocabulary expanded from 355 curated tech skills to 20,253 ESCO + tech skills. "
             "skill_overlap > 0 rose from 36% to 68% of pairs.",
             ha="center", fontsize=10, style="italic", color="#6b7280")
    save_fig("07_esco_impact.png")


# ===========================================================================
# 08 — Per-source positive rate (the topK 47.5% headline finding)
# ===========================================================================
def chart_08_per_source_breakdown():
    sources    = [s[0] for s in PER_SOURCE]
    pos_rates  = [s[2] for s in PER_SOURCE]
    unanimous  = [s[3] for s in PER_SOURCE]
    colors     = [s[4] for s in PER_SOURCE]
    counts     = [s[1] for s in PER_SOURCE]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: positive rate by source
    ax = axes[0]
    bars = ax.bar(sources, pos_rates, color=colors, edgecolor="white", linewidth=2)
    add_value_labels(ax, bars, fmt="{:.1f}%", padding=1.0, fontsize=12)
    ax.set_ylabel("% judged positive by LLM majority vote", fontsize=13)
    ax.set_ylim(0, 60)
    ax.set_title("Positive Rate by Pool Source\n(500-pair gold standard)", fontsize=15)
    ax.grid(axis="y", alpha=0.3)
    # Annotation
    ax.annotate("Only 47.5% of SBERT top-50\nare real matches per LLMs",
                xy=(0, 47.5), xytext=(0.5, 56), ha="left",
                fontsize=11, fontweight="bold", color=COLOR_DARK,
                arrowprops=dict(arrowstyle="->", color=COLOR_DARK, lw=1.5))

    # Right: unanimous-vote rate
    ax = axes[1]
    bars = ax.bar(sources, unanimous, color=colors, edgecolor="white", linewidth=2)
    add_value_labels(ax, bars, fmt="{:.0f}%", padding=1.5, fontsize=12)
    ax.set_ylabel("% with 3-LLM unanimous vote", fontsize=13)
    ax.set_ylim(0, 105)
    ax.set_title("LLM Inter-Rater Unanimity by Pool Source", fontsize=15)
    ax.grid(axis="y", alpha=0.3)
    ax.annotate("topK contains the\nmost ambiguous cases",
                xy=(0, 46.5), xytext=(0.3, 25), ha="left",
                fontsize=10, fontweight="bold", color=COLOR_DARK,
                arrowprops=dict(arrowstyle="->", color=COLOR_DARK, lw=1.5))

    fig.suptitle("Why a Reranker is Needed — SBERT Retrieval Surfaces Many Bad Matches",
                 fontsize=17, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02,
             "topK = SBERT top-50 retrieval. A_mid = mid-similarity (0.30–0.50, not top-50). "
             "B_xcat = cross-category strict. C_rand = uniform random.",
             ha="center", fontsize=10, style="italic", color="#6b7280")
    save_fig("08_per_source_breakdown.png")


# ===========================================================================
# 09 — Inter-LLM agreement panel
# ===========================================================================
def chart_09_inter_llm_agreement():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: 3-LLM vote outcome distribution
    ax = axes[0]
    sizes  = [LLM_AGREEMENT["unanimous"], LLM_AGREEMENT["disputed"]]
    labels = [f"Unanimous (3-0)\n{sizes[0]:.0f}%",
              f"Disputed (2-1)\n{sizes[1]:.0f}%"]
    colors = [COLOR_WINNER, COLOR_ACCENT]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=3),
        textprops=dict(fontsize=12, fontweight="bold"),
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    ax.set_title("3-LLM Vote Distribution\non 500-pair gold standard",
                 fontsize=14)

    # Right: per-LLM positive rates + pairwise agreement
    ax = axes[1]
    llms       = ["Claude", "GPT-4o", "Gemini", "Majority"]
    pos_rates  = [LLM_AGREEMENT["rate_claude"],
                   LLM_AGREEMENT["rate_gpt"],
                   LLM_AGREEMENT["rate_gemini"],
                   LLM_AGREEMENT["rate_majority"]]
    colors_l   = ["#0ea5e9", "#10b981", "#f59e0b", COLOR_DARK]
    bars = ax.bar(llms, pos_rates, color=colors_l,
                   edgecolor="white", linewidth=2)
    for b, v in zip(bars, pos_rates):
        ax.text(b.get_x() + b.get_width()/2, v + 0.7,
                f"{v:.1f}%", ha="center",
                fontsize=12, fontweight="bold")
    ax.set_ylabel("% labeled positive", fontsize=13)
    ax.set_ylim(0, 32)
    ax.set_title("Per-LLM Positive-Class Rate", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Inter-LLM Agreement on the 500-Pair Gold Standard",
                 fontsize=17, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02,
             f"Mean pairwise agreement: {LLM_AGREEMENT['mean_pairwise']:.1f}% "
             f"(Claude-Gemini {LLM_AGREEMENT['claude_gemini']:.1f}%, "
             f"GPT-Gemini {LLM_AGREEMENT['gpt_gemini']:.1f}%, "
             f"Claude-GPT {LLM_AGREEMENT['claude_gpt']:.1f}%). "
             f"Chance-adjusted κ = {LLM_AGREEMENT['kappa']:.2f}.",
             ha="center", fontsize=10, style="italic", color="#6b7280")
    save_fig("09_inter_llm_agreement.png")


# ===========================================================================
# 10 — Confusion matrix for the final model
# ===========================================================================
def chart_10_confusion_matrix():
    cm = np.array([[CONFUSION["TN"], CONFUSION["FP"]],
                    [CONFUSION["FN"], CONFUSION["TP"]]])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Greens", vmin=0, vmax=cm.max())

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted\nNegative", "Predicted\nPositive"], fontsize=13)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual\nNegative", "Actual\nPositive"], fontsize=13)

    # Cell numbers with descriptive labels
    cell_labels = [["TN", "FP"], ["FN", "TP"]]
    cell_colors = [["#065f46", "#7f1d1d"], ["#7f1d1d", "#065f46"]]
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            ax.text(j, i, f"{cell_labels[i][j]}\n{val}",
                    ha="center", va="center",
                    fontsize=20, fontweight="bold",
                    color="white" if val > cm.max() / 2 else cell_colors[i][j])

    ax.set_title("Confusion Matrix — Active Learning Model (Final)\n"
                  "100 held-out test pairs (22 positive, 78 negative)",
                  fontsize=15, pad=15)

    # Outside text with metrics
    fig.text(0.5, -0.02,
             f"Accuracy = (TN+TP)/100 = ({CONFUSION['TN']}+{CONFUSION['TP']})/100 = 0.88   "
             f"Precision = TP/(TP+FP) = {CONFUSION['TP']}/{CONFUSION['TP']+CONFUSION['FP']} = 0.667   "
             f"Recall = TP/(TP+FN) = {CONFUSION['TP']}/{CONFUSION['TP']+CONFUSION['FN']} = 0.909",
             ha="center", fontsize=11, style="italic", color="#374151")
    plt.colorbar(im, ax=ax, shrink=0.6, label="count")
    save_fig("10_confusion_matrix.png")


# ===========================================================================
# 11 — Precision@K comparison
# ===========================================================================
def chart_11_pk_comparison():
    fig, ax = plt.subplots(figsize=(13, 6.5))

    names    = [a[0] for a in APPROACHES]
    p5_vals  = [a[5] for a in APPROACHES]
    p10_vals = [a[6] for a in APPROACHES]
    p20_vals = [a[7] for a in APPROACHES]
    colors   = [a[8] for a in APPROACHES]

    x = np.arange(len(names))
    w = 0.25

    bars1 = ax.bar(x - w, p5_vals,  w, label="P@5",  color="#3b82f6", edgecolor="white")
    bars2 = ax.bar(x,      p10_vals, w, label="P@10", color="#10b981", edgecolor="white")
    bars3 = ax.bar(x + w, p20_vals, w, label="P@20", color="#f59e0b", edgecolor="white")

    for bars, vals in [(bars1, p5_vals), (bars2, p10_vals), (bars3, p20_vals)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.015,
                    f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.tick_params(axis="x", rotation=8)
    ax.set_ylabel("Precision @ K", fontsize=14)
    ax.set_ylim(0, 1.15)
    ax.set_title("Precision @ K — Top-Ranked Predictions on the Test Set",
                 fontsize=17, pad=15)
    ax.legend(loc="upper right", fontsize=12, ncol=3)
    ax.grid(axis="y", alpha=0.3)

    fig.text(0.5, -0.01,
             "P@K is the production-relevant reranker metric: of the K highest-probability "
             "predictions, what fraction are real matches? Both Active Learning and supervised "
             "achieve perfect P@5; the gap widens at P@10 and P@20.",
             ha="center", fontsize=10, style="italic", color="#6b7280")
    save_fig("11_pk_comparison.png")


# ===========================================================================
# Markdown tables (for direct paste into slides / docs)
# ===========================================================================
def write_markdown_tables():
    # Lit review comparison table
    md = """# Literature Comparison Table

| Dimension | P1 (Dash et al., IJEDR 2025) | P2 (Daberao et al., GITCON 2025) | P3 (Sribharathi et al., ICSSS 2025) | **Our Project** |
|---|---|---|---|---|
| Skill extraction | BERT/RoBERTa NER fine-tuned | spaCy + Levenshtein | TF-IDF + BERT | **ESCO vocabulary (20k) + flashtext** |
| Skill taxonomy | ESCO + O*NET | None (string distance) | None (BERT vectors) | **ESCO + O*NET, depth-weighted importance** |
| Match scoring | Sentence-BERT cosine | 7-dim feature vector + Random Forest | BERT cosine + supervised classifier | **8-feature Logistic Regression** |
| Label source | Undisclosed | K-Means cluster labels (flagged weakness) | Undisclosed | **3-LLM consensus majority vote** |
| Inter-rater agreement disclosed | No | N/A | No | **Yes: 71% unanimous, 80.7% pairwise, κ=0.61** |
| Reported metric | 0.90 accuracy (RoBERTa) | 0.72 accuracy (vs K-Means labels) | 0.918 accuracy | **0.769 F1 on stratified LLM gold** |
| Methodology comparisons | BERT vs RoBERTa | RF vs XGBoost vs ANN | BERT vs TF-IDF baseline | **5 methodologies on same test** |
| Leakage detection | Not addressed | Not addressed | Not addressed | **Empirical: CV 0.99 → test 0.44** |
| Self-training tested | No | No | No | **Yes — 2 variants, both fail** |
| Active learning tested | No | No | No | **Yes — +6.5 F1 lift** |
| Production deployment | Docker/K8s stack | Not demonstrated | 200+ user pilot | **Prototype dashboard only** |
"""
    (TABLES_DIR / "lit_review_comparison.md").write_text(md, encoding="utf-8")

    # Performance metrics table
    md = """# Performance Metrics — All Six Approaches

Evaluated on the same 100-pair held-out gold standard (22 positive / 78 negative).
LLM consensus labels (Claude + GPT + Gemini majority vote).

| Approach | Threshold | Accuracy | Precision | Recall | F1 | P@5 | P@10 | P@20 |
|---|---|---|---|---|---|---|---|---|
"""
    sorted_approaches = sorted(APPROACHES, key=lambda a: -a[1])
    for (name, f1, acc, prec, rec, p5, p10, p20, _, _) in sorted_approaches:
        marker = "**" if "final" in name.lower() else ""
        md += (f"| {marker}{name}{marker} | "
                f"— | "
                f"{marker}{acc:.2f}{marker} | "
                f"{marker}{prec:.3f}{marker} | "
                f"{marker}{rec:.3f}{marker} | "
                f"{marker}{f1:.3f}{marker} | "
                f"{marker}{p5:.2f}{marker} | "
                f"{marker}{p10:.2f}{marker} | "
                f"{marker}{p20:.2f}{marker} |\n")
    md += "\n**Final model in bold.** Lift of active learning over SBERT-only baseline: +0.165 F1. Over plain supervised: +0.066 F1.\n"
    (TABLES_DIR / "performance_metrics.md").write_text(md, encoding="utf-8")

    # Feature importance table
    md = """# Feature Importance — Three Methods on the Final Model

| Feature | Coefficient | \\|coef\\| | Permutation F1 drop | Ablation F1 drop |
|---|---|---|---|---|
"""
    # Sort by absolute coefficient descending
    for f in sorted(FEATURES, key=lambda f: -abs(COEFS[f])):
        md += (f"| `{f}` | "
                f"{COEFS[f]:+.3f} | "
                f"{abs(COEFS[f]):.3f} | "
                f"{PERM[f]:.3f} | "
                f"{ABLATION_DROP[f]:+.3f} |\n")
    md += "\nNotes: skill_overlap and weighted_skill_score have near-equal opposite-sign coefficients — multicollinearity signature. years_of_experience contributes essentially nothing.\n"
    (TABLES_DIR / "feature_importance.md").write_text(md, encoding="utf-8")

    # Dataset stats
    md = """# Dataset Specifications

## Raw inputs

| File | Size | Source | Content |
|---|---|---|---|
| Resume (1).csv | 53.7 MB | Kaggle | 2,484 anonymized resumes |
| training_data.csv | 3.6 MB | Public | 853 job descriptions with structured `model_response` |
| Skills.xlsx | 3.2 MB | O*NET (U.S. DoL) | 35 generic competencies with importance scores |
| skills_en.csv | 9.3 MB | ESCO (European Commission) | 13,960 skill entries |
| broaderRelationsSkillPillar_en.csv | 4.9 MB | ESCO | 20,819 hierarchy edges |

## Processed dataset

| Artifact | Rows | Description |
|---|---|---|
| cleaned_resumes.parquet | 2,484 | Cleaned + skill-extracted resumes |
| cleaned_jobs.parquet | 853 | Cleaned + skill-extracted jobs |
| pair_features.parquet | 42,650 | Top-50 SBERT retrieval × 13 features |
| pair_features_diversified.parquet | 50,650 | + 5,000 A_mid + 2,000 B_xcat + 1,000 C_rand |
| gold_labels.csv | 500 | 3-LLM majority-vote labels with metadata |
| active_labels.csv | 200 | Active-learning uncertain-band labels |

## Test set

100 held-out pairs (22 positive, 78 negative), stratified 80/20 split with random_state=42.
Used identically across all six methodology comparisons.
"""
    (TABLES_DIR / "dataset_stats.md").write_text(md, encoding="utf-8")

    # ESCO impact
    md = """# ESCO Vocabulary Integration — Before vs After

| Metric | Before ESCO | After ESCO | Change |
|---|---|---|---|
"""
    for (metric, before, after, unit) in ESCO_IMPACT:
        if unit == "%":
            md += f"| {metric} | {before:.0f}% | {after:.0f}% | +{after - before:.0f} pp |\n"
        elif metric.startswith("title"):
            md += f"| {metric} | {before:,.0f} | {after:,.0f} | +{after - before:,.0f} |\n"
        else:
            md += f"| {metric} | {before:.2f} | {after:.2f} | +{after - before:.2f} |\n"
    md += """

## Vocabulary sizes

| Component | Skills |
|---|---|
| TECH_SKILLS (curated) | 355 |
| ESCO knowledge (preferredLabel) | 3,208 |
| ESCO altLabels added | 16,570 |
| ESCO hiddenLabels added | 190 |
| **Total merged vocabulary** | **20,253** |

Filter applied during ESCO integration: skillType=knowledge, length ≥ 4 chars, ≥ 50% alphabetic, not in 90-word stopword/verb blocklist.
"""
    (TABLES_DIR / "esco_impact.md").write_text(md, encoding="utf-8")

    print(f"[write] {TABLES_DIR}/lit_review_comparison.md")
    print(f"[write] {TABLES_DIR}/performance_metrics.md")
    print(f"[write] {TABLES_DIR}/feature_importance.md")
    print(f"[write] {TABLES_DIR}/dataset_stats.md")
    print(f"[write] {TABLES_DIR}/esco_impact.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(" Generating presentation visuals + markdown tables")
    print("=" * 70)

    print("\n[charts]")
    chart_01_lit_review_table()
    chart_02_main_performance()
    chart_03_full_metrics_table()
    chart_04_methodology_journey()
    chart_05_feature_importance()
    chart_06_sbert_vs_skill()
    chart_07_esco_impact()
    chart_08_per_source_breakdown()
    chart_09_inter_llm_agreement()
    chart_10_confusion_matrix()
    chart_11_pk_comparison()

    print("\n[tables]")
    write_markdown_tables()

    print("\n" + "=" * 70)
    print(f" Done. {len(list(OUT_DIR.glob('*.png')))} PNGs + "
          f"{len(list(TABLES_DIR.glob('*.md')))} markdown tables in:")
    print(f"   {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
