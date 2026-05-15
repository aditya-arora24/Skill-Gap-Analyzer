# Performance Metrics — All Six Approaches

Evaluated on the same 100-pair held-out gold standard (22 positive / 78 negative).
LLM consensus labels (Claude + GPT + Gemini majority vote).

| Approach | Threshold | Accuracy | Precision | Recall | F1 | P@5 | P@10 | P@20 |
|---|---|---|---|---|---|---|---|---|
| **Active learning (final)** | — | **0.88** | **0.667** | **0.909** | **0.769** | **1.00** | **0.90** | **0.70** |
| LLM supervised | — | 0.84 | 0.594 | 0.864 | 0.704 | 1.00 | 0.80 | 0.70 |
| SBERT only | — | 0.79 | 0.516 | 0.727 | 0.604 | 1.00 | 0.70 | 0.65 |
| Self-trained v2 | — | 0.84 | 0.688 | 0.500 | 0.579 | 1.00 | 0.90 | 0.70 |
| Self-trained v1 | — | 0.83 | 0.647 | 0.500 | 0.564 | 1.00 | 0.80 | 0.70 |
| Weak supervised | — | 0.72 | 0.393 | 0.500 | 0.440 | 0.40 | 0.60 | 0.40 |

**Final model in bold.** Lift of active learning over SBERT-only baseline: +0.165 F1. Over plain supervised: +0.066 F1.
