# Evaluation

Held-out events: 72 (never seen in training). Greedy decoding.

| model | relevance | all facts | non-empty | `!` in excited | `!` in calm |
|---|---|---|---|---|---|
| base distilgpt2 (no LoRA) | 10% | 6% | 100% | 0% | 0% |
| LoRA adapter | 95% | 85% | 100% | 91% | 15% |

