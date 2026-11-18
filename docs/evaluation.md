# Evaluation

Held-out events: 72 (never seen in training). Greedy decoding.

| model | relevance | all facts | non-empty | `!` in excited | `!` in calm |
|---|---|---|---|---|---|
| base distilgpt2 (no LoRA) | 10% | 6% | 100% | 0% | 0% |
| LoRA adapter | 95% | 85% | 100% | 91% | 15% |
| LoRA adapter, merged | 95% | 85% | 100% | 91% | 15% |

Merging folds B@A into the weights. Merged and un-merged commentary is identical for 72 of 72 events.

| model | generation time |
|---|---|
| base distilgpt2 (no LoRA) | 87 s |
| LoRA adapter | 43 s |
| LoRA adapter, merged | 41 s |
