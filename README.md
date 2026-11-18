# Sports Commentary Generator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/f59j6rg4a8/badge.svg)](https://ask.thili.ai/b/f59j6rg4a8)
[![tests](https://github.com/CS1411/sports-commentary-generator/actions/workflows/ci.yml/badge.svg)](https://github.com/CS1411/sports-commentary-generator/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

A Streamlit tool for game developers. Describe a simulated sports event and a GPT-2 model, adapted with **LoRA written from scratch** (no `peft`), writes the commentary line, in an excited or a calm tone.

```text
Tone: excited
Event: over 14.4 CENTURY: Emil Dubois of Highland Lions reaches 100 against Desert Eagles.
Commentary: Emil Dubois reaches the milestone for Highland Lions against Desert Eagles!
```

The base `distilgpt2` given the same prompt rambles about something unrelated. The adapter is **0.18% of the model's parameters** (147,456 of 82M) and ships as a **595 KB** file.

## Contents

- [Quick start](#quick-start)
- [Using the app](#using-the-app)
- [How it works](#how-it-works)
- [Results](#results)
- [Reproduce it](#reproduce-it)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Docker](#docker)
- [Limitations](#limitations)
- [License](#license)

## Quick start

Requires Python 3.12 and git. The first run downloads `distilgpt2` (about 350 MB) from the Hugging Face Hub.

```bash
git clone https://github.com/CS1411/sports-commentary-generator.git
cd sports-commentary-generator
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/main.py
```

The trained adapter is already in the repo (`adapter/commentary-lora.pt`), so there is nothing to train first. No API keys are needed and it runs on a CPU.

## Using the app

1. Pick one of the held-out example events, or write your own in the same style.
2. Choose a **tone** (excited or calm), the maximum length, and the creativity (0 always gives the same line).
3. Press **Generate commentary**.

Other options: **Merge the adapter into the weights** (on by default, same commentary but no extra computation) and **Also show the base model without LoRA** to compare.

Events follow the dataset's format: a time or clock, a type in capitals, then who did what, for example `34' GOAL! Alex Moreau scores for Riverside FC against Harbor United. It is now 2-1.` The model knows 12 event types across football, basketball and cricket.

## How it works

1. **Dataset** (`app/dataset.py`): 480 simulated events paired with commentary, generated from templates (football, basketball, cricket; 12 event types; two tones). Every commentary mentions the facts of its event. The data is synthetic, so no real players or matches are involved. A `Tone:` line in the prompt selects the style.
2. **LoRA** (`app/lora.py`): each frozen weight `W` gets a trainable low-rank detour, so the layer computes `W x + (alpha / r) * B A x`. `A` starts small and random and `B` starts at zero, so the model begins exactly as the pretrained one. The adapter wraps GPT-2's fused attention projection (`c_attn`, which holds Q, K and V) in all 6 blocks, with rank 8 and alpha 16.
3. **Training** (`app/train.py`): the base model is frozen and only the adapter trains, on the 408 training events, with the loss on the commentary tokens only. 6 epochs, learning rate 1e-3, about 5 minutes on a CPU. See `docs/training_log.txt`.
4. **Inference** (`app/model.py`): load the base, attach the adapter and generate from the prompt. `merge=True` folds the update into the weights (`W + (alpha / r) B A`), leaving a plain GPT-2 with no adapter code.
5. **Evaluation** (`app/evaluate.py`): relevance and tone checks on 72 held-out events, with side-by-side samples.

## Results

Measured on 72 events held out from training, with greedy decoding (`python -m app.evaluate`; details in [`docs/evaluation.md`](docs/evaluation.md)).

| model | relevance | all facts | non-empty | `!` in excited | `!` in calm |
|---|---|---|---|---|---|
| base distilgpt2 (no LoRA) | 10% | 6% | 100% | 0% | 0% |
| LoRA adapter | 95% | 85% | 100% | 91% | 15% |
| LoRA adapter, merged | 95% | 85% | 100% | 91% | 15% |

- **Relevance** is the share of the event's key facts (player, team, score and so on) that appear in the commentary. **All facts** is the share of commentaries that mention every one.
- The base model almost never mentions the event; the adapter does, and it follows the requested tone: excited lines carry an exclamation mark 91% of the time, calm lines 15%.
- **Merging is exact.** Merged and un-merged commentary is identical on all 72 events, and the logits agree to float32 rounding in the tests.
- Training loss fell from 2.39 to 0.75 over 6 epochs.

Side-by-side examples are in [`docs/sample_commentary.md`](docs/sample_commentary.md). Typical output:

```text
Event:  Q3 02:23 DUNK: Nico Castillo throws it down for Metro Hawks against Bay Wolves.   (calm)
Base:   I think the game is going to be a little bit different. I think the game is ...
LoRA:   Nico Castillo slams it home for Metro Hawks for Metro Hawks against Bay Wolves.
```

## Reproduce it

```bash
python -m app.dataset     # regenerate dataset/commentary.jsonl
python -m app.train       # train the adapter, writes adapter/commentary-lora.pt and docs/training_log.txt
python -m app.evaluate    # writes docs/evaluation.md and docs/sample_commentary.md
```

Everything is seeded, so the dataset and the training run are repeatable on the same library versions (`requirements.txt` pins them). The commentary is deterministic at creativity 0.

## Project structure

```text
app/
  dataset.py    templates, dataset generation, prompt format
  lora.py       LoRALinear, add_lora, merge_lora
  train.py      adapter training and saving
  model.py      loading, adapter attach/merge, generation
  evaluate.py   relevance and tone metrics, reports
  main.py       Streamlit interface
adapter/        trained adapter (595 KB)
dataset/        generated events and commentary
docs/           training log, evaluation and sample commentary
tests/          one test file per milestone
Dockerfile      runs the Streamlit app
.github/workflows/ci.yml   runs the tests on every push
```

## Testing

```bash
pytest
```

21 tests cover the dataset (every commentary mentions its event's facts), LoRA (starts as a no-op, only the adapter trains, loss falls, merging matches), the Streamlit page and the evaluation metrics. They use a tiny randomly initialised GPT-2, so no model is downloaded, and they run on GitHub for every push.

## Docker

```bash
docker build -t sports-commentary .
docker run -p 8080:8080 sports-commentary
```

Then open http://localhost:8080. The image downloads `distilgpt2` on first use.

## Milestones

- [x] **1. Dataset of simulated sports events and commentary pairs:** `app/dataset.py`, `dataset/commentary.jsonl`
- [x] **2. LoRA adapter injection and training on a pre-trained LLM:** `app/lora.py`, `app/train.py`, `adapter/`
- [x] **3. Streamlit interface to input events and generate commentary:** `app/main.py`, `app/model.py`
- [x] **4. Testing that the commentary is relevant and engaging:** `app/evaluate.py`, `docs/`
- [x] **5. Adapter merging for inference:** `merge_lora` in `app/lora.py`, the merge option in the app

Stretch ideas covered: a tone control in the interface, and a quantitative relevance and tone metric. Not built: a wider range of sports and styles.

## Limitations

- **Synthetic, templated data.** The model learns the dataset's phrasing. Held-out events use new names and scores but the same event types, so the results above are optimistic. Events in very different styles will not get good commentary.
- **It can get a fact wrong.** Relevance is 95%, not 100%: for example, it once described a made three-pointer as a miss. The metrics only check that names and scores appear, not that the described action is right. Read the output before using it in a game.
- **Repetition and filler.** Some lines repeat a phrase ("for Metro Hawks for Metro Hawks") or reuse a stock sentence.
- **Tone is imperfect for calm commentary.** 15% of calm lines still end with an exclamation mark.
- **Small model, small data.** `distilgpt2` with 408 training events. It is meant to show LoRA, not to produce broadcast-quality commentary.
- The relevance check is string matching, a proxy rather than a human judgement of how engaging the commentary is.

## License

MIT. See [LICENSE](LICENSE).
