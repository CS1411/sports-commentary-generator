# Sports Commentary Generator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/f59j6rg4a8/badge.svg)](https://ask.thili.ai/b/f59j6rg4a8)

This project provides game developers with a powerful tool to generate dynamic sports commentary. By applying the LoRA fine-tuning technique from the "Build LoRA from Scratch" drill, a pre-trained language model is adapted to produce context-aware commentary for simulated sports events, enhancing game immersion.

## What you are building

**Goal.** You will have built a tool for game developers that uses a LoRA-adapted language model to generate dynamic and context-aware commentary snippets for simulated sports events.

**Done when.** The project is done when you can input event descriptions into a Streamlit interface and the LoRA-adapted model generates relevant and engaging sports commentary, demonstrating the successful fine-tuning and merging of the LoRA adapter.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Create a small dataset of simulated sports events and commentary pairs.**  
      Hint: Dataset creation and formatting.
- [x] **2. Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).**  
      Hint: LoRA adapter injection and training.
- [x] **3. Build a Streamlit interface to input event descriptions and generate commentary.**  
      Hint: Streamlit UI development.
- [x] **4. Test the model's ability to produce relevant and engaging commentary.**  
      Hint: Qualitative model testing.
- [x] **5. Implement adapter merging for inference.**  
      Hint: LoRA adapter merging.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Expand the dataset to include a wider variety of sports events and commentary styles.
- Add options in the Streamlit interface to control the tone or verbosity of the generated commentary.
- Implement a simple quantitative evaluation metric for commentary relevance or engagement.
