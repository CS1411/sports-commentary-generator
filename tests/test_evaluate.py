"""Milestone 4: the commentary quality checks. The model is replaced by the dataset's own reference lines."""
from app import evaluate
from app import model as lm
from app.dataset import generate_dataset


def test_relevance_counts_the_facts_mentioned():
    facts = ["Alex Moreau", "Riverside FC", "2-1"]
    assert evaluate.relevance("Alex Moreau scores for Riverside FC, it is 2-1!", facts) == 1.0
    assert evaluate.relevance("alex moreau scores", facts) == 1 / 3
    assert evaluate.relevance("What a game.", facts) == 0.0


def test_summary_of_perfect_and_empty_commentary():
    examples = generate_dataset(60)
    perfect = evaluate.summarize([ex["commentary"] for ex in examples], examples)
    assert perfect["relevance"] == perfect["all_facts"] == perfect["non_empty"] == 1.0
    assert perfect["bang_excited"] > 0.9 and perfect["bang_calm"] == 0.0  # the dataset's tone is learnable
    empty = evaluate.summarize([""] * len(examples), examples)
    assert empty["relevance"] == 0.0 and empty["non_empty"] == 0.0


def test_run_uses_each_examples_event_and_tone(monkeypatch):
    seen = []
    monkeypatch.setattr(lm, "generate_commentary", lambda m, t, event, tone, **kw: seen.append((event, tone)) or "x")
    examples = generate_dataset(30)
    outputs, seconds = evaluate.run("model", "tok", examples)
    assert outputs == ["x"] * len(examples) and seen == [(ex["event"], ex["tone"]) for ex in examples] and seconds >= 0


def test_metrics_table_is_markdown():
    row = {"relevance": 0.5, "all_facts": 0.25, "non_empty": 1.0, "bang_excited": 0.9, "bang_calm": 0.0}
    table = evaluate.metrics_table({"LoRA": row})
    assert table.splitlines()[2] == "| LoRA | 50% | 25% | 100% | 90% | 0% |"
