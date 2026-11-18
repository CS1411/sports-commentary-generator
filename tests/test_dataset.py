"""Milestone 1: the simulated event/commentary dataset."""
from app.dataset import (EVENT_TYPES, TONES, build_prompt, generate_dataset, load_dataset, split_dataset,
                         training_text)


def test_every_commentary_mentions_the_facts_of_its_event():
    for ex in generate_dataset(300):
        for fact in ex["facts"]:
            assert fact in ex["event"], ex
            assert fact in ex["commentary"], ex


def test_dataset_is_reproducible_distinct_and_covers_everything():
    data = generate_dataset(200, seed=1)
    assert data == generate_dataset(200, seed=1)
    assert len({(e["event"], e["tone"]) for e in data}) == 200
    assert {e["type"] for e in data} == set(EVENT_TYPES)
    assert {e["tone"] for e in data} == set(TONES)
    assert {e["sport"] for e in data} == {"football", "basketball", "cricket"}


def test_split_and_format():
    train, test = split_dataset(generate_dataset(100), test_frac=0.2)
    assert len(train) == 80 and len(test) == 20
    ex = train[0]
    prompt = build_prompt(ex["event"], ex["tone"])
    assert prompt.startswith("Tone: ") and prompt.endswith("Commentary:")
    assert training_text(ex).startswith(prompt) and ex["commentary"] in training_text(ex)


def test_committed_dataset_file_is_valid():
    data = load_dataset()
    assert len(data) >= 400
    assert all({"event", "commentary", "tone", "facts"} <= set(e) for e in data)
