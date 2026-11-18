"""Milestone 4: test whether the commentary is relevant and has the right tone. `python -m app.evaluate`.

Quantitative checks, run on held-out events the adapter never trained on:
- relevance: share of the event's key facts (player, team, score...) that the commentary mentions
- all facts: share of commentaries that mention every key fact
- tone: how often "!" appears in excited commentary versus calm commentary
Qualitative check: side-by-side examples of the base model, the LoRA model and the reference line.
"""
import time
from pathlib import Path

from app import model as lm
from app.dataset import TONES, load_dataset, split_dataset

DOCS = Path(__file__).resolve().parent.parent / "docs"


def relevance(commentary, facts):
    "Fraction of the event's key facts that the commentary mentions (case-insensitive)."
    text = commentary.lower()
    return sum(fact.lower() in text for fact in facts) / len(facts)


def tone_rate(outputs, examples, tone):
    "Share of `tone` commentaries that contain an exclamation mark."
    marks = ["!" in out for out, ex in zip(outputs, examples) if ex["tone"] == tone]
    return sum(marks) / len(marks) if marks else 0.0


def summarize(outputs, examples):
    scores = [relevance(out, ex["facts"]) for out, ex in zip(outputs, examples)]
    return {
        "relevance": sum(scores) / len(scores),
        "all_facts": sum(s == 1 for s in scores) / len(scores),
        "non_empty": sum(bool(out.strip()) for out in outputs) / len(outputs),
        "bang_excited": tone_rate(outputs, examples, "excited"),
        "bang_calm": tone_rate(outputs, examples, "calm"),
    }


def run(model, tok, examples, **options):
    "Greedy commentary for every example (deterministic). Returns (outputs, seconds)."
    start = time.time()
    outputs = [lm.generate_commentary(model, tok, ex["event"], ex["tone"], **options) for ex in examples]
    return outputs, time.time() - start


def metrics_table(rows):
    "Markdown table. rows: {name: summary dict}."
    lines = ["| model | relevance | all facts | non-empty | `!` in excited | `!` in calm |",
             "|---|---|---|---|---|---|"]
    for name, s in rows.items():
        lines.append(f"| {name} | {s['relevance']:.0%} | {s['all_facts']:.0%} | {s['non_empty']:.0%} | "
                     f"{s['bang_excited']:.0%} | {s['bang_calm']:.0%} |")
    return "\n".join(lines)


def write_reports(rows, examples, outputs, extra=""):
    DOCS.mkdir(exist_ok=True)
    (DOCS / "evaluation.md").write_text(
        f"# Evaluation\n\nHeld-out events: {len(examples)} (never seen in training). Greedy decoding.\n\n"
        f"{metrics_table(rows)}\n\n{extra}", encoding="utf-8")
    lines = ["# Sample commentary\n", "Held-out events. Reference is the line the dataset pairs with the event.\n"]
    for i in range(0, 8):
        ex = examples[i]
        lines += [f"### {i + 1}. {ex['sport']}, {ex['tone']}", f"**Event:** {ex['event']}", ""]
        for name, outs in outputs.items():
            lines.append(f"- **{name}:** {outs[i] or '(empty)'}")
        lines += [f"- **Reference:** {ex['commentary']}", ""]
    (DOCS / "sample_commentary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    examples = split_dataset(load_dataset())[1]
    tok = lm.load_tokenizer()
    models = {"base distilgpt2 (no LoRA)": lm.load_base(), "LoRA adapter": lm.load_adapted()}
    rows, outputs = {}, {}
    for name, model in models.items():
        outputs[name], seconds = run(model, tok, examples)
        rows[name] = summarize(outputs[name], examples)
        print(f"{name}: {seconds:.0f} s for {len(examples)} events")
    print(metrics_table(rows))
    write_reports(rows, examples, outputs)
    print("\nsame event, both tones:")
    event = examples[0]["event"]
    for tone in TONES:
        print(f"  {tone}: {lm.generate_commentary(models['LoRA adapter'], tok, event, tone)}")


if __name__ == "__main__":
    main()
