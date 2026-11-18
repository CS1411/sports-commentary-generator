"""Milestone 2: inject LoRA into distilgpt2 and train only the adapter. `python -m app.train`."""
import argparse
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.dataset import build_prompt, load_dataset, split_dataset, training_text
from app.lora import add_lora, adapter_state, count_params

BASE_MODEL = "distilgpt2"
ROOT = Path(__file__).resolve().parent.parent
ADAPTER_PATH = ROOT / "adapter" / "commentary-lora.pt"
LOG_PATH = ROOT / "docs" / "training_log.txt"


def encode(tok, ex, max_len=96):
    "Token ids and labels for one example. The loss covers only the commentary, never the prompt."
    prompt_ids = tok(build_prompt(ex["event"], ex["tone"]))["input_ids"]
    ids = (tok(training_text(ex, eos=""))["input_ids"] + [tok.eos_token_id])[:max_len]
    labels = ([-100] * len(prompt_ids) + ids[len(prompt_ids):])[:len(ids)]
    return ids, labels


def collate(batch, pad_id):
    "Right-pad a list of (ids, labels) to the longest in the batch."
    n = max(len(ids) for ids, _ in batch)
    ids = torch.tensor([i + [pad_id] * (n - len(i)) for i, _ in batch])
    labels = torch.tensor([lab + [-100] * (n - len(lab)) for _, lab in batch])
    mask = torch.tensor([[1] * len(i) + [0] * (n - len(i)) for i, _ in batch])
    return ids, mask, labels


def train(epochs=6, lr=1e-3, batch_size=8, r=8, alpha=16, seed=0, log=print):
    "Train only A and B on the training split; returns (model, tokenizer, loss per epoch)."
    torch.manual_seed(seed)
    rng = random.Random(seed)
    train_set, _ = split_dataset(load_dataset())
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    add_lora(model, r=r, alpha=alpha)
    trainable, total = count_params(model)
    log(f"trainable {trainable:,} / {total:,} parameters = {100 * trainable / total:.2f}%")

    data = [encode(tok, ex) for ex in train_set]
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    epoch_losses = []
    for epoch in range(1, epochs + 1):
        rng.shuffle(data)
        total_loss, steps = 0.0, 0
        for i in range(0, len(data), batch_size):
            ids, mask, labels = collate(data[i:i + batch_size], tok.pad_token_id)
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            optim.step()
            optim.zero_grad()
            total_loss += loss.item()
            steps += 1
        epoch_losses.append(total_loss / steps)
        log(f"epoch {epoch}/{epochs}  loss {epoch_losses[-1]:.4f}")
    return model.eval(), tok, epoch_losses, (r, alpha)


def save_adapter(model, config, path=ADAPTER_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"r": config[0], "alpha": config[1], "state": adapter_state(model)}, path)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the LoRA commentary adapter")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--rank", type=int, default=8)
    args = ap.parse_args()
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    start = time.time()
    model, tok, losses, cfg = train(epochs=args.epochs, r=args.rank, alpha=2 * args.rank, log=log)
    path = save_adapter(model, cfg)
    log(f"saved {path.name}: {path.stat().st_size:,} bytes, trained in {time.time() - start:.0f} s")
    LOG_PATH.parent.mkdir(exist_ok=True)
    LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
