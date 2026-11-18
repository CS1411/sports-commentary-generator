"""Milestone 2: LoRA injection and training setup. No model downloads: a tiny random GPT-2 is built locally."""
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from app.lora import LoRALinear, add_lora, adapter_state, count_params
from app.train import collate, encode


def tiny_gpt2():
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=120, n_positions=64)).eval()


def test_lora_linear_starts_as_a_no_op_then_comes_alive():
    torch.manual_seed(0)
    base = nn.Linear(16, 48)
    lora = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(3, 16)
    assert torch.allclose(lora(x), base(x))
    with torch.no_grad():
        lora.B += 0.1 * torch.randn_like(lora.B)
    assert not torch.allclose(lora(x), base(x))
    assert not any(p.requires_grad for p in base.parameters())


def test_injection_freezes_everything_but_the_adapters():
    model = tiny_gpt2()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        before = model(ids).logits
    add_lora(model, r=4, alpha=8)
    with torch.no_grad():
        after = model(ids).logits
    assert torch.allclose(before, after, atol=1e-6)
    trainable, total = count_params(model)
    assert trainable == 2 * (4 * 32 + 3 * 32 * 4)  # A and B for each of the 2 blocks
    assert {n.split(".")[-1] for n, p in model.named_parameters() if p.requires_grad} == {"A", "B"}
    assert len(adapter_state(model)) == 4 and trainable < total


def test_a_few_training_steps_reduce_the_loss():
    model = add_lora(tiny_gpt2(), r=4, alpha=8, dropout=0.0).train()
    ids = torch.randint(0, 120, (4, 12))
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    losses = []
    for _ in range(15):
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        optim.step()
        optim.zero_grad()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


class StubTokenizer:
    "Words become ids; enough to test how examples are encoded."
    eos_token_id = 0

    def __call__(self, text):
        return {"input_ids": [hash(w) % 100 + 1 for w in text.split()]}


def test_loss_covers_only_the_commentary():
    ex = {"event": "12' GOAL! A scores.", "tone": "calm", "commentary": "A scores today."}
    ids, labels = encode(StubTokenizer(), ex)
    n_prompt = len(StubTokenizer()("Tone: calm\nEvent: 12' GOAL! A scores.\nCommentary:")["input_ids"])
    assert all(lab == -100 for lab in labels[:n_prompt])
    assert labels[n_prompt:] == ids[n_prompt:] and ids[-1] == 0


def test_collate_pads_to_the_longest():
    ids, mask, labels = collate([([5, 6, 7], [-100, 6, 7]), ([5], [5])], pad_id=0)
    assert ids.shape == (2, 3) and mask.tolist() == [[1, 1, 1], [1, 0, 0]]
    assert labels[1].tolist() == [5, -100, -100]
