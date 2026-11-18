"""Load the base model, attach the trained adapter, and generate commentary."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.dataset import build_prompt
from app.lora import add_lora
from app.train import ADAPTER_PATH, BASE_MODEL


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token
    return tok


def load_base():
    "The untouched pretrained distilgpt2."
    return AutoModelForCausalLM.from_pretrained(BASE_MODEL).eval()


def load_adapted(path=ADAPTER_PATH):
    "A fresh distilgpt2 with the saved LoRA adapter attached (the base weights stay frozen)."
    checkpoint = torch.load(path)
    model = add_lora(load_base(), r=checkpoint["r"], alpha=checkpoint["alpha"])
    result = model.load_state_dict(checkpoint["state"], strict=False)
    assert not result.unexpected_keys, result.unexpected_keys
    return model.eval()


def clean_output(text):
    "Keep the first generated line; the model sometimes starts a new 'Tone:' block after the commentary."
    text = text.strip().split("\n")[0].strip()
    return text.split("Tone:")[0].strip()


@torch.no_grad()
def generate_commentary(model, tok, event, tone="excited", max_new_tokens=40, temperature=0.0, seed=0):
    """Commentary for one event. temperature 0 is greedy (deterministic); above 0 it samples."""
    inputs = tok(build_prompt(event, tone), return_tensors="pt")
    options = {"do_sample": False}
    if temperature > 0:
        torch.manual_seed(seed)
        options = {"do_sample": True, "temperature": temperature, "top_p": 0.9}
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id, **options)
    return clean_output(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
