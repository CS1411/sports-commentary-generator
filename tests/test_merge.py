"""Milestone 5: merging the adapter into the weights. Uses a tiny random GPT-2, no downloads."""
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from app.lora import LoRALinear, add_lora, adapter_state, count_params, merge_lora


def tiny_gpt2():
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=120, n_positions=64)).eval()


def trained_like(model):
    "Pretend training happened: move B away from zero so the detour is live."
    torch.manual_seed(1)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.B += 0.2 * torch.randn_like(m.B)
    return model.eval()


def test_merged_model_matches_the_adapter_model():
    model = trained_like(add_lora(tiny_gpt2(), r=4, alpha=8, dropout=0.0))
    ids = torch.tensor([[3, 14, 15, 9, 26]])
    with torch.no_grad():
        adapter_logits = model(ids).logits
    merge_lora(model)
    with torch.no_grad():
        merged_logits = model(ids).logits
    assert (adapter_logits - merged_logits).abs().max() < 1e-4


def test_merge_removes_the_wrappers_and_changes_the_weights():
    plain = tiny_gpt2()
    before = plain.transformer.h[0].attn.c_attn.weight.clone()
    model = merge_lora(trained_like(add_lora(plain, r=4, alpha=8, dropout=0.0)))
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    assert type(model.transformer.h[0].attn.c_attn).__name__ == "Conv1D"
    assert not torch.equal(model.transformer.h[0].attn.c_attn.weight, before)


def test_adapter_survives_a_save_and_load_round_trip():
    trained = trained_like(add_lora(tiny_gpt2(), r=4, alpha=8, dropout=0.0))
    state = adapter_state(trained)
    fresh = add_lora(tiny_gpt2(), r=4, alpha=8, dropout=0.0)
    fresh.load_state_dict(state, strict=False)
    ids = torch.tensor([[7, 8, 9]])
    with torch.no_grad():
        assert torch.allclose(trained(ids).logits, fresh.eval()(ids).logits, atol=1e-6)
    assert count_params(fresh)[0] == sum(v.numel() for v in state.values())
