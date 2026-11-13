import torch

from finetuning.lora_sql.adapters import (
    LoRAConfig,
    adapter_parameters,
    attach_lora,
    load_adapter,
    merge_lora,
    save_adapter,
)
from finetuning.tinysql.model import ModelConfig, TinySQLLM


def tiny():
    torch.manual_seed(1)
    return TinySQLLM(ModelConfig(vocab_size=50, context=40, dim=32, layers=1, heads=4, dropout=0))


def test_identity_initialization():
    m = tiny()
    ids = torch.randint(0, 50, (2, 7))
    a = m(ids)[0].detach().clone()
    attach_lora(m, LoRAConfig(rank=4, alpha=8))
    assert torch.equal(a, m(ids)[0])
    assert len(adapter_parameters(m)) == 8


def test_frozen_base_and_gradient():
    m = tiny()
    ref = {k: p.detach().clone() for k, p in m.named_parameters()}
    attach_lora(m, LoRAConfig(rank=4, alpha=8))
    optim = torch.optim.AdamW(adapter_parameters(m), lr=0.05)
    x = torch.randint(0, 50, (2, 7))
    y = torch.randint(0, 50, (2, 7))
    loss = m(x, y)[1]
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter_parameters(m))
    optim.step()
    for k, p in m.named_parameters():
        if ".original." in k:
            assert torch.equal(p.detach(), ref[k.replace(".original.", ".")])
        elif ".lora_" not in k:
            assert torch.equal(p.detach(), ref[k])


def test_checkpoint_roundtrip(tmp_path):
    m = tiny()
    cfg = LoRAConfig(4, 8, 0.0)
    attach_lora(m, cfg)
    with torch.no_grad():
        for p in adapter_parameters(m):
            p.add_(0.05)
    ids = torch.randint(0, 50, (1, 10))
    before = m(ids)[0].detach()
    save_adapter(m, tmp_path / "a.pt", cfg)
    torch.manual_seed(1)
    other = tiny()
    load_adapter(other, tmp_path / "a.pt")
    assert torch.allclose(before, other(ids)[0], atol=1e-6)


def test_merge_equivalence():
    m = tiny()
    attach_lora(m, LoRAConfig(rank=4, alpha=8))
    with torch.no_grad():
        for p in adapter_parameters(m):
            p.add_(0.02)
    ids = torch.randint(0, 50, (1, 9))
    before = m(ids)[0].detach()
    merge_lora(m)
    after = m(ids)[0].detach()
    assert torch.allclose(before, after, rtol=1e-5, atol=1e-6)


def test_only_lora_tensors_saved(tmp_path):
    m = tiny()
    cfg = LoRAConfig(rank=2)
    attach_lora(m, cfg)
    save_adapter(m, tmp_path / "a.pt", cfg)
    saved = torch.load(tmp_path / "a.pt", weights_only=True)
    assert all("lora_" in k for k in saved["state_dict"])
    assert all("original" not in k for k in saved["state_dict"])
