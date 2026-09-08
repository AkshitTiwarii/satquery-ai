"""A second learning rate for the vision tower has to actually reach it.

The failure this guards is silent: the flag is set, the notebook prints a
multiplier, and the optimizer holds one group at one rate because the name
filter matched nothing. The run then costs four GPU-hours and answers a
question it never asked. So the partition is pinned, the ratio is pinned after
a real AdamW step, and an empty group raises instead of degrading.

Plain torch only - peft is on neither of our machines, so the LoRA parameter
names are reproduced by hand from an actual adapter's state dict.
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from lr_groups import describe, param_groups  # noqa: E402


def _named(vision=True, text=True):
    """(name, parameter) pairs shaped like Qwen2.5-VL + peft produces.

    Real names, from the ground-ref-vis adapter: the language tower is
    `base_model.model.model.layers.<i>.self_attn.q_proj.lora_A.default.weight`
    and the vision tower is
    `base_model.model.visual.blocks.<i>.attn.qkv.lora_A.default.weight`.
    A frozen base weight is included so the filter has something to reject.
    """
    out = []
    if text:
        for i in (0, 17):
            for proj in ("q_proj", "down_proj"):
                for ab in ("lora_A", "lora_B"):
                    p = torch.nn.Parameter(torch.randn(4, 4))
                    out.append((f"base_model.model.model.layers.{i}.self_attn."
                                f"{proj}.{ab}.default.weight", p))
    if vision:
        for i in (0, 31):
            for proj in ("attn.qkv", "mlp.down_proj"):
                for ab in ("lora_A", "lora_B"):
                    p = torch.nn.Parameter(torch.randn(4, 4))
                    out.append((f"base_model.model.visual.blocks.{i}."
                                f"{proj}.{ab}.default.weight", p))
        p = torch.nn.Parameter(torch.randn(4, 4))
        out.append(("base_model.model.visual.merger.mlp.0.lora_A.default.weight", p))
    frozen = torch.nn.Parameter(torch.randn(4, 4), requires_grad=False)
    out.append(("base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight",
                frozen))
    return out


def test_partition_covers_every_lora_tensor_and_nothing_else():
    named = _named()
    groups = param_groups(named, lr=1e-4, mult=3.0)
    got = sum(len(g["params"]) for g in groups)
    expected = sum(1 for n, p in named if p.requires_grad and "lora_" in n)
    assert got == expected == 17
    frozen = [p for n, p in named if not p.requires_grad][0]
    for g in groups:
        assert all(p is not frozen for p in g["params"])


def test_vision_group_gets_the_multiplied_rate():
    groups = param_groups(_named(), lr=1e-4, mult=3.0)
    by = {g["name"]: g for g in groups}
    assert by["text"]["lr"] == pytest.approx(1e-4)
    assert by["vision"]["lr"] == pytest.approx(3e-4)
    assert len(by["vision"]["params"]) == 9   # 2 blocks x 2 proj x 2 + merger
    assert len(by["text"]["params"]) == 8


def test_merger_counts_as_vision():
    """The patch merger is the ViT's last stage. Trained at the language rate
    it would be the one vision module left behind."""
    groups = param_groups(_named(), lr=1e-4, mult=2.0)
    vision = [g for g in groups if g["name"] == "vision"][0]
    assert len(vision["params"]) == 9


def test_an_empty_vision_group_raises():
    """THE CONTROL. Without --vision-lora there are no vision LoRA tensors, so
    a vision learning rate is meaningless and must not pass silently."""
    with pytest.raises(ValueError, match="no trainable vision LoRA"):
        param_groups(_named(vision=False), lr=1e-4, mult=3.0)


def test_an_empty_text_group_raises():
    with pytest.raises(ValueError, match="no trainable text LoRA"):
        param_groups(_named(text=False), lr=1e-4, mult=3.0)


def test_a_nonpositive_multiplier_raises():
    with pytest.raises(ValueError, match="must be positive"):
        param_groups(_named(), lr=1e-4, mult=0.0)


def _step_ratio(mult, steps=1, scheduler=None):
    """One or more real AdamW steps; -> mean |dW| vision / mean |dW| text."""
    named = _named()
    groups = param_groups(named, lr=1e-4, mult=mult)
    opt = torch.optim.AdamW(groups)
    sched = scheduler(opt) if scheduler else None
    before = {id(p): p.detach().clone() for g in groups for p in g["params"]}
    for _ in range(steps):
        opt.zero_grad()
        for g in groups:
            for p in g["params"]:
                p.grad = torch.ones_like(p)
        opt.step()
        if sched:
            sched.step()
    moved = {}
    for g in groups:
        d = [(p.detach() - before[id(p)]).abs().mean().item() for p in g["params"]]
        moved[g["name"]] = sum(d) / len(d)
    return moved["vision"] / moved["text"]


def test_both_groups_move_and_vision_moves_by_the_multiplier():
    """Adam's first update is about lr per element whatever the gradient is,
    so the ratio of the two groups' movement is the multiplier."""
    for mult in (2.0, 3.0):
        assert _step_ratio(mult) == pytest.approx(mult, rel=0.05)


def test_the_single_group_control_fails_the_same_check():
    """If create_optimizer were never overridden - the flag set and ignored -
    both towers would move together. This is what the ratio check catches."""
    named = _named()
    params = [p for n, p in named if p.requires_grad and "lora_" in n]
    opt = torch.optim.AdamW(params, lr=1e-4)
    before = [p.detach().clone() for p in params]
    opt.zero_grad()
    for p in params:
        p.grad = torch.ones_like(p)
    opt.step()
    vision, text = [], []
    for (n, p), b in zip([x for x in named if x[1].requires_grad and "lora_" in x[0]],
                         before):
        (vision if ".visual." in n else text).append(
            (p.detach() - b).abs().mean().item())
    ratio = (sum(vision) / len(vision)) / (sum(text) / len(text))
    assert ratio == pytest.approx(1.0, rel=0.05), "one group means one rate"


def test_cosine_schedule_preserves_the_ratio():
    """get_scheduler reads each group's own initial_lr, so one cosine decays
    both peaks together. If it did not, the multiplier would drift away over
    the run and the last steps would train the towers at the same rate."""
    from torch.optim.lr_scheduler import LambdaLR

    def cosine(opt):
        import math
        return LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / 100)))

    for steps in (1, 30, 60):
        assert _step_ratio(3.0, steps=steps, scheduler=cosine) == pytest.approx(3.0, rel=0.1)


def test_describe_names_both_groups():
    text = describe(param_groups(_named(), lr=1e-4, mult=3.0))
    assert "text" in text and "vision" in text and "3.00e-04" in text
