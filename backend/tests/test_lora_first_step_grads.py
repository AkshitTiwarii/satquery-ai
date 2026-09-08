"""Why exactly half the LoRA tensors have a zero gradient on step one.

The Kaggle run stopped at our own guard:

    only 252 of 504 LoRA tensors got a gradient

504 is 252 target modules times two tensors each, lora_A and lora_B, so
exactly one of every pair was dead. That is not a broken model, it is what
LoRA initialisation MEANS, and the guard was the thing that was wrong.

peft initialises lora_A from a Kaiming-uniform draw and lora_B to ZEROS, so
the adapter contributes nothing at step 0 and the fine-tune starts exactly at
the base model. The consequence is arithmetic:

    out = base(x) + B @ (A @ x) * s

    dL/dB = g @ (A @ x).T          -> nonzero, A @ x is a real vector
    dL/dA = (B.T @ g) @ x.T * s    -> B is all zeros, so this is EXACTLY zero

After the first optimiser step B is nonzero and A starts learning. So on the
first backward, and only the first, half the tensors legitimately carry a
zero gradient.

The distinction the guard has to make is therefore between:

    grad is None       - backward never reached this tensor. Real bug.
    grad is all zeros  - reached it; at step 0 for lora_A this is correct.

This test pins that down in plain torch, because peft is not installed on
either of our machines and the claim is about autograd, not about peft.
"""
import torch


def _lora_pair(d=16, r=4, seed=0):
    """A LoRA-shaped adapter with peft's initialisation: A random, B zeros."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(d, d, generator=g)
    a = torch.randn(r, d, generator=g) * 0.02
    b = torch.zeros(d, r)
    a.requires_grad_(True)
    b.requires_grad_(True)
    return base, a, b


def _step(base, a, b, x, target, scaling=2.0):
    out = x @ base.T + (x @ a.T) @ b.T * scaling
    loss = ((out - target) ** 2).mean()
    loss.backward()
    return loss


def test_lora_A_grad_is_exactly_zero_on_the_first_backward():
    base, a, b = _lora_pair()
    x = torch.randn(8, 16)
    target = torch.randn(8, 16)

    _step(base, a, b, x, target)

    # Both were REACHED - that is the property a real plumbing check wants.
    assert a.grad is not None, "backward never reached lora_A"
    assert b.grad is not None, "backward never reached lora_B"

    # And exactly one of them is zero, because B started at zeros.
    assert a.grad.abs().sum() == 0, (
        "lora_A should have an exactly-zero gradient while lora_B is zeros")
    assert b.grad.abs().sum() > 0, "lora_B must carry gradient from step one"


def test_lora_A_starts_learning_once_B_has_moved():
    """The zero is a step-0 property, not a permanent one.

    If it survived an optimiser step the adapter really would be half dead,
    and the guard would have been right for the wrong reason.
    """
    base, a, b = _lora_pair()
    x = torch.randn(8, 16)
    target = torch.randn(8, 16)
    opt = torch.optim.SGD([a, b], lr=0.1)

    _step(base, a, b, x, target)
    opt.step()
    opt.zero_grad(set_to_none=True)

    assert b.abs().sum() > 0, "lora_B did not move on the first step"

    _step(base, a, b, x, target)
    assert a.grad.abs().sum() > 0, (
        "lora_A still has no gradient after B moved - this WOULD be the bug "
        "the guard was written to catch")


def test_a_genuinely_detached_adapter_is_still_caught():
    """The control. A guard that cannot go red proves nothing.

    Detach the adapter branch the way a wrong module list or a frozen path
    would, and the plumbing check must fail - grad is None, not zero.

    The base has to stay differentiable here. Detaching the adapter when it is
    the ONLY thing on the graph makes backward raise "does not require grad"
    before it can measure anything, which is a broken control rather than a
    red one - it went red for the wrong reason on the first draft.
    """
    base, a, b = _lora_pair()
    base.requires_grad_(True)
    x = torch.randn(8, 16)
    target = torch.randn(8, 16)

    out = x @ base.T + ((x @ a.T) @ b.T).detach()
    ((out - target) ** 2).mean().backward()

    assert base.grad is not None, "the control lost its differentiable path"

    assert a.grad is None and b.grad is None, (
        "the detached control still received gradients - this test is not "
        "measuring what it claims to")
