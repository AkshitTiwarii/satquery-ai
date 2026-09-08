"""Two learning rates over one LoRA: the vision tower separately from the text.

WHY
---
`--vision-lora` puts LoRA on the 32 ViT blocks and the patch merger as well as
the language projections, and on grounding it moved the referring form from
20.5 to 33.0 on data that had already been tried without it. But both towers
then train at the single `learning_rate` in TrainingArguments, and there is a
reason to think the vision side wants more of it: Qwen2.5-VL ships its ViT
frozen through the model's own SFT and DPO, so those adapters start from a
tower that has never been asked to represent 10-metre multispectral imagery,
while the language side is only being nudged toward an answer format it
half-knows already.

HOW
---
`param_groups` splits the trainable LoRA parameters in two by name and gives
the vision group `lr * mult`. Handing those groups to the optimizer is enough:
transformers' `get_scheduler` reads each group's own `initial_lr`, so one
cosine schedule decays both groups from their own peaks and the ratio holds
from the first step to the last. There is no second scheduler and no manual
stepping.

WHAT WILL BITE
--------------
1. A resume leg must be generated with the same multiplier. The optimizer
   state dict records its group count, so resuming a two-group run into a
   one-group optimizer fails to load - loudly, which is the good case.
2. Only LoRA parameters are collected. Everything else is frozen, and quietly
   sweeping a frozen base weight into a param group would hand it to AdamW.
3. An empty group is always a bug, never a no-op: it means the name filter
   stopped matching after a peft or transformers upgrade, and the run would
   silently train one tower. Both cases raise.
"""

VISION_MARKER = ".visual."
LORA_MARKER = "lora_"


def is_lora(name, param):
    return getattr(param, "requires_grad", False) and LORA_MARKER in name


def is_vision(name):
    return VISION_MARKER in name


def param_groups(named_parameters, lr, mult):
    """[{params, lr, name}, ...] - the text group at `lr`, vision at `lr*mult`.

    `named_parameters` is an iterable of (name, parameter), i.e. exactly what
    `model.named_parameters()` yields. Raises if either group comes out empty.
    """
    if mult <= 0:
        raise ValueError(f"vision lr multiplier must be positive, got {mult}")
    text, vision = [], []
    for name, param in named_parameters:
        if not is_lora(name, param):
            continue
        (vision if is_vision(name) else text).append(param)
    if not text:
        raise ValueError(
            "no trainable text LoRA parameters matched - the name filter "
            f"({LORA_MARKER!r}) found nothing outside the vision tower")
    if not vision:
        raise ValueError(
            "no trainable vision LoRA parameters matched "
            f"({VISION_MARKER!r} + {LORA_MARKER!r}). A vision learning rate "
            "only means something with --vision-lora, and an empty group here "
            "means the run would train the language side alone")
    return [
        {"params": text, "lr": lr, "name": "text"},
        {"params": vision, "lr": lr * mult, "name": "vision"},
    ]


def describe(groups):
    return " | ".join(
        f"{g['name']}: {len(g['params'])} tensors, "
        f"{sum(p.numel() for p in g['params']):,} params, lr {g['lr']:.2e}"
        for g in groups)
