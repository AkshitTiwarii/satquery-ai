"""One place that decides how the weights are quantised, for every harness.

Four eval scripts each carried their own copy of this branch, which is how the
precisions drifted apart in the first place: the engine serves nf4 with an fp16
compute dtype, the harnesses use bf16. This module is the single definition,
and satquery/models.py calls it too, so a number scored here is scored on the
weights the demo actually runs.

THE PRECISIONS
--------------
nf4   4-bit, ~2.5 GB resident. What the office RTX 4060 has served since the
      start, and what every recorded number was measured on.
int8  8-bit, ~4 GB. The middle rung. nf4 costs real accuracy - on RSVQA-LR the
      joint adapter reads 84.5 in nf4 and 87.5 in fp16 on the same questions,
      so three points are being paid to quantisation, not to the model. fp16
      does not fit an 8 GB card (7.51 GB of weights, measured, before any
      activations), and int8 is the only rung between them.
fp16  full half precision, 7.51 GB. The Kaggle T4 path. bitsandbytes is not
      installed in the kernels, which is why the import below is lazy: a
      BitsAndBytesConfig constructed where the package metadata is absent
      raises, and that is exactly how the v8 run died after ten hours.

WHAT TO EXPECT OF int8
----------------------
Accuracy should land between nf4 and fp16. Throughput may well be WORSE than
nf4: LLM.int8 splits each matmul into a normal 8-bit part and an fp16 outlier
part, and that second path costs time the 4-bit kernel does not spend. So
measure questions per second alongside the score, and if the demo gets slower
the honest answer is to serve nf4 live and report int8 separately, saying so.
"""

DTYPES = ("nf4", "int8", "fp16")


def load_model(model_path, dtype, device=None, adapter=None):
    """-> a Qwen2.5-VL model on one device, optionally with a LoRA adapter.

    `device` defaults to {"": 0}: pinning to a single card matters because
    Trainer and accelerate both read torch.cuda.device_count(), and Kaggle
    hands out two T4s.
    """
    # Checked before anything heavy is imported, so a typo fails in
    # milliseconds rather than after a model download, and so the check is
    # testable on a machine with no torch.
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, got {dtype!r}")
    if device is None:
        device = {"": 0}

    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration

    kwargs = {"device_map": device}
    if dtype == "fp16":
        kwargs["dtype"] = torch.float16
    else:
        # Lazy, and only on a quantised path. See the module docstring.
        from transformers import BitsAndBytesConfig
        if dtype == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True)
            kwargs["dtype"] = torch.bfloat16
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["dtype"] = torch.float16

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model
