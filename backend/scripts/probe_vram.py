"""Measure real training VRAM for the QLoRA recipe on this card.

One question: does the notebook's fp16 recipe - sized for a T4's 15.6 GB -
fit on the office 4060's 8.19 GB? A forward+backward at the real settings
answers it; arithmetic does not, because activation memory depends on the
processor's actual visual-token count.

    python scripts/probe_vram.py --model F:\sih\models\Qwen2.5-VL-3B-Instruct
"""
import argparse, sys, time
import numpy as np
import torch


def gb(x):
    return x / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-visual-tokens", type=int, default=256)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--4bit", dest="four_bit", action="store_true",
                    help="NF4 weights. Qwen2.5-VL's 3420 inner dim misses "
                         "bitsandbytes' fast kernel, so the dequant fallback "
                         "runs - slow at inference on this card, and fatal in "
                         "training on a T4. Untested here; that is the point.")
    ap.add_argument("--cpu-vision", action="store_true",
                    help="keep the frozen vision tower on CPU; it is never "
                         "trained (LoRA excludes it) and reBEN patches are "
                         "120x120, so the CPU pass is cheap")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no CUDA")
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory
    print("device      : %s" % props.name)
    print("total VRAM  : %.2f GB" % gb(total))
    print("in use now  : %.2f GB" % gb(torch.cuda.memory_allocated()))

    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    kw = dict(device_map={"": 0}, dtype=torch.float16)
    if args.four_bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16)

    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, **kw)
    print("\nweights     : %.2f GB  (loaded in %.0f s)"
          % (gb(torch.cuda.memory_allocated()), time.time() - t0))

    if args.cpu_vision:
        model.model.visual.to("cpu")
        torch.cuda.empty_cache()
        print("vision->CPU : %.2f GB on GPU now"
              % gb(torch.cuda.memory_allocated()))

    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=64 * 28 * 28,
        max_pixels=args.max_visual_tokens * 28 * 28)

    WANT = ("q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj")
    targets = sorted({n for n, _ in model.named_modules()
                      if n.split(".")[-1] in WANT
                      and not n.startswith("visual") and ".visual." not in n})
    print("LoRA targets: %d  (vision touched: %s)"
          % (len(targets), any("visual" in t for t in targets)))

    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.1, bias="none",
        task_type="CAUSAL_LM", target_modules=targets))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    print("after LoRA  : %.2f GB" % gb(torch.cuda.memory_allocated()))

    # A reBEN patch is 120x120 after the by-name RGB pick, percentile-stretched
    # to uint8. Synthetic pixels are faithful for memory: only the shape counts.
    from PIL import Image
    img = Image.fromarray(
        (np.random.rand(120, 120, 3) * 255).astype(np.uint8))
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text",
                                          "text": "Is there a water body in this image?"}]}]
    prompt = processor.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
    enc = processor(text=[prompt + "yes<|im_end|>"], images=[img],
                    return_tensors="pt", padding=True)
    n_img_tok = int((enc["input_ids"][0] == processor.tokenizer.convert_tokens_to_ids(
        "<|image_pad|>")).sum())
    print("visual tok  : %d   | seq len: %d"
          % (n_img_tok, enc["input_ids"].shape[1]))

    labels = enc["input_ids"].clone()
    pad = processor.tokenizer.pad_token_id
    labels[labels == pad] = -100
    dev_img = "cpu" if args.cpu_vision else "cuda"
    batch = {k: v.to(dev_img if k.startswith("pixel") or k == "image_grid_thw"
                     else "cuda")
             for k, v in enc.items()}
    batch["labels"] = labels.to("cuda")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    for i in range(args.steps):
        out = model(**batch)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        print("step %d      : loss %.4f | peak %.2f GB"
              % (i + 1, out.loss.item(), gb(torch.cuda.max_memory_allocated())))

    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    print("\nPEAK alloc  : %.2f GB" % gb(peak))
    print("PEAK reserv : %.2f GB" % gb(reserved))
    print("headroom    : %.2f GB of %.2f GB" % (gb(total - reserved), gb(total)))
    print("VERDICT     : %s" % ("FITS" if reserved < total * 0.92 else "TOO TIGHT"))


if __name__ == "__main__":
    main()
