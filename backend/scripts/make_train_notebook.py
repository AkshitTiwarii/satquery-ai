"""Build the QLoRA training notebook that survives Kaggle's 12-hour wall.

    python scripts/make_train_notebook.py   ->  train/qlora-kaggle.ipynb

WHY THIS IS NOT JUST "THE TRAINING SCRIPT IN A NOTEBOOK"
--------------------------------------------------------
Kaggle kills a session at 12 hours. A T4 is roughly half a 4060's fp16
throughput, so our run lands near that wall rather than comfortably inside it.
An earlier QLoRA project of ours learned the shape of this the expensive way:
part 1 stopped itself at 9.50 h at step 614 of 950 and part 2 resumed from the
checkpoint with optimizer and scheduler intact. That worked because the stop
was DELIBERATE.

So three things are built in from the first run, not added after one dies:

  1. TIMESTOP. The trainer stops itself with time to spare, saves, and exits
     cleanly. A run killed by Kaggle mid-write loses the checkpoint too.
  2. RESUME. It looks for a checkpoint dataset at /kaggle/input and continues
     from it. Optimizer and scheduler state included - restarting the LR
     schedule from zero on a resume quietly ruins the run.
  3. CHECKPOINT OFTEN. Cheap insurance against the session dying for any other
     reason.

THE TRAP THAT MATTERS MOST
--------------------------
An LMDB value is a SAFETENSORS DICT keyed by band name, not a stacked cube -
measured, 160,832 bytes per entry. reBEN stores Sentinel-2 at native
resolution, so the bands are different sizes: four 120x120 (10 m) and six
60x60 (20 m). That is TEN bands, which settles the 12-vs-10 contradiction in
our own runbook.

So the loader picks RGB BY NAME (B04/B03/B02) and never by index - an index
into a differently-ordered set is silently the wrong colour channel. It then
stretches by percentile and casts to 8-bit, because Qwen needs 3-channel RGB.
Get any of that wrong and nothing raises: you train on black or false-colour
images and find out days later when the eval sits at chance. DOFA wants the
opposite (all bands, raw, unstretched). Two loaders, one database.
"""

import argparse
import sys
import base64
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "train")
OUT = os.path.join(OUT_DIR, "qlora-kaggle.ipynb")

EMBED = ["scripts/make_subset.py", "scripts/make_rsvqa_subset.py", "scripts/eval_rsvqa_lr.py",
         "scripts/make_cdvqa_subset.py", "scripts/eval_cdvqa.py",
         "scripts/eval_ground.py", "scripts/ben_images.py", "scripts/eval_ben_vqa.py",
         "scripts/lr_groups.py", "scripts/qwen_loader.py",
         "scripts/eval_vrsbench.py", "scripts/get_vrsbench.py", "scripts/make_vrsbench_subset.py"]


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": list(lines)}


def _payload():
    blob = {}
    for rel in EMBED:
        with open(os.path.join(HERE, rel), "rb") as fh:
            blob[rel] = base64.b64encode(fh.read()).decode("ascii")
    return base64.b64encode(json.dumps(blob).encode("utf-8")).decode("ascii")


_CDVQA_EVAL = None
_GROUND_EVAL = None
_VRSBENCH_EVAL = None
_JOINT_EVAL = None
_FUSION_EVAL = None


# Which harness answers for which training variant. An eval-only kernel
# runs exactly the harnesses the trained adapter is scored on, at full
# split, so the number needs no caveat about sampling.
EVAL_HARNESS = {
    "ben":        [("eval_ben_vqa.py", "", "ben_vqa")],
    "rsvqa":      [("eval_rsvqa_lr.py", "", "rsvqa_lr")],
    "joint":      [("eval_rsvqa_lr.py", "", "rsvqa_lr"),
                   ("eval_ben_vqa.py", "", "ben_vqa")],
    "cdvqa":      [("eval_cdvqa.py", "", "cdvqa")],
    "ground":     [("eval_ground.py", "", "ground")],
    "ground-ref": [("eval_ground.py", "", "ground")],
    "fusion":     [("eval_ben_vqa.py", " --sar", "fusion")],
    # VRSBench lives under /kaggle/tmp: 12.4 GB of zips would blow the 20 GB output
    # quota of /kaggle/working and be saved as kernel output for nothing.
    "vrsbench-vqa": [("eval_vrsbench.py",
                      " --task vqa --ann-dir %s --images %s/Images_val.zip --model Qwen/Qwen2.5-VL-3B-Instruct --max-pixels 400" % ("%s" % "/kaggle/tmp/vrsbench", "%s" % "/kaggle/tmp/vrsbench"),
                      "vrsbench_vqa")],
    "vrsbench-ref": [("eval_vrsbench.py",
                      " --task refer --answer-format vrsbench --ann-dir %s --images %s/Images_val.zip --model Qwen/Qwen2.5-VL-3B-Instruct --max-pixels 400" % ("%s" % "/kaggle/tmp/vrsbench", "%s" % "/kaggle/tmp/vrsbench"),
                      "vrsbench_ref")],
}


def eval_only_cells(cells, data, adapter_kernel, limit, dtype, count_eval):
    """Keep setup and data, drop LoRA and training, score a mounted adapter.

    Cells 0-3 are the title, the preflight, the embedded scripts and the
    data download - all of which an eval needs unchanged. Cells 4 to 7 cut
    a TRAINING subset, build a training loader, attach LoRA and train; none
    of that applies. The adapter arrives already trained, mounted read-only
    from the producing kernel's output.
    """
    runs = EVAL_HARNESS.get(data)
    if not runs:
        raise SystemExit("--eval-only has no harness mapped for --data %s" % data)
    slug = adapter_kernel.split("/")[-1]
    src = []
    src.append("# ---- score a trained adapter. No training in this kernel. ----\n")
    src.append("#\n")
    src.append("# The adapter is mounted from %s's output. Kaggle puts a\n" % adapter_kernel)
    src.append("# kernel's output under /kaggle/input/<slug>/, read-only, so nothing\n")
    src.append("# here can modify the weights being measured.\n")
    src.append("import glob, os\n")
    src.append("cands = sorted(glob.glob('/kaggle/input/*/lora_adapter'))\n")
    src.append("assert cands, ('no lora_adapter under /kaggle/input - is %s\\n'\n"
               % adapter_kernel)
    src.append("                'attached as a kernel source, and did it finish?')\n")
    src.append("ADAPTER = cands[0]\n")
    src.append("assert os.path.isfile(ADAPTER + '/adapter_config.json'), \\\n")
    src.append("    ADAPTER + ' has no adapter_config.json'\n")
    src.append("print('scoring', ADAPTER)\n")
    src.append("import json\n")
    src.append("cfg = json.load(open(ADAPTER + '/adapter_config.json'))\n")
    src.append("# Says in the log whether the vision tower was trained, so a result\n")
    src.append("# can never be attributed to the wrong recipe later.\n")
    src.append("tm = cfg.get('target_modules') or []\n")
    src.append("print('rank', cfg.get('r'), '| alpha', cfg.get('lora_alpha'),\n")
    src.append("      '| vision tower trained:', any('visual' in str(t) or t in ('qkv',)\n")
    src.append("                                     for t in tm))\n")
    src.append("\n")
    for script, extra, tag in runs:
        flags = extra
        if script == "eval_rsvqa_lr.py":
            flags += count_eval
        src.append("!python /kaggle/working/src/scripts/%s%s \\\n" % (script, flags))
        src.append("    --adapter {ADAPTER} --dtype %s --limit %d \\\n" % (dtype, limit))
        src.append("    --out /kaggle/working/score_%s\n" % tag)
    src.append("\n")
    src.append("for tag in %r:\n" % [t for _, _, t in runs])
    src.append("    d = json.load(open('/kaggle/working/score_%s.json' % tag))\n")
    src.append("    print()\n")
    src.append("    print('===', tag, 'n =', d.get('n'))\n")
    src.append("    for k in ('overall_accuracy', 'overall_accuracy_binned',\n")
    src.append("              'majority_baseline', 'majority_baseline_binned',\n")
    src.append("              'overall_accuracy_natural', 'acc50', 'miou'):\n")
    src.append("        if k in d:\n")
    src.append("            print('   %-28s %s' % (k, d[k]))\n")
    head = md(
        "# SatQuery - score an adapter, no training\n",
        "\n",
        "Measuring is the bottleneck now, not training: one evaluation card and a\n",
        "full split takes hours on it. This kernel trains nothing. It mounts the\n",
        "adapter from `%s`, downloads the same data the training kernel used, and\n" % adapter_kernel,
        "scores it over %s.\n" % ("the FULL test split" if limit == 0 else "%d rows" % limit),
        "\n",
        "**Accelerator = GPU T4 x2, Internet = On.** A CLI push resets it to P100 and\n",
        "4-bit will not load there.\n")
    # The preflight cell announces a training budget. Nothing here trains,
    # and a log that says '10.5 h of training' in a scoring kernel is the
    # kind of stray line someone later reads as evidence of what ran.
    setup = [dict(c) for c in cells[1:4]]
    setup[0]["source"] = [
        line.replace(
            "OK. Budget: %.1f h of training, then a clean stop.' % TRAIN_HOURS",
            "OK. Scoring only - this kernel trains nothing.'")
        for line in setup[0]["source"]]
    return [head] + setup + [code(*src)]

# The Lithuania/Summer TRAIN cell holds exactly this many reference rows - measured on the
# whole parquet 8 Sep (3,107 patches carry boxes, 10,986 reference + 11,404 point). Both
# grounding runs that asked for "16,000" and "40,000" reference rows trained on this number,
# because the cap in cell 4 never bound and the per-patch cap was not the limit either. Above
# it the cut has to leave the season: Lithuania's other seasons hold Fall 37,478, Spring
# 22,020 and Winter under 18,629 reference rows, the whole corpus 509,069. Keeping the
# country fixed isolates "more rows" from "more geography"; the test split stays Summer.
CELL_REF_ROWS = 10986
REF_ROWS_PER_PATCH = 3.0   # 10,986 / 3,107 = 3.54 in the cell; 3.0 leaves margin so the row cap binds


def ground_ref_cell_flags(ref_rows):
    if ref_rows <= CELL_REF_ROWS:
        return "--country Lithuania --season Summer"
    return "--country Lithuania"


def ground_ref_max_patches(ref_rows):
    if ref_rows <= CELL_REF_ROWS:
        return 4000
    return int(-(-ref_rows // REF_ROWS_PER_PATCH))


def ground_ref_cut_comment(ref_rows):
    if ref_rows <= CELL_REF_ROWS:
        return []
    return [
        "# %d reference rows is more than the Lithuania/Summer train cell holds (%d, measured\n" % (ref_rows, CELL_REF_ROWS),
        "# 8 Sep), so this cut takes all four Lithuanian seasons. Same country, same annotation\n",
        "# form; the test split is still Lithuania/Summer.\n",
    ]


def build(data="ben", vision_lora=False, count_answer="integer",
          vision_lr_mult=1.0, rsvqa_rows=20000, ref_rows=CELL_REF_ROWS, lora_rank=8,
          visual_tokens=256, vrsbench_rows=30000):
    # RSVQA counting rows carry the published bucket label instead of a raw
    # integer when count_answer == "bucket", and the eval cell has to ask the
    # same way or it scores the adapter on a format it never saw. One choice
    # drives both, so the notebook cannot ship a mismatched pair.
    count_flags = " --count-answer bucket" if count_answer == "bucket" else ""
    count_eval = " --count-prompt bucket" if count_answer == "bucket" else ""
    cells = []

    cells.append(md(
        "# SatQuery - QLoRA on Qwen2.5-VL-3B, on a Kaggle T4\n",
        "\n",
        "Trains the VQA adapter on BigEarthNet.txt annotations joined to reBEN\n",
        "imagery, then scores it against our 60.8% baseline on the same harness.\n",
        "\n",
        "## Before you press anything\n",
        "\n",
        "1. **Settings -> GPU hours.** 30 a week. This run needs ~11 of them.\n",
        "   **If quota is spent you get a Tesla P100 and nothing here will load.**\n",
        "2. **Accelerator = GPU T4 x2**, **Internet = On**.\n",
        "3. **Save Version -> Save & Run All**, then close the tab. Do not sit on it.\n",
        "\n",
        "## If it stops at ~10.5 hours, that is correct\n",
        "\n",
        "Kaggle kills a session at 12 h. This stops itself early, saves, and exits\n",
        "cleanly. To continue: publish `/kaggle/working/ckpt` as a dataset, attach it\n",
        "to this notebook, and Run All again - it resumes from the last step with the\n",
        "optimizer and LR schedule intact.",
    ))

    cells.append(code(
        "# ---- 1. preflight: which card, and how long have we got? ----\n",
        "# ONE GPU, and this line has to come before torch is imported.\n",
        "#\n",
        "# This is what killed run 7, and it is worth naming precisely because\n",
        "# nothing in the traceback says it. Kaggle hands out TWO T4s. Trainer sets\n",
        "# n_gpu = torch.cuda.device_count() and, seeing 2, silently wraps the model\n",
        "# in nn.DataParallel. DataParallel REPLICATES the whole model on every\n",
        "# forward, so GPU 0 carried the 7.51 GB master AND a 6.6 GB replica: 14.16\n",
        "# GiB allocated of a 14.56 GiB card, and the OOM landed in backward.\n",
        "#\n",
        "# The tell in the log is one warning from torch/autograd/function.py:\n",
        "# 'Was asked to gather along dimension 0' - that is DataParallel's Gather,\n",
        "# and it appeared in the Trainer step but NOT in the smoke test one cell\n",
        "# earlier, which is why the smoke test passed at 8.20 GB and the real step\n",
        "# died. device_map={'': 0} does not prevent this; it pins where the weights\n",
        "# LOAD, and Trainer wraps whatever it is given afterwards.\n",
        "#\n",
        "# Batch size is 1 here, so the second card was never going to help anyway:\n",
        "# DataParallel splits along the batch and there is nothing to split.\n",
        "import os\n",
        "os.environ['CUDA_VISIBLE_DEVICES'] = '0'\n",
        "\n",
        "import time, torch\n",
        "T_START = time.time()\n",
        "\n",
        "assert torch.cuda.device_count() == 1, (\n",
        "    'CUDA_VISIBLE_DEVICES did not take - torch sees %d cards. It has to be '\n",
        "    'set before the first torch import, and a re-run of this cell alone in a '\n",
        "    'live kernel will not do it. Restart the session.'\n",
        "    % torch.cuda.device_count())\n",
        "\n",
        "# Stop with 1.5 h of head-room. Saving a 3B checkpoint is not instant, and a\n",
        "# session killed mid-write loses the checkpoint as well as the run.\n",
        "TRAIN_HOURS = 10.5\n",
        "\n",
        "if not torch.cuda.is_available():\n",
        "    raise RuntimeError('No GPU. Right panel -> Accelerator -> GPU T4 x2.')\n",
        "name = torch.cuda.get_device_name(0)\n",
        "major, minor = torch.cuda.get_device_capability(0)\n",
        "sm = major * 10 + minor\n",
        "print('device     :', name)\n",
        "print('capability : sm_%d' % sm)\n",
        "print('memory     : %.1f GB' % (torch.cuda.get_device_properties(0).total_memory / 1e9))\n",
        "if sm < 75:\n",
        "    raise RuntimeError(\n",
        "        'STOP. %s is sm_%d and 4-bit will not load on it. This is your weekly '\n",
        "        'GPU quota being spent, not a broken install - Kaggle substitutes this '\n",
        "        'card when it cannot give you a T4. Check Settings -> GPU hours.'\n",
        "        % (name, sm))\n",
        "print('\\nOK. Budget: %.1f h of training, then a clean stop.' % TRAIN_HOURS)",
    ))

    cells.append(code(
        "# ---- 2. deps + our two scripts (embedded, nothing to upload) ----\n",
        "# PIN transformers below 5. The v5 major release rewrote TrainingArguments\n",
        "# and dropped warmup_ratio among others - `-U` pulled it and the run died\n",
        "# three minutes in, AFTER the 7.5 GB model had already loaded. Qwen2.5-VL\n",
        "# needs >=4.49, so this is the supported window, not a downgrade.\n",
        "# torchao is upgraded alongside because transformers imports it during\n",
        "# Trainer setup to probe quantisation backends, and Kaggle's image ships\n",
        "# 0.10.0, which a 4.5x transformers rejects:\n",
        "#     ImportError: Found an incompatible version of torchao. Found 0.10.0\n",
        "# We do not quantise at all any more, but the import happens regardless.\n",
        "# VERIFIED 3 Sep, v8: this cell, the fp16 load and 1,500 training steps ran\n",
        "# end to end on the T4 (transformers 4.57.6, peft 0.20.0). bitsandbytes is\n",
        "# deliberately NOT installed here - nothing in this notebook quantises - so\n",
        "# anything that loads the model must ask for fp16, eval included (cell 8).\n",
        "!pip install -q 'transformers>=4.49,<5' -U torchao accelerate peft datasets pyarrow lmdb safetensors\n",
        "import accelerate, peft, transformers\n",
        "print('transformers', transformers.__version__,\n",
        "      '| peft', peft.__version__,\n",
        "      '| accelerate', accelerate.__version__)\n",
        "import base64, json, os, sys\n",
        "\n",
        "PAYLOAD = '''%s'''\n" % _payload(),
        "\n",
        "root = '/kaggle/working/src'\n",
        "for rel, b64 in json.loads(base64.b64decode(PAYLOAD)).items():\n",
        "    dst = os.path.join(root, rel)\n",
        "    os.makedirs(os.path.dirname(dst), exist_ok=True)\n",
        "    open(dst, 'wb').write(base64.b64decode(b64))\n",
        "sys.path.insert(0, root)\n",
        "print('wrote', list(json.loads(base64.b64decode(PAYLOAD))))",
    ))

    cells.append(code(
        "# ---- 3. data, straight from HuggingFace at Kaggle's bandwidth ----\n",
        "# 467 MB of annotations + 2.3 GB of imagery. Neither touches your laptop.\n",
        "import os\n",
        "from huggingface_hub import hf_hub_download, snapshot_download\n",
        "os.makedirs('/kaggle/working/data', exist_ok=True)\n",
        "\n",
        "PARQUET = hf_hub_download('BIFOLD-BigEarthNetv2-0/BigEarthNet.txt',\n",
        "                          'BigEarthNet.txt.parquet', repo_type='dataset',\n",
        "                          local_dir='/kaggle/working/data')\n",
        "print('annotations:', PARQUET, '%.0f MB' % (os.path.getsize(PARQUET) / 1e6))\n",
        "\n",
        "LMDB_DIR = snapshot_download('hackelle/BigEarthNetV2-Lithuania-Summer-LMDB',\n",
        "                             repo_type='dataset', local_dir='/kaggle/working/data/ben')\n",
        "print('imagery    :', LMDB_DIR)\n",
        "for r, d, f in os.walk(LMDB_DIR):\n",
        "    for n in f:\n",
        "        print('   %-40s %8.1f MB' % (n, os.path.getsize(os.path.join(r, n)) / 1e6))",
    ))

    cells.append(code(
        "# ---- 4. cut the smoke subset ----\n",
        "# Lithuania/Summer is the cell whose imagery we actually have. Country\n",
        "# imbalance is real - Finland alone is a third of BigEarthNet - so taking a\n",
        "# cell deliberately beats sampling 'at random' and getting boreal forest.\n",
        "!python /kaggle/working/src/scripts/make_subset.py \\\n",
        "    --parquet /kaggle/working/data/BigEarthNet.txt.parquet \\\n",
        "    --split train --country Lithuania --season Summer \\\n",
        "    --max-patches 4000 --per-patch 6 --types binary,mcq \\\n",
        "    --out /kaggle/working/data/subset_train.jsonl\n",
        "\n",
        "import json\n",
        "rows = [json.loads(l) for l in open('/kaggle/working/data/subset_train.jsonl')]\n",
        "print('rows:', len(rows), '| patches:', len({r['patch_id'] for r in rows}))\n",
        "print('types:', {t: sum(1 for r in rows if r['type'] == t) for t in {r['type'] for r in rows}})\n",
        "print('\\nexample:', json.dumps(rows[0], indent=2)[:400])",
    ))

    cells.append(code(
        "# ---- 5. the loader. THIS is where runs die silently. ----\n",
        "# MEASURED on the first run, not assumed: an LMDB value is 160,832 bytes,\n",
        "# which is NOT a stacked cube. reBEN stores Sentinel-2 at NATIVE\n",
        "# resolutions, so the bands are different sizes:\n",
        "#\n",
        "#     4 x 120x120 (B02 B03 B04 B08, 10 m)  = 57,600 px\n",
        "#     6 x  60x60  (B05 B06 B07 B8A B11 B12, 20 m) = 21,600 px\n",
        "#     79,200 px x 2 bytes (uint16) = 158,400 + header = 160,832\n",
        "#\n",
        "# That is TEN bands, not twelve - the 60 m bands really are excluded - and\n",
        "# the value is a safetensors dict keyed BY BAND NAME. So pick RGB by name\n",
        "# (B04/B03/B02), never by index: an index into a differently-ordered or\n",
        "# differently-sized set is silently the wrong colour channel.\n",
        "import glob, lmdb, numpy as np\n",
        "from PIL import Image\n",
        "\n",
        "cands = [p for p in glob.glob('/kaggle/working/data/ben/**/*', recursive=True)\n",
        "         if os.path.isdir(p) and glob.glob(os.path.join(p, 'data.mdb'))]\n",
        "LMDB_PATH = cands[0] if cands else '/kaggle/working/data/ben'\n",
        "env = lmdb.open(LMDB_PATH, readonly=True, lock=False, readahead=False, meminit=False)\n",
        "with env.begin() as txn:\n",
        "    print('lmdb :', LMDB_PATH, '|', txn.stat()['entries'], 'entries')\n",
        "\n",
        "with env.begin() as txn:\n",
        "    present = [r['patch_id'] for r in rows[:500] if txn.get(r['patch_id'].encode())]\n",
        "print('patch_id hit rate in first 500 rows: %d/500' % len(present))\n",
        "assert present, 'no patch_id resolves in the LMDB - check keys before going on.'\n",
        "\n",
        "# Decode ONE value and PRINT its structure. Never guess a layout.\n",
        "from safetensors.numpy import load as st_load\n",
        "with env.begin() as txn:\n",
        "    raw = txn.get(present[0].encode())\n",
        "print('value: %d bytes' % len(raw))\n",
        "sample = st_load(bytes(raw))\n",
        "for k in sorted(sample):\n",
        "    print('   %-6s %-12s %s' % (k, sample[k].dtype, sample[k].shape))\n",
        "\n",
        "def _pick(keys, *wanted):\n",
        "    \"\"\"Find a band by name, tolerating B04 / B4 / b04 spellings.\"\"\"\n",
        "    for w in wanted:\n",
        "        for k in keys:\n",
        "            if k.upper().replace('0', '') == w.upper().replace('0', ''):\n",
        "                return k\n",
        "    return None\n",
        "\n",
        "KEYS = sorted(sample)\n",
        "RGB = [_pick(KEYS, 'B04'), _pick(KEYS, 'B03'), _pick(KEYS, 'B02')]\n",
        "assert all(RGB), (\n",
        "    'could not find B04/B03/B02 among %s - do NOT fall back to positional '\n",
        "    'indexing, that is how you train on the wrong colour channels' % KEYS)\n",
        "print('rgb bands:', RGB)\n",
        "\n",
        "def to_rgb(patch_id, size=224):\n",
        "    with env.begin() as txn:\n",
        "        raw = txn.get(patch_id.encode())\n",
        "    if raw is None:\n",
        "        return None\n",
        "    d = st_load(bytes(raw))\n",
        "    out = []\n",
        "    for b in RGB:\n",
        "        ch = d[b].astype('float32')\n",
        "        if ch.ndim == 3:\n",
        "            ch = ch[0]\n",
        "        lo, hi = np.percentile(ch, (2, 98))\n",
        "        out.append(np.clip((ch - lo) / max(hi - lo, 1e-6), 0, 1))\n",
        "    img = Image.fromarray((np.dstack(out) * 255).astype('uint8'), 'RGB')\n",
        "    return img.resize((size, size), Image.BILINEAR)\n",
        "\n",
        "# PROVE the images are not blank before training on them.\n",
        "ok = 0\n",
        "for pid in present[:200]:\n",
        "    im = to_rgb(pid)\n",
        "    if im is not None and np.asarray(im).std() > 5:\n",
        "        ok += 1\n",
        "print('non-black images: %d / %d' % (ok, len(present[:200])))\n",
        "assert ok > len(present[:200]) * 0.5, (\n",
        "    'the loader is producing blank images - FIX THIS BEFORE TRAINING. Training '\n",
        "    'on black images does not raise anything; it just wastes the run.')\n",
        "display(to_rgb(present[0]))",
    ))


    if data == "rsvqa":
        cells[0] = md(
            "# SatQuery - QLoRA on Qwen2.5-VL-3B, on a Kaggle T4 - RSVQA-LR train split\n",
            "\n",
            "Second recipe. The first (BigEarthNet.txt, v8, 3 Sep) moved RSVQA-LR from\n",
            "60.75 to 62.0 and every point of it was a yes-bias shift (PLAN section 9).\n",
            "This trains on RSVQA-LR's OWN train split - 572 images, no overlap with the\n",
            "100 test images - which is the recipe that reached 92.11 on this test set\n",
            "with this exact LoRA config (RSHallu, arXiv:2602.10799).\n",
            "\n",
            "## Before you press anything\n",
            "\n",
            "1. **Settings -> GPU hours.** 30 a week. This run needs ~4 of them.\n",
            "   **If quota is spent you get a Tesla P100 and nothing here will load.**\n",
            "2. **Accelerator = GPU T4 x2**, **Internet = On**.\n",
            "3. **Save Version -> Save & Run All**, then close the tab. Do not sit on it.\n",
        )
        cells[3:6] = [
            code(
                "# ---- 3. data: RSVQA-LR, straight from Zenodo ----\n",
                "# 95 MB of images plus the six split files. The same directory serves the\n",
                "# eval in cell 8, so the test split is downloaded here too - and cell 4\n",
                "# uses it to PROVE the train rows share no image with it.\n",
                "import os\n",
                "RSVQA = '/kaggle/working/rsvqa'\n",
                "os.makedirs(RSVQA, exist_ok=True)\n",
                "base = 'https://zenodo.org/records/6344334/files'\n",
                "for f in ['Images_LR.zip',\n",
                "          'LR_split_train_questions.json', 'LR_split_train_answers.json',\n",
                "          'LR_split_test_questions.json', 'LR_split_test_answers.json']:\n",
                "    p = os.path.join(RSVQA, f)\n",
                "    if not os.path.exists(p) or os.path.getsize(p) < 10000:\n",
                "        !curl -sL -C - -o {p} \"{base}/{f}?download=1\"\n",
                "    print('%-34s %9.1f MB' % (f, os.path.getsize(p) / 1e6))\n",
                "import json\n",
                "for f in os.listdir(RSVQA):\n",
                "    if f.endswith('.json'):\n",
                "        json.load(open(os.path.join(RSVQA, f)))   # a truncated download dies HERE, not in cell 7\n",
                "print('all split files parse')",
            ),
            code(
                "# ---- 4. cut the training subset ----\n",
                "# Balanced across the four question types, strided across all 572 images,\n",
                "# wrapped in the eval harness's OWN prompt templates (imported, not copied).\n",
                "# 20,000 rows. Sized when v8 was believed to have taken ~10 h; Kaggle\n",
                "# records that run at 3 h 44 min wall clock for 24,000 rows, so the 10.5 h\n",
                "# stop has room for the whole 57k split. Raise --limit on the next run.\n",
                "!python /kaggle/working/src/scripts/make_rsvqa_subset.py \\\n",
                "    --data-dir /kaggle/working/rsvqa --limit %d%s \\\n" % (rsvqa_rows, count_flags),
                "    --out /kaggle/working/data/rsvqa_train.jsonl\n",
                "\n",
                "import json\n",
                "rows = [json.loads(l) for l in open('/kaggle/working/data/rsvqa_train.jsonl')]\n",
                "print('rows:', len(rows), '| images:', len({r['patch_id'] for r in rows}))\n",
                "print('types:', {t: sum(1 for r in rows if r['type'] == t) for t in {r['type'] for r in rows}})\n",
                "print('\\nexample:', json.dumps(rows[0], indent=2)[:400])",
            ),
            code(
                "# ---- 5. the loader: the eval's own image decoder, so train and test see\n",
                "# identical pixels. 256x256 native, NOT resized - the harness scores at 256.\n",
                "import numpy as np\n",
                "from scripts.eval_rsvqa_lr import load_images\n",
                "\n",
                "images = load_images(os.path.join(RSVQA, 'Images_LR.zip'), {r['patch_id'] for r in rows})\n",
                "print('decoded %d images at %s' % (len(images), next(iter(images.values())).size))\n",
                "missing = [r['patch_id'] for r in rows if r['patch_id'] not in images]\n",
                "assert not missing, '%d rows point at images the zip does not hold, e.g. %s' % (len(missing), missing[:5])\n",
                "\n",
                "def to_rgb(patch_id, size=None):\n",
                "    return images[patch_id]\n",
                "\n",
                "# PROVE the images are not blank before training on them.\n",
                "present = sorted(images)[:200]\n",
                "ok = sum(1 for pid in present if np.asarray(to_rgb(pid)).std() > 5)\n",
                "print('non-black images: %d / %d' % (ok, len(present)))\n",
                "assert ok > len(present) * 0.5, 'the loader is producing blank images - FIX THIS BEFORE TRAINING.'\n",
                "display(to_rgb(present[0]))",
            ),
        ]

    if data == "cdvqa":
        cells[0] = md(
            "# SatQuery - QLoRA on Qwen2.5-VL-3B, on a Kaggle T4 - CDVQA change pairs\n",
            "\n",
            "Third recipe: bi-temporal change VQA. The base model, asked what changed\n",
            "between two images, answered that it could not see any images to compare\n",
            "(PLAN section 2). Capability 3 is fine-tune or nothing, so this trains on\n",
            "CDVQA (Yuan et al., TGRS 2022): 1,600 SECOND image pairs, eight question\n",
            "types with closed vocabularies, scored by exact match on the dataset's\n",
            "own tokens. Same LoRA config as the VQA adapter that reached 85.5%.\n",
            "\n",
            "## Before you press anything\n",
            "\n",
            "1. **Settings -> GPU hours.** 30 a week. Two images per row: budget ~7 of them.\n",
            "   **If quota is spent you get a Tesla P100 and nothing here will load.**\n",
            "2. **Accelerator = GPU T4 x2**, **Internet = On**.\n",
            "3. **Save Version -> Save & Run All**, then close the tab. Do not sit on it.\n",
        )
        cells[3:6] = [
            code(
                "# ---- 3. data: CDVQA annotations from GitHub, SECOND pairs from Drive ----\n",
                "# CDVQA ships annotations only. Its images are the SECOND train set,\n",
                "# a 2.4 GB rar on the authors' Google Drive. The FIRST Drive link on the\n",
                "# SCD page was over its daily quota on 4 Sep (Google returns a 2 KB HTML\n",
                "# page that gdown happily saves as the file); the second link served the\n",
                "# whole rar at 10 MB/s. Size is asserted, because a truncated archive\n",
                "# fails in 7z with a message that looks like a corrupt download.\n",
                "import os, subprocess\n",
                "D = '/kaggle/working/cdvqa'\n",
                "os.makedirs(D, exist_ok=True)\n",
                "if not os.path.exists(D + '/CDVQA/Train_questions.json'):\n",
                "    !git clone -q --depth 1 https://github.com/YZHJessica/CDVQA {D}/CDVQA\n",
                "for f in ['Train', 'Test', 'Val']:\n",
                "    for k in ['questions', 'answers', 'images']:\n",
                "        p = f'{D}/CDVQA/{f}_{k}.json'\n",
                "        assert os.path.exists(p), p\n",
                "print('annotations ok')\n",
                "\n",
                "RAR = D + '/SECOND_train_set.rar'\n",
                "RAR_BYTES = 2406111691\n",
                "URL = ('https://drive.usercontent.google.com/download'\n",
                "       '?id=1QlAdzrHpfBIOZ6SK78yHF2i1u6tikmBc&export=download&confirm=t')\n",
                "for attempt in range(8):\n",
                "    if os.path.exists(RAR) and os.path.getsize(RAR) >= RAR_BYTES:\n",
                "        break\n",
                "    !curl -sL -C - -o {RAR} \"{URL}\"\n",
                "assert os.path.getsize(RAR) >= RAR_BYTES, 'rar is %d bytes, expected %d' % (os.path.getsize(RAR), RAR_BYTES)\n",
                "print('rar: %.2f GB' % (os.path.getsize(RAR) / 1e9))\n",
                "\n",
                "IMG = D   # the rar unpacks im1/ im2/ label1/ label2/ at its ROOT - measured 4 Sep\n",
                "# Kaggle's 7z has NO RAR codec. Version 2 (4 Sep) extracted a handful of\n",
                "# stored entries and printed 'Unsupported Method' 9,800 times to stderr,\n",
                "# which -bso0 hid; the loader's decode assert caught it forty minutes\n",
                "# later. So: install p7zip-rar first, keep 7z's stderr, refuse on any\n",
                "# 'Unsupported Method', and decode a sample of PNGs before going on.\n",
                "from PIL import Image\n",
                "if not os.path.isdir(IMG + '/im1') or len(os.listdir(IMG + '/im1')) < 2968:\n",
                "    !apt-get -qq update > /dev/null && apt-get -qq install -y p7zip-full p7zip-rar > /dev/null\n",
                "    !cd {D} && 7z x -y -bso0 -bsp0 SECOND_train_set.rar 2> {D}/7z.err; true\n",
                "    err = open(D + '/7z.err').read()\n",
                "    bad = err.count('Unsupported Method')\n",
                "    assert bad == 0, '7z could not decode %d entries - the RAR codec is missing:\\n%s' % (bad, err[:500])\n",
                "for sub in ['im1', 'im2']:\n",
                "    names = sorted(os.listdir(f'{IMG}/{sub}'))\n",
                "    print('%-4s %d files' % (sub, len(names)))\n",
                "    assert len(names) >= 2968, 'expected 2,968 SECOND train images in ' + sub\n",
                "    for nm in names[::300]:\n",
                "        Image.open(f'{IMG}/{sub}/{nm}').verify()   # raises on a truncated or garbage PNG\n",
                "print('sampled PNGs decode in both folders')",
            ),
            code(
                "# ---- 4. cut the training subset ----\n",
                "# Balanced across the eight question types, strided across all 1,600\n",
                "# train files (each file carries 16 augmented image ids - one row per id\n",
                "# would be sixteen near-copies of one picture), no file shared with the\n",
                "# test split, wrapped in the harness's own templates and pre/post sentence.\n",
                "!python /kaggle/working/src/scripts/make_cdvqa_subset.py \\\n",
                "    --data-dir {D}/CDVQA --limit 20000 \\\n",
                "    --out /kaggle/working/data/cdvqa_train.jsonl\n",
                "\n",
                "import json\n",
                "rows = [json.loads(l) for l in open('/kaggle/working/data/cdvqa_train.jsonl')]\n",
                "print('rows:', len(rows), '| files:', len({r['patch_id'] for r in rows}))\n",
                "print('types:', {t: sum(1 for r in rows if r['type'] == t) for t in {r['type'] for r in rows}})\n",
                "print('\\nexample:', json.dumps(rows[0], indent=2)[:600])",
            ),
            code(
                "# ---- 5. the pair loader: (pre, post), the harness's own decoder ----\n",
                "# 512 px native; the processor's 256-token cap resizes each to ~448 px,\n",
                "# the same as at eval time. Both images must exist or the row is refused -\n",
                "# a silently missing second image turns a change question into a\n",
                "# single-image one and the loss still goes down.\n",
                "import numpy as np\n",
                "from scripts.eval_cdvqa import load_pairs\n",
                "\n",
                "pairs = load_pairs(IMG, {r['patch_id'] for r in rows})\n",
                "missing = [r['patch_id'] for r in rows if r['patch_id'] not in pairs]\n",
                "assert not missing, '%d rows have no pair on disk, e.g. %s' % (len(missing), missing[:5])\n",
                "print('decoded %d pairs at %s' % (len(pairs), next(iter(pairs.values()))[0].size))\n",
                "\n",
                "def to_rgb(patch_id, size=None):\n",
                "    return list(pairs[patch_id])      # [pre, post] - the collator takes a list\n",
                "\n",
                "# PROVE the pairs differ and are not blank before training on them.\n",
                "present = sorted(pairs)[:200]\n",
                "ok = sum(1 for f in present if np.asarray(pairs[f][0]).std() > 5 and np.asarray(pairs[f][1]).std() > 5)\n",
                "diff = sum(1 for f in present if np.abs(np.asarray(pairs[f][0], dtype='float32') - np.asarray(pairs[f][1], dtype='float32')).mean() > 1)\n",
                "print('non-black pairs: %d / %d | pre != post: %d / %d' % (ok, len(present), diff, len(present)))\n",
                "assert ok > len(present) * 0.5 and diff > len(present) * 0.5, 'the loader is producing blank or identical pairs - FIX THIS BEFORE TRAINING.'\n",
                "display(pairs[present[0]][0]); display(pairs[present[0]][1])",
            ),
        ]
        # cell 8 does not exist yet at this point; it is swapped in below,
        # right after the generic eval cell is appended.
        global _CDVQA_EVAL
        _CDVQA_EVAL = code(
            "# ---- 8. the number that matters: change VQA, measured ----\n",
            "# Same harness, same 400 stratified test questions the office box scores.\n",
            "import gc, os\n",
            "for _n in ('trainer', 'model', '_b', '_dev', '_out'):\n",
            "    if _n in globals():\n",
            "        del globals()[_n]\n",
            "gc.collect()\n",
            "torch.cuda.empty_cache()\n",
            "_held = torch.cuda.memory_allocated() / 1e9\n",
            "print('GPU held by this kernel after release: %.2f GB' % _held)\n",
            "assert _held < 1.0, 'this kernel still holds %.2f GB - the eval subprocess will OOM' % _held\n",
            "\n",
            "!CUDA_VISIBLE_DEVICES=1 python /kaggle/working/src/scripts/eval_cdvqa.py \\\n",
            "    --data-dir {D}/CDVQA --images-dir {IMG} --split Test \\\n",
            "    --model Qwen/Qwen2.5-VL-3B-Instruct \\\n",
            "    --adapter /kaggle/working/lora_adapter \\\n",
            "    --dtype fp16 --limit 400 --out /kaggle/working/after_lora\n",
            "\n",
            "import json\n",
            "a = json.load(open('/kaggle/working/after_lora.json'))\n",
            "print('\\n' + '=' * 52)\n",
            "print('after LoRA : %.1f%%  vs majority %.1f%%  (%+.1f)'\n",
            "      % (a['overall_accuracy'], a['majority_baseline'], a['beats_majority_by']))\n",
            "print('=' * 52)\n",
            "for t, d in sorted(a['per_type'].items()):\n",
            "    print('  %-20s %6.1f%%  majority %5.1f%%  %+6.1f'\n",
            "          % (t, d['accuracy'], d['majority'], d['accuracy'] - d['majority']))",
        )

    if data in ("ground", "ground-ref"):
        cells[0] = md(
            "# SatQuery - QLoRA on Qwen2.5-VL-3B, on a Kaggle T4 - grounding on BigEarthNet.txt boxes\n",
            "\n",
            "Fourth recipe: grounding. Base Qwen answers 'where is the road?' with a box\n",
            "around the whole tile (PLAN section 9). BigEarthNet.txt carries 2.2 M box\n",
            "annotations - a box around a given point, and a box for a referring\n",
            "expression such as <ref>largest connected region of pastures</ref> -\n",
            "normalised 0-1 as `[x0 y0, x1 y1]`. The adapter learns that string; the\n",
            "engine parses it back. Scored by mean IoU and Acc@0.5 against the\n",
            "whole-image box, on the Lithuania/Summer test split.\n",
            "\n",
            "## Before you press anything\n",
            "\n",
            "1. **Settings -> GPU hours.** 30 a week. This run needs ~4 of them.\n",
            "   **If quota is spent you get a Tesla P100 and nothing here will load.**\n",
            "2. **Accelerator = GPU T4 x2**, **Internet = On**.\n",
            "3. **Save Version -> Save & Run All**, then close the tab. Do not sit on it.\n",
        )
        cells[4] = code(
            "# ---- 4. cut the grounding subset ----\n",
            "# Same cell as the VQA recipe (Lithuania/Summer, the imagery we hold), but\n",
            "# ONLY bounding-box rows: 307,756 of them in this cell, two categories.\n",
            "# 4,000 patches x 5 = 20,000 rows, v8's budget. The harness's own prompt\n",
            "# suffix is appended so train and test ask in one form.\n",
            "!python /kaggle/working/src/scripts/make_subset.py \\\n",
            "    --parquet /kaggle/working/data/BigEarthNet.txt.parquet \\\n",
            "    --split train --country Lithuania --season Summer \\\n",
            "    --max-patches 4000 --per-patch 5 --types \"bounding box\" \\\n",
            "    --out /kaggle/working/data/subset_ground.jsonl\n",
            "\n",
            "import json\n",
            "from scripts.eval_ground import PROMPT_SUFFIX, parse_gold\n",
            "rows = [json.loads(l) for l in open('/kaggle/working/data/subset_ground.jsonl')]\n",
            "for r in rows:\n",
            "    r['input'] = r['input'] + PROMPT_SUFFIX\n",
            "bad = [r for r in rows if parse_gold(r['output']) is None]\n",
            "assert not bad, '%d rows have a gold box that does not parse, e.g. %r' % (len(bad), bad[:2])\n",
            "print('rows:', len(rows), '| patches:', len({r['patch_id'] for r in rows}))\n",
            "print('categories:', {c: sum(1 for r in rows if r['category'] == c) for c in {r['category'] for r in rows}})\n",
            "print('\\nexample:', json.dumps(rows[0], indent=2)[:500])",
        )
        global _GROUND_EVAL
        _GROUND_EVAL = code(
            "# ---- 8. the number that matters: grounding, measured ----\n",
            "# Same harness and 400-row stratified test sample the office box scores.\n",
            "import gc, os\n",
            "for _n in ('trainer', 'model', '_b', '_dev', '_out'):\n",
            "    if _n in globals():\n",
            "        del globals()[_n]\n",
            "gc.collect()\n",
            "torch.cuda.empty_cache()\n",
            "_held = torch.cuda.memory_allocated() / 1e9\n",
            "print('GPU held by this kernel after release: %.2f GB' % _held)\n",
            "assert _held < 1.0, 'this kernel still holds %.2f GB - the eval subprocess will OOM' % _held\n",
            "\n",
            "!CUDA_VISIBLE_DEVICES=1 python /kaggle/working/src/scripts/eval_ground.py \\\n",
            "    --parquet /kaggle/working/data/BigEarthNet.txt.parquet \\\n",
            "    --lmdb /kaggle/working/data/ben \\\n",
            "    --model Qwen/Qwen2.5-VL-3B-Instruct \\\n",
            "    --adapter /kaggle/working/lora_adapter \\\n",
            "    --dtype fp16 --limit 400 --out /kaggle/working/after_lora\n",
            "\n",
            "import json\n",
            "a = json.load(open('/kaggle/working/after_lora.json'))\n",
            "print('\\n' + '=' * 52)\n",
            "print('after LoRA : mIoU %.3f  Acc@0.5 %.1f%%   (whole-image box: mIoU %.3f  Acc@0.5 %.1f%%)'\n",
            "      % (a['miou'], a['acc50'], a['whole_miou'], a['whole_acc50']))\n",
            "print('=' * 52)\n",
            "for c, d in sorted(a['per_category'].items()):\n",
            "    print('  %-10s mIoU %.3f  Acc@0.5 %5.1f%%  whole %.3f / %5.1f%%'\n",
            "          % (c, d['miou'], d['acc50'], d['whole_miou'], d['whole_acc50']))",
        )

    if data in ("vrsbench-vqa", "vrsbench-ref"):
        vtask = "vqa" if data == "vrsbench-vqa" else "refer"
        cells[0] = md(
            "# SatQuery - QLoRA on Qwen2.5-VL-3B, on a Kaggle T4 - VRSBench %s\n" % vtask,
            "\n",
            "VRSBench is a prescribed grading split we hold no trained number on: VQA 42.7\n",
            "zero-shot (base 43.4), referring 13.0 Acc@0.5 (base 37.8) - the adapters\n",
            "trained on 120 px / 10 m tiles do not survive 512 px sub-metre imagery.\n",
            "Others reach 60.6 VQA / 39.6 referring by training on VRSBench's own train\n",
            "split (GeoChat, LoRA r=64, 5 epochs). This trains a SEPARATE adapter on it so\n",
            "the live 10 m adapters keep their numbers; the dispatcher picks by tile size.\n",
            "\n",
            "## Before you press anything\n",
            "\n",
            "1. **Settings -> GPU hours.** 30 a week. This run needs ~8-10 of them.\n",
            "   **If quota is spent you get a Tesla P100 and nothing here will load.**\n",
            "2. **Accelerator = GPU T4 x2**, **Internet = On**.\n",
            "3. **Save Version -> Save & Run All**, then close the tab. Do not sit on it.\n",
        )
        cells[3:6] = [
            code(
                "# ---- 3. data: VRSBench train + val, from the authors' HF repo ----\n",
                "# 12.4 GB of zips, never extracted (the loader reads members lazily), and\n",
                "# under /kaggle/tmp rather than /kaggle/working: the working dir is the\n",
                "# kernel's OUTPUT and has a 20 GB cap. The val files are here for two\n",
                "# reasons - cell 8 scores on them, and cell 4 PROVES no train image is\n",
                "# one of them.\n",
                "import os\n",
                "VRS = '/kaggle/tmp/vrsbench'\n",
                "!cd /kaggle/working/src && python scripts/get_vrsbench.py --dest {VRS} --split both\n",
                "for f in ['VRSBench_train.json', 'Images_train.zip', 'VRSBench_EVAL_vqa.json',\n",
                "          'VRSBench_EVAL_referring.json', 'Images_val.zip']:\n",
                "    p = os.path.join(VRS, f)\n",
                "    assert os.path.exists(p), f\n",
                "    print('%-30s %12d bytes' % (f, os.path.getsize(p)))\n",
            ),
            code(
                "# ---- 4. cut the training subset ----\n",
                "# The dataset's own answer strings (referring boxes stay {<x1><y1><x2><y2>}\n",
                "# on 0-100), asked in the eval harness's OWN templates (imported, not\n",
                "# copied). Refuses any image shared with the EVAL split.\n",
                "!cd /kaggle/working/src && python scripts/make_vrsbench_subset.py \\\n",
                "    --train-json {VRS}/VRSBench_train.json --task %s --limit %d \\\n" % (vtask, vrsbench_rows),
                "    --eval-json {VRS}/VRSBench_EVAL_vqa.json --eval-json {VRS}/VRSBench_EVAL_referring.json \\\n",
                "    --out /kaggle/working/data/vrsbench_train.jsonl\n",
                "\n",
                "import json\n",
                "rows = [json.loads(l) for l in open('/kaggle/working/data/vrsbench_train.jsonl')]\n",
                "print('rows:', len(rows), '| images:', len({r['patch_id'] for r in rows}))\n",
                "print('\\nexample:', json.dumps(rows[0], indent=2)[:400])",
            ),
            code(
                "# ---- 5. the loader: the eval's own lazy zip reader, so train and test see\n",
                "# identical pixels. 512x512 native; the processor caps at %d visual tokens\n" % visual_tokens,
                "# (512 px is 334 tokens at 28 px patches - 256 would crop).\n",
                "import numpy as np\n",
                "from scripts.eval_vrsbench import ZipImages\n",
                "\n",
                "images = ZipImages(os.path.join(VRS, 'Images_train.zip'))\n",
                "sample = sorted({r['patch_id'] for r in rows})[:200]\n",
                "missing = [pid for pid in sample if pid not in images]\n",
                "assert not missing, '%d of the first 200 rows point at images the zip does not hold, e.g. %s' % (len(missing), missing[:5])\n",
                "\n",
                "def to_rgb(patch_id, size=None):\n",
                "    return images.get(patch_id)\n",
                "\n",
                "ok = sum(1 for pid in sample[:50] if np.asarray(to_rgb(pid)).std() > 5)\n",
                "print('non-black images: %d / 50 | size %s' % (ok, to_rgb(sample[0]).size))\n",
                "assert ok > 25, 'the loader is producing blank images - FIX THIS BEFORE TRAINING.'\n",
                "display(to_rgb(sample[0]))",
            ),
        ]
        global _VRSBENCH_EVAL
        _VRSBENCH_EVAL = code(
            "# ---- 8. the number that matters: VRSBench %s, measured ----\n" % vtask,
            "# Same harness and 400-row stratified sample the 6 Sep zero-shot numbers used\n",
            "# (VQA 42.7 / referring 13.0 with the 10 m adapters; base 43.4 / 37.8).\n",
            "import gc, os\n",
            "for _n in ('trainer', 'model', '_b', '_dev', '_out'):\n",
            "    if _n in globals():\n",
            "        del globals()[_n]\n",
            "gc.collect()\n",
            "torch.cuda.empty_cache()\n",
            "_held = torch.cuda.memory_allocated() / 1e9\n",
            "print('GPU held by this kernel after release: %.2f GB' % _held)\n",
            "assert _held < 1.0, 'this kernel still holds %.2f GB - the eval subprocess will OOM' % _held\n",
            "\n",
            "!CUDA_VISIBLE_DEVICES=1 python /kaggle/working/src/scripts/eval_vrsbench.py \\\n",
            "    --task %s%s \\\n" % (vtask, " --answer-format vrsbench" if vtask == "refer" else ""),
            "    --ann-dir {VRS} --images {VRS}/Images_val.zip \\\n",
            "    --model Qwen/Qwen2.5-VL-3B-Instruct --max-pixels %d \\\n" % visual_tokens,
            "    --adapter /kaggle/working/lora_adapter \\\n",
            "    --dtype fp16 --limit 400 --out /kaggle/working/after_lora\n",
            "\n",
            "import json\n",
            "a = json.load(open('/kaggle/working/after_lora.json'))\n",
            "print('\\n' + '=' * 52)\n",
            "if a['task'] == 'vqa':\n",
            "    print('after LoRA : %.1f%% exact   (zero-shot 42.7, base 43.4; published fine-tuned 60.6-60.9)' % a['overall_accuracy'])\n",
            "    for t, d in sorted(a.get('per_type', {}).items()):\n",
            "        print('  %-20s %6.1f%%  majority %5.1f%%' % (t, d['accuracy'], d['majority']))\n",
            "else:\n",
            "    print('after LoRA : Acc@0.5 %.1f%%  mIoU %.3f   (10 m adapter 13.0, base 37.8; GeoChat fine-tuned 39.6)' % (a['acc50'], a['miou']))\n",
            "print('=' * 52)",
        )

    if data == "joint":
        cells[0] = md(
            "# SatQuery - QLoRA on Qwen2.5-VL-3B, on a Kaggle T4 - ONE VQA adapter, both datasets\n",
            "\n",
            "The full-split run. Measured 5 Sep on identical questions: the RSVQA-trained\n",
            "adapter scores 85.5 on RSVQA-LR but only 46.2 on BigEarthNet.txt (majority 47.0),\n",
            "and the BigEarthNet-trained one scores 75.3 there but 62.0 on RSVQA-LR. Neither\n",
            "transfers. The statement names BigEarthNet.txt and the benchmark table needs\n",
            "RSVQA-LR, so this trains one adapter on both: 44,000 RSVQA-LR train questions\n",
            "(of 57k) plus 12,000 BigEarthNet.txt binary/mcq questions from the Lithuania\n",
            "cell, each family in its own harness's templates, scored on both harnesses.\n",
            "\n",
            "## Before you press anything\n",
            "\n",
            "1. **Settings -> GPU hours.** 30 a week. This run needs ~10 of them - the stop is 10.5 h.\n",
            "   **If quota is spent you get a Tesla P100 and nothing here will load.**\n",
            "2. **Accelerator = GPU T4 x2**, **Internet = On**.\n",
            "3. **Save Version -> Save & Run All**, then close the tab. Do not sit on it.\n",
        )
        cells[3:6] = [
            code(
                "# ---- 3. data: RSVQA-LR from Zenodo, BigEarthNet.txt + reBEN imagery from HF ----\n",
                "import os, json\n",
                "RSVQA = '/kaggle/working/rsvqa'\n",
                "os.makedirs(RSVQA, exist_ok=True)\n",
                "base = 'https://zenodo.org/records/6344334/files'\n",
                "for f in ['Images_LR.zip',\n",
                "          'LR_split_train_questions.json', 'LR_split_train_answers.json',\n",
                "          'LR_split_test_questions.json', 'LR_split_test_answers.json']:\n",
                "    p = os.path.join(RSVQA, f)\n",
                "    if not os.path.exists(p) or os.path.getsize(p) < 10000:\n",
                "        !curl -sL -C - -o {p} \"{base}/{f}?download=1\"\n",
                "    print('%-34s %9.1f MB' % (f, os.path.getsize(p) / 1e6))\n",
                "for f in os.listdir(RSVQA):\n",
                "    if f.endswith('.json'):\n",
                "        json.load(open(os.path.join(RSVQA, f)))   # a truncated download dies HERE\n",
                "print('rsvqa split files parse')\n",
                "\n",
                "from huggingface_hub import hf_hub_download, snapshot_download\n",
                "os.makedirs('/kaggle/working/data', exist_ok=True)\n",
                "PARQUET = hf_hub_download('BIFOLD-BigEarthNetv2-0/BigEarthNet.txt',\n",
                "                          'BigEarthNet.txt.parquet', repo_type='dataset',\n",
                "                          local_dir='/kaggle/working/data')\n",
                "print('annotations:', PARQUET, '%.0f MB' % (os.path.getsize(PARQUET) / 1e6))\n",
                "LMDB_DIR = snapshot_download('hackelle/BigEarthNetV2-Lithuania-Summer-LMDB',\n",
                "                             repo_type='dataset', local_dir='/kaggle/working/data/ben')\n",
                "print('imagery    :', LMDB_DIR)",
            ),
            code(
                "# ---- 4. two subsets, one training set ----\n",
                "# 44,000 RSVQA-LR rows (balanced across the four types, strided over all 572\n",
                "# train images) + 12,000 BigEarthNet.txt binary/mcq rows (3,000 patches x 4,\n",
                "# Lithuania/Summer, the v8 mix). 56,000 rows = 3,500 steps of 16, about 9 h.\n",
                "# Each family is wrapped in ITS OWN harness's templates, so the adapter\n",
                "# learns both answer forms and each harness asks in the form it learned.\n",
                "!python /kaggle/working/src/scripts/make_rsvqa_subset.py \\\n",
                "    --data-dir /kaggle/working/rsvqa --limit 44000%s \\\n" % count_flags,
                "    --out /kaggle/working/data/rsvqa_train.jsonl\n",
                "!python /kaggle/working/src/scripts/make_subset.py \\\n",
                "    --parquet /kaggle/working/data/BigEarthNet.txt.parquet \\\n",
                "    --split train --country Lithuania --season Summer \\\n",
                "    --max-patches 3000 --per-patch 4 --types binary,mcq \\\n",
                "    --out /kaggle/working/data/ben_train.jsonl\n",
                "\n",
                "import json, random\n",
                "from scripts.eval_ben_vqa import PROMPTS as BEN_PROMPTS\n",
                "rs = [json.loads(l) for l in open('/kaggle/working/data/rsvqa_train.jsonl')]\n",
                "bn = [json.loads(l) for l in open('/kaggle/working/data/ben_train.jsonl')]\n",
                "for r in bn:\n",
                "    r['input'] = BEN_PROMPTS[r['type']].format(q=r['input'].strip())\n",
                "    r['output'] = str(r['output']).strip().lower()\n",
                "    r['source'] = 'ben'\n",
                "for r in rs:\n",
                "    r['source'] = 'rsvqa'\n",
                "rows = rs + bn\n",
                "random.Random(1337).shuffle(rows)\n",
                "print('rows:', len(rows), '| rsvqa:', len(rs), '| ben:', len(bn))\n",
                "print('rsvqa types:', {t: sum(1 for r in rs if r['type'] == t) for t in {r['type'] for r in rs}})\n",
                "print('ben types  :', {t: sum(1 for r in bn if r['type'] == t) for t in {r['type'] for r in bn}})\n",
                "print('\\nrsvqa example:', json.dumps(rs[0], indent=1)[:300])\n",
                "print('\\nben example  :', json.dumps(bn[0], indent=1)[:400])",
            ),
            code(
                "# ---- 5. two loaders behind one to_rgb ----\n",
                "# RSVQA ids are integers-as-strings and live in Images_LR.zip; BigEarthNet\n",
                "# ids start with S2 and live in the LMDB (RGB picked by band NAME, never\n",
                "# index - and the LMDB also holds S1 entries, keys sorting before S2).\n",
                "import numpy as np\n",
                "from scripts.eval_rsvqa_lr import load_images\n",
                "from scripts.ben_images import BENImages\n",
                "\n",
                "images = load_images(os.path.join(RSVQA, 'Images_LR.zip'), {r['patch_id'] for r in rs})\n",
                "ben = BENImages(LMDB_DIR)\n",
                "print('rsvqa images:', len(images), '| lmdb entries:', ben.n, '| rgb bands:', ben.rgb)\n",
                "missing = [r['patch_id'] for r in rs if r['patch_id'] not in images] + [r['patch_id'] for r in bn if not ben.has(r['patch_id'])]\n",
                "assert not missing, '%d rows have no image, e.g. %s' % (len(missing), missing[:5])\n",
                "\n",
                "def to_rgb(patch_id, size=224):\n",
                "    if patch_id.startswith('S2'):\n",
                "        return ben.get(patch_id, size=size)\n",
                "    return images[patch_id]          # 256 px native, the harness scores at 256\n",
                "\n",
                "# PROVE both sources are not blank before training on them.\n",
                "for name, ids in (('rsvqa', sorted(images)[:100]), ('ben', sorted({r['patch_id'] for r in bn})[:100])):\n",
                "    ok = sum(1 for pid in ids if np.asarray(to_rgb(pid)).std() > 5)\n",
                "    print('%-5s non-black: %d / %d' % (name, ok, len(ids)))\n",
                "    assert ok > len(ids) * 0.5, name + ' loader is producing blank images - FIX THIS BEFORE TRAINING.'\n",
                "display(to_rgb(rs[0]['patch_id'])); display(to_rgb(bn[0]['patch_id']))",
            ),
        ]
        global _JOINT_EVAL
        _JOINT_EVAL = code(
            "# ---- 8. the numbers that matter: both harnesses ----\n",
            "import gc, os\n",
            "for _n in ('trainer', 'model', '_b', '_dev', '_out'):\n",
            "    if _n in globals():\n",
            "        del globals()[_n]\n",
            "gc.collect()\n",
            "torch.cuda.empty_cache()\n",
            "_held = torch.cuda.memory_allocated() / 1e9\n",
            "print('GPU held by this kernel after release: %.2f GB' % _held)\n",
            "assert _held < 1.0, 'this kernel still holds %.2f GB - the eval subprocess will OOM' % _held\n",
            "\n",
            "!CUDA_VISIBLE_DEVICES=1 python /kaggle/working/src/scripts/eval_rsvqa_lr.py \\\n",
            "    --data-dir /kaggle/working/rsvqa --model Qwen/Qwen2.5-VL-3B-Instruct \\\n",
            "    --adapter /kaggle/working/lora_adapter --dtype fp16 --limit 400%s \\\n"
            % count_eval,
            "    --out /kaggle/working/after_lora_rsvqa\n",
            "!CUDA_VISIBLE_DEVICES=1 python /kaggle/working/src/scripts/eval_ben_vqa.py \\\n",
            "    --parquet /kaggle/working/data/BigEarthNet.txt.parquet --lmdb /kaggle/working/data/ben \\\n",
            "    --model Qwen/Qwen2.5-VL-3B-Instruct --adapter /kaggle/working/lora_adapter \\\n",
            "    --dtype fp16 --limit 400 --out /kaggle/working/after_lora_ben\n",
            "\n",
            "import json\n",
            "a = json.load(open('/kaggle/working/after_lora_rsvqa.json'))\n",
            "b = json.load(open('/kaggle/working/after_lora_ben.json'))\n",
            "print('\\n' + '=' * 60)\n",
            "print('RSVQA-LR      : %.1f%% binned  (base 60.75, majority %.1f, prev adapter 85.5)'\n",
            "      % (a['overall_accuracy_binned'], a['majority_baseline_binned']))\n",
            "print('BigEarthNet.txt: %.1f%%          (base 38.9, majority %.1f, prev adapters 46.2 / 75.3)'\n",
            "      % (b['overall_accuracy'], b['majority_baseline']))\n",
            "print('=' * 60)\n",
            "for t, d in sorted(a['per_type'].items()):\n",
            "    print('  rsvqa %-12s %6.1f%%  majority %5.1f%%' % (t, d['accuracy_binned'], d['majority_binned']))\n",
            "for t, d in sorted(b['by_type'].items()):\n",
            "    print('  ben   %-12s %6.1f%%  majority %5.1f%%' % (t, d['accuracy'], d['majority']))",
        )

    if data == "fusion":
        cells[0] = md(
            "# SatQuery - QLoRA on Qwen2.5-VL-3B, on a Kaggle T4 - optical + SAR fusion\n",
            "\n",
            "Fifth recipe: capability 4, optical-SAR joint extraction. The reBEN\n",
            "Lithuania/Summer LMDB holds one Sentinel-1 VV/VH entry per Sentinel-2 patch\n",
            "(measured 5 Sep: 8,775 of each, one-to-one, already in dB). So the\n",
            "BigEarthNet.txt questions are asked over the PAIR - optical tile first, the\n",
            "SAR rendered as (VV, VH, VV-VH) dB on fixed windows second - with a prefix\n",
            "that says which is which. Scored by the BigEarthNet harness in --sar mode,\n",
            "against the optical-only adapters on the identical questions.\n",
            "\n",
            "## Before you press anything\n",
            "\n",
            "1. **Settings -> GPU hours.** 30 a week. Two images per row: budget ~6 of them.\n",
            "   **If quota is spent you get a Tesla P100 and nothing here will load.**\n",
            "2. **Accelerator = GPU T4 x2**, **Internet = On**.\n",
            "3. **Save Version -> Save & Run All**, then close the tab. Do not sit on it.\n",
        )
        cells[4] = code(
            "# ---- 4. cut the fusion subset ----\n",
            "# BigEarthNet.txt binary+mcq rows, Lithuania/Summer, 3,000 patches x 6 =\n",
            "# 18,000 rows (two images per row, so under v8's budget). s1_name rides\n",
            "# along from the parquet; each question gets the harness's SAR prefix and\n",
            "# its own template so train and test ask in one form.\n",
            "!python /kaggle/working/src/scripts/make_subset.py \\\n",
            "    --parquet /kaggle/working/data/BigEarthNet.txt.parquet \\\n",
            "    --split train --country Lithuania --season Summer \\\n",
            "    --max-patches 3000 --per-patch 6 --types binary,mcq \\\n",
            "    --out /kaggle/working/data/subset_fusion.jsonl\n",
            "\n",
            "import json\n",
            "from scripts.eval_ben_vqa import PROMPTS as BEN_PROMPTS, SAR_PREFIX\n",
            "rows = [json.loads(l) for l in open('/kaggle/working/data/subset_fusion.jsonl')]\n",
            "for r in rows:\n",
            "    r['input'] = SAR_PREFIX + BEN_PROMPTS[r['type']].format(q=r['input'].strip())\n",
            "    r['output'] = str(r['output']).strip().lower()\n",
            "assert all(r.get('s1_name') for r in rows), 'rows carry no s1_name - make_subset must keep that column'\n",
            "print('rows:', len(rows), '| patches:', len({r['patch_id'] for r in rows}))\n",
            "print('types:', {t: sum(1 for r in rows if r['type'] == t) for t in {r['type'] for r in rows}})\n",
            "print('\\nexample:', json.dumps(rows[0], indent=2)[:500])",
        )
        cells[5] = code(
            "# ---- 5. the pair loader: optical RGB + SAR pseudo-RGB, one LMDB ----\n",
            "# Both from scripts/ben_images.py, the harness's own decoders: RGB by band\n",
            "# NAME, SAR as (VV, VH, VV-VH) dB on FIXED windows so a lake looks like a\n",
            "# lake on every tile. Both must exist or the row is refused.\n",
            "import numpy as np\n",
            "from scripts.ben_images import BENImages\n",
            "\n",
            "ben = BENImages(LMDB_DIR)\n",
            "print('lmdb entries:', ben.n, '| rgb bands:', ben.rgb, '| sar keys:', ben.s1_keys)\n",
            "s1_of = {r['patch_id']: r['s1_name'] for r in rows}\n",
            "missing = [r['patch_id'] for r in rows if not ben.has(r['patch_id']) or ben.get_sar(r['s1_name']) is None]\n",
            "assert not missing, '%d rows lack an optical or SAR entry, e.g. %s' % (len(missing), missing[:3])\n",
            "\n",
            "def to_rgb(patch_id, size=224):\n",
            "    return [ben.get(patch_id, size=size), ben.get_sar_rgb(s1_of[patch_id], size=size)]\n",
            "\n",
            "present = sorted(s1_of)[:200]\n",
            "ok = sum(1 for pid in present if all(np.asarray(im).std() > 5 for im in to_rgb(pid)))\n",
            "print('non-black pairs: %d / %d' % (ok, len(present)))\n",
            "assert ok > len(present) * 0.5, 'a loader is producing blank images - FIX THIS BEFORE TRAINING.'\n",
            "a, b = to_rgb(present[0]); display(a); display(b)",
        )
        global _FUSION_EVAL
        _FUSION_EVAL = code(
            "# ---- 8. the number that matters: fusion, measured ----\n",
            "import gc, os\n",
            "for _n in ('trainer', 'model', '_b', '_dev', '_out'):\n",
            "    if _n in globals():\n",
            "        del globals()[_n]\n",
            "gc.collect()\n",
            "torch.cuda.empty_cache()\n",
            "_held = torch.cuda.memory_allocated() / 1e9\n",
            "print('GPU held by this kernel after release: %.2f GB' % _held)\n",
            "assert _held < 1.0, 'this kernel still holds %.2f GB - the eval subprocess will OOM' % _held\n",
            "\n",
            "!CUDA_VISIBLE_DEVICES=1 python /kaggle/working/src/scripts/eval_ben_vqa.py --sar \\\n",
            "    --parquet /kaggle/working/data/BigEarthNet.txt.parquet --lmdb /kaggle/working/data/ben \\\n",
            "    --model Qwen/Qwen2.5-VL-3B-Instruct --adapter /kaggle/working/lora_adapter \\\n",
            "    --dtype fp16 --limit 400 --out /kaggle/working/after_lora\n",
            "\n",
            "import json\n",
            "a = json.load(open('/kaggle/working/after_lora.json'))\n",
            "print('\\n' + '=' * 60)\n",
            "print('optical+SAR adapter: %.1f%%   (majority %.1f; optical-only adapters on the same questions: 46.2 / 75.3)'\n",
            "      % (a['overall_accuracy'], a['majority_baseline']))\n",
            "print('=' * 60)\n",
            "for t, d in sorted(a['by_type'].items()):\n",
            "    print('  %-8s %6.1f%%  majority %5.1f%%' % (t, d['accuracy'], d['majority']))",
        )

    if data == "ground-ref":
        cells[4] = code(
            "# ---- 4. cut the REFERENCE-heavy grounding subset ----\n",
            "# The first grounding adapter (v3, 5 Sep) learned the point form (Acc@0.5\n",
            "# 85.5%) and not the referring form (15.0%, the whole-image floor) - on\n",
            "# 6,878 reference rows. This cut takes every reference row the cap allows\n",
            "# and a quarter as many point rows, so the harder form gets the budget.\n",
            *ground_ref_cut_comment(ref_rows),
            "!python /kaggle/working/src/scripts/make_subset.py \\\n",
            "    --parquet /kaggle/working/data/BigEarthNet.txt.parquet \\\n",
            "    --split train %s \\\n" % ground_ref_cell_flags(ref_rows),
            "    --max-patches %d --per-patch 14 --types \"bounding box\" \\\n" % ground_ref_max_patches(ref_rows),
            "    --out /kaggle/working/data/subset_ground_all.jsonl\n",
            "\n",
            "import json, random\n",
            "from scripts.eval_ground import PROMPT_SUFFIX, parse_gold\n",
            "allrows = [json.loads(l) for l in open('/kaggle/working/data/subset_ground_all.jsonl')]\n",
            "ref = [r for r in allrows if r['category'] == 'reference']\n",
            "pt = [r for r in allrows if r['category'] == 'point']\n",
            "random.Random(1337).shuffle(ref); random.Random(1338).shuffle(pt)\n",
            "ref, pt = ref[:%d], pt[:4000]\n" % ref_rows,
            "rows = ref + pt\n",
            "random.Random(1339).shuffle(rows)\n",
            "for r in rows:\n",
            "    r['input'] = r['input'] + PROMPT_SUFFIX\n",
            "bad = [r for r in rows if parse_gold(r['output']) is None]\n",
            "assert not bad, '%d rows have a gold box that does not parse' % len(bad)\n",
            "print('rows:', len(rows), '| reference:', len(ref), '| point:', len(pt), '| patches:', len({r['patch_id'] for r in rows}))\n",
            "print('\\nexample:', json.dumps(rows[0], indent=2)[:500])",
        )

    if vision_lora:
        _cell6_head = (
            "# ---- 6. model + LoRA, on the LANGUAGE side AND the VISION TOWER ----\n",
            "# The language-only adapters learned every form that a description of the\n",
            "# picture can answer, and not the one that needs the picture read more\n",
            "# finely: referring-expression grounding sat at 15.0% then 20.5% after a\n",
            "# reference-heavy pass, against a whole-image box at 12.5%. Qwen2.5-VL's\n",
            "# vision tower was trained from scratch on web images and frozen during\n",
            "# Qwen's own SFT; it has never been asked to separate pasture from arable\n",
            "# land at 10 m. This run puts LoRA on every vision block (qkv, proj, the\n",
            "# SwiGLU MLP) and on the patch merger that hands tokens to the LLM, as\n",
            "# well as the language projections, so the picture side can move.\n",
            "#\n",
            "# Non-reentrant checkpointing is REQUIRED for this: the first vision block\n",
            "# takes the patch embedding, which carries no grad, and reentrant\n",
            "# checkpointing returns None for every parameter inside such a block.\n",
            "# The 'never reached' guard in cell 7 is what catches it if this regresses.\n",
        )
    else:
        _cell6_head = (
            "# ---- 6. model + LoRA, on the LANGUAGE side only ----\n",
            "# Qwen2.5-VL's vision tower was trained from scratch and frozen during Qwen's\n",
            "# own SFT, so LLM-only LoRA leaves it as shipped. RSHallu reached 92.11 on\n",
            "# this exact benchmark that way, so it is sufficient. Vision-tower LoRA is the\n",
            "# NEXT lever if small-object presence lags, not the first one.\n",
        )
    if vision_lora:
        _cell6_targets = (
            "import re\n",
            "WANT = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')\n",
            "VIS_WANT = ('qkv', 'proj', 'gate_proj', 'up_proj', 'down_proj')\n",
            "lang = sorted({n for n, _ in model.named_modules()\n",
            "               if n.split('.')[-1] in WANT and not n.startswith('visual')\n",
            "               and '.visual.' not in n})\n",
            "vis = sorted({n for n, m in model.named_modules()\n",
            "              if ('.visual.blocks.' in n and n.split('.')[-1] in VIS_WANT)\n",
            "              or ('.visual.merger.' in n and isinstance(m, torch.nn.Linear))})\n",
            "n_blocks = len({n.split('.visual.blocks.')[1].split('.')[0] for n in vis if '.visual.blocks.' in n})\n",
            "n_merger = sum('.visual.merger.' in n for n in vis)\n",
            "print('LoRA target modules: %d language + %d vision (%d blocks x %d, merger %d)'\n",
            "      % (len(lang), len(vis), n_blocks, len(VIS_WANT), n_merger))\n",
            "assert n_blocks == 32 and len(vis) == n_blocks * len(VIS_WANT) + n_merger and n_merger == 2, (\n",
            "    'vision tower module names do not match Qwen2.5-VL as expected: %s' % vis[:6])\n",
            "targets = lang + vis\n",
            "print('  vision tower touched:', any('visual' in t for t in targets), '(must be True)')\n",
            "print('  e.g.', targets[:2], vis[:2], vis[-2:])\n",
        )
    else:
        _cell6_targets = (
            "import re\n",
            "WANT = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')\n",
            "targets = sorted({n for n, _ in model.named_modules()\n",
            "                  if n.split('.')[-1] in WANT and not n.startswith('visual')\n",
            "                  and '.visual.' not in n})\n",
            "print('LoRA target modules:', len(targets))\n",
            "print('  vision tower touched:', any('visual' in t for t in targets), '(must be False)')\n",
            "print('  e.g.', targets[:3])\n",
        )
    cells.append(code(
        *_cell6_head,
        "#\n",
        "# fp16, NOT 4-bit - and that is a deliberate reversal.\n",
        "#\n",
        "# 4-bit was chosen for the office 4060's 8.6 GB. This T4 has 15.6 GB, and\n",
        "# Qwen2.5-VL's layer shapes do not hit bitsandbytes' fast 4-bit kernel:\n",
        "# PLAN section 9 already measured the fallback ('inner dimension (3420) is\n",
        "# not aligned for fast kernel'). At inference that was merely slow. In\n",
        "# TRAINING the fallback dequantises the full weight matrix every forward\n",
        "# pass, and the run died in _dequant_linear_fallback with\n",
        "# CUBLAS_STATUS_NOT_INITIALIZED - which is out-of-memory in disguise.\n",
        "#\n",
        "# fp16 removes bitsandbytes from the training path entirely and is faster,\n",
        "# because there is no dequantisation per step. Budget on 15.6 GB:\n",
        "#     weights    7.51 GB  MEASURED on a 4060, not derived. The older\n",
        "#                         ~6.2 GB here counted the LLM alone (3.09 B x 2).\n",
        "#                         The checkpoint is 3.76 B params including the\n",
        "#                         vision tower, and all of it loads. Harmless on a\n",
        "#                         15.6 GB T4; fatal on 8 GB, which is why this is\n",
        "#                         not on the office box - peak reserved there was\n",
        "#                         8.12 GB of 8.59 GB at only 64 visual tokens.\n",
        "#     LoRA+Adam ~0.2 GB   (15 M trainable)\n",
        "#     activations 1-2 GB  (checkpointed, batch 1)\n",
        "import torch\n",
        "from peft import LoraConfig, get_peft_model\n",
        "from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration\n",
        "\n",
        "BASE = 'Qwen/Qwen2.5-VL-3B-Instruct'\n",
        "# Pin to GPU 0. Kaggle gives TWO T4s, and device_map='auto' would split the\n",
        "# model across both, adding a cross-device transfer to every forward pass.\n",
        "# Two cards only pay off with real data parallelism, which a notebook\n",
        "# cannot launch.\n",
        "model = Qwen2_5_VLForConditionalGeneration.from_pretrained(\n",
        "    BASE, device_map={'': 0}, torch_dtype=torch.float16)\n",
        "processor = AutoProcessor.from_pretrained(\n",
        "    BASE, min_pixels=64 * 28 * 28, max_pixels=%d * 28 * 28)\n" % visual_tokens,
        "print('weights on GPU: %.2f GB' % (torch.cuda.memory_allocated() / 1e9))\n",
        "\n",
        "# Name the target modules EXPLICITLY. Passing bare names like 'q_proj' lets\n",
        "# peft match the vision tower too, and then you are training something you\n",
        "# meant to leave frozen - silently. The vision set, when it is wanted, is\n",
        "# named just as explicitly and counted against the architecture.\n",
        *_cell6_targets,
        "\n",
        "cfg = LoraConfig(r=%d, lora_alpha=32, lora_dropout=0.1, bias='none',\n" % lora_rank,
        "                 task_type='CAUSAL_LM', target_modules=targets)\n",
        "model = get_peft_model(model, cfg)\n",
        "model.print_trainable_parameters()",
    ))

    cells.append(code(
        "# ---- 7. train, with a deliberate stop and a real resume ----\n",
        "# THE COLLATION BUG THAT COST A RUN, recorded so it is not repeated:\n",
        "#\n",
        "#     item = {k: v[0] for k, v in enc.items()}      # WRONG\n",
        "#\n",
        "# For Qwen2.5-VL, pixel_values is ALREADY flattened across patches - shape\n",
        "# [n_patches, dim], not [batch, ...]. Indexing [0] keeps patch zero and\n",
        "# throws away the other 255, so the vision tower receives one patch and\n",
        "# dies inside its own reshape: 'shape [0, 4, -1] is invalid for input of\n",
        "# size 1280' (1280 = the vision hidden dim x exactly one patch).\n",
        "#\n",
        "# The processor must be called ONCE PER BATCH in a collate_fn, and its\n",
        "# output passed through untouched.\n",
        "import glob, os, time\n",
        "import torch\n",
        "from torch.utils.data import Dataset\n",
        "from transformers import Trainer, TrainingArguments, TrainerCallback\n",
        "\n",
        "IMAGE_TOKEN = '<|image_pad|>'\n",
        "IMG_ID = processor.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)\n",
        "PAD_ID = processor.tokenizer.pad_token_id\n",
        "\n",
        "class BENText(Dataset):\n",
        "    \"\"\"Returns raw parts. All tensor work happens in the collator.\"\"\"\n",
        "    def __init__(self, rows):\n",
        "        self.rows = rows\n",
        "    def __len__(self):\n",
        "        return len(self.rows)\n",
        "    def __getitem__(self, i):\n",
        "        r = self.rows[i]\n",
        "        return {'pid': r['patch_id'], 'q': r['input'], 'a': str(r['output'])}\n",
        "\n",
        "def collate(batch):\n",
        "    texts, prompts, images = [], [], []\n",
        "    for b in batch:\n",
        "        # to_rgb returns ONE image for VQA rows and a (pre, post) pair for\n",
        "        # change rows; the chat template gets one image slot per picture and\n",
        "        # the processor gets them flat, in order. Same collator, both recipes.\n",
        "        ims = to_rgb(b['pid'])\n",
        "        ims = list(ims) if isinstance(ims, (list, tuple)) else [ims]\n",
        "        msgs = [{'role': 'user', 'content': [{'type': 'image'} for _ in ims]\n",
        "                                            + [{'type': 'text', 'text': b['q']}]}]\n",
        "        p = processor.apply_chat_template(msgs, tokenize=False,\n",
        "                                          add_generation_prompt=True)\n",
        "        prompts.append(p)\n",
        "        texts.append(p + b['a'] + '<|im_end|>')\n",
        "        images.append(ims)\n",
        "\n",
        "    flat = [im for ims in images for im in ims]\n",
        "    enc = processor(text=texts, images=flat, return_tensors='pt', padding=True)\n",
        "    labels = enc['input_ids'].clone()\n",
        "    labels[labels == PAD_ID] = -100\n",
        "    labels[labels == IMG_ID] = -100          # never predict image placeholders\n",
        "\n",
        "    # Mask the PROMPT too, so loss lands only on the answer. Without this the\n",
        "    # model mostly learns to echo the question back, and the answers here are\n",
        "    # one character long - the signal would be swamped.\n",
        "    for i, p in enumerate(prompts):\n",
        "        n = processor(text=[p], images=images[i],\n",
        "                      return_tensors='pt')['input_ids'].shape[1]\n",
        "        labels[i, :n] = -100\n",
        "\n",
        "    enc['labels'] = labels\n",
        "    return enc\n",
        "\n",
        "# Hold out 400 rows for an eval loss every 250 steps. Cheap (400 forward\n",
        "# passes, no generation) and it is the curve a train loss cannot give: a\n",
        "# train loss that keeps falling while eval loss rises is memorisation, which\n",
        "# is exactly how the BigEarthNet adapter's 100% season/country cells arose.\n",
        "val_rows, rows = rows[-400:], rows[:-400]\n",
        "ds, ds_val = BENText(rows), BENText(val_rows)\n",
        "print('train rows:', len(ds), '| held-out rows:', len(ds_val))\n",
        "print('rows:', len(ds))\n",
        "\n",
        "# Prove the collator produces sane tensors BEFORE committing 10 hours to it.\n",
        "_b = collate([ds[0], ds[1]])\n",
        "for k, v in _b.items():\n",
        "    print('   %-18s %s' % (k, tuple(v.shape)))\n",
        "_sup = int((_b['labels'] != -100).sum())\n",
        "print('supervised tokens in 2 examples:', _sup)\n",
        "assert _sup > 0, 'every label is masked - the model would have nothing to learn'\n",
        "assert _b['pixel_values'].shape[0] > 4, (\n",
        "    'pixel_values has %d patches - the image was destroyed before it reached '\n",
        "    'the vision tower' % _b['pixel_values'].shape[0])\n",
        "\n",
        "# Grad plumbing goes on BEFORE the smoke test, or the test is not testing\n",
        "# what training will actually do.\n",
        "model.enable_input_require_grads()\n",
        "model.config.use_cache = False\n",
        "GC_KWARGS = %s\n" % ("{'use_reentrant': False}" if vision_lora else "None"),
        "model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=GC_KWARGS)\n",
        "\n",
        "# ONE REAL FORWARD + BACKWARD before committing ten hours. Two runs have\n",
        "# now died inside the first training step, so make that cost five seconds\n",
        "# instead of a full Trainer setup - and print peak memory, so a CUDA error\n",
        "# can be READ as out-of-memory rather than guessed at.\n",
        "torch.cuda.reset_peak_memory_stats()\n",
        "model.train()\n",
        "_dev = {k: v.to(model.device) for k, v in _b.items()}\n",
        "_out = model(**_dev)\n",
        "_out.loss.backward()\n",
        "print('SMOKE: loss %.4f | peak %.2f GB of %.1f GB'\n",
        "      % (_out.loss.item(), torch.cuda.max_memory_allocated() / 1e9,\n",
        "         torch.cuda.get_device_properties(0).total_memory / 1e9))\n",
        "assert torch.isfinite(_out.loss), 'loss is not finite on the very first batch'\n",
        "\n",
        "# A finite loss is NOT evidence that anything is learning. Check that the\n",
        "# backward actually reached the adapters, because the run-7 log carried\n",
        "#     checkpoint.py:232 UserWarning: None of the inputs have\n",
        "#     requires_grad=True. Gradients will be None\n",
        "# which is exactly what a silently-frozen model looks like. It is expected\n",
        "# from the FROZEN vision tower and harmless there; it is fatal if it is the\n",
        "# language layers, and the warning does not say which.\n",
        "#\n",
        "# THE DISTINCTION THAT MATTERS, and that the first version of this check\n",
        "# got wrong - it counted a zero gradient as a dead one and stopped the run\n",
        "# at 'only 252 of 504 LoRA tensors got a gradient':\n",
        "#\n",
        "#     grad is None      -> backward never reached the tensor. A real bug.\n",
        "#     grad is all zeros -> it was reached. At step 0, for lora_A, CORRECT.\n",
        "#\n",
        "# peft initialises lora_A from a random draw and lora_B to ZEROS, so the\n",
        "# adapter contributes nothing before the first step and the fine-tune\n",
        "# starts exactly at the base model. Then, with out = base(x) + B @ (A @ x),\n",
        "#     dL/dB = g @ (A @ x).T      nonzero\n",
        "#     dL/dA = (B.T @ g) @ x.T    EXACTLY zero, because B is zeros\n",
        "# so half the tensors carry a zero gradient on the first backward and only\n",
        "# the first. 504 = 252 target modules x (lora_A, lora_B), which is why the\n",
        "# count was exactly half. Proved in tests/test_lora_first_step_grads.py.\n",
        "#\n",
        "# So: refuse on None, refuse if nothing at all is nonzero, and refuse if\n",
        "# any zero is on the B side - a zero lora_B gradient really would mean the\n",
        "# adapter is off the path.\n",
        "_lora = [(n, p) for n, p in model.named_parameters()\n",
        "         if p.requires_grad and 'lora_' in n]\n",
        "_none = [n for n, p in _lora if p.grad is None]\n",
        "_zero = [n for n, p in _lora if p.grad is not None and p.grad.abs().sum() == 0]\n",
        "_live = [n for n, p in _lora if p.grad is not None and p.grad.abs().sum() > 0]\n",
        "print('LoRA tensors: %d total | %d nonzero | %d zero | %d never reached'\n",
        "      % (len(_lora), len(_live), len(_zero), len(_none)))\n",
        "assert _lora, 'no trainable LoRA parameters at all - get_peft_model did nothing'\n",
        "assert not _none, (\n",
        "    'backward never reached %d LoRA tensors, e.g. %s. They are not on the '\n",
        "    'path the loss flows through, and training would run for hours changing '\n",
        "    'nothing.' % (len(_none), _none[:3]))\n",
        "assert _live, 'every LoRA gradient is zero - nothing would move on step one'\n",
        "_bad = [n for n in _zero if 'lora_A' not in n]\n",
        "assert not _bad, (\n",
        "    'a zero gradient on the B side is not initialisation, it is a dead '\n",
        "    'adapter: %s' % _bad[:3])\n",
        "\n",
        "model.zero_grad(set_to_none=True)\n",
        "del _out, _dev\n",
        "torch.cuda.empty_cache()\n",
        "torch.cuda.reset_peak_memory_stats()\n",
        "\n",
        "class TimeStop(TrainerCallback):\n",
        "    \"\"\"Stop on OUR schedule, not Kaggle's.\n",
        "\n",
        "    A session killed at 12 h loses whatever it was writing. Stopping early and\n",
        "    saving deliberately is the difference between a resumable run and a lost\n",
        "    day.\n",
        "    \"\"\"\n",
        "    def on_step_end(self, args, state, control, **kw):\n",
        "        if time.time() - T_START > TRAIN_HOURS * 3600:\n",
        "            print('\\nTIMESTOP at step %d - saving and exiting cleanly.' % state.global_step)\n",
        "            control.should_save = True\n",
        "            control.should_training_stop = True\n",
        "        return control\n",
        "\n",
        "cks = sorted(glob.glob('/kaggle/input/*/ckpt/checkpoint-*'),\n",
        "             key=lambda p: int(p.rsplit('-', 1)[1]))\n",
        "resume = cks[-1] if cks else None\n",
        "print('RESUME FROM:', resume or 'none - fresh start')\n",
        "\n",
        "args = TrainingArguments(\n",
        "    output_dir='/kaggle/working/ckpt',\n",
        "    per_device_train_batch_size=1, gradient_accumulation_steps=16,\n",
        "    num_train_epochs=1, learning_rate=1e-4, lr_scheduler_type='cosine',\n",
        "    # adamw_torch, not paged_adamw_8bit: bitsandbytes is out of this run\n",
        "    # entirely now, and 15 M trainable params make optimiser memory a\n",
        "    # rounding error either way.\n",
        "    warmup_ratio=0.01, optim='adamw_torch', logging_steps=10,\n",
        "    eval_strategy='steps', eval_steps=250, per_device_eval_batch_size=1,\n",
        "    save_strategy='steps', save_steps=100, save_total_limit=2,\n",
        "    fp16=True, gradient_checkpointing=True, max_grad_norm=1.0,\n",
        "    gradient_checkpointing_kwargs=GC_KWARGS,\n",
        "    dataloader_num_workers=2, seed=1337, report_to='none',\n",
        "    remove_unused_columns=False, label_names=['labels'])\n",
        "\n",
        "# THE GATE THAT RUN 7 NEEDED. n_gpu is read off device_count() when\n",
        "# TrainingArguments is constructed, and anything above 1 means Trainer will\n",
        "# wrap the model in nn.DataParallel and replicate 7.5 GB of weights onto\n",
        "# GPU 0. Refuse here, by name, rather than discover it in a backward pass\n",
        "# forty minutes into a session.\n",
        "assert args.n_gpu == 1, (\n",
        "    'n_gpu is %d, so Trainer will wrap this in nn.DataParallel and OOM in '\n",
        "    'backward exactly as run 7 did. CUDA_VISIBLE_DEVICES in cell 1 is what '\n",
        "    'prevents it, and it only takes effect on a fresh session.' % args.n_gpu)\n",
        "\n",
        "# ---- memory probe -------------------------------------------------\n",
        "# Kept now that the OOM is understood, because it is the instrument that\n",
        "# says whether the fix held: the first six micro-batches print their real\n",
        "# shapes and peak. Print BEFORE the step as well as after - run 7's probe\n",
        "# printed only on the way out, so dying inside micro-batch 0 produced no\n",
        "# measurement at all and the diagnosis had to come from a stray warning.\n",
        "print(\"gradient checkpointing live:\", model.is_gradient_checkpointing)\n",
        "print(\"use_cache               :\", model.config.use_cache)\n",
        "print(\"weights resident        : %.2f GB\" % (torch.cuda.memory_allocated() / 1e9))\n",
        "\n",
        "VISION_LR_MULT = %s\n" % vision_lr_mult,
        "\n",
        "class ProbeTrainer(Trainer):\n",
        "    n = 0\n",
        "    def create_optimizer(self):\n",
        "        # Two rates over one LoRA. Handing AdamW two param groups is\n",
        "        # enough: get_scheduler reads each group's own initial_lr, so\n",
        "        # the one cosine decays both peaks and the ratio holds to the\n",
        "        # last step. scripts/lr_groups.py refuses an empty group, which\n",
        "        # is what a stale name filter would produce.\n",
        "        if VISION_LR_MULT != 1.0 and self.optimizer is None:\n",
        "            from scripts.lr_groups import describe, param_groups\n",
        "            groups = param_groups(self.model.named_parameters(),\n",
        "                                  self.args.learning_rate, VISION_LR_MULT)\n",
        "            try:\n",
        "                cls, kw = Trainer.get_optimizer_cls_and_kwargs(self.args, self.model)\n",
        "            except TypeError:\n",
        "                cls, kw = Trainer.get_optimizer_cls_and_kwargs(self.args)\n",
        "            kw.pop('lr', None)\n",
        "            self.optimizer = cls(groups, **kw)\n",
        "            print('optimizer groups |', describe(groups), flush=True)\n",
        "        return super().create_optimizer()\n",
        "    def training_step(self, model, inputs, *a, **kw):\n",
        "        first = ProbeTrainer.n < 6\n",
        "        if first:\n",
        "            torch.cuda.reset_peak_memory_stats()\n",
        "            ids = inputs.get(\"input_ids\")\n",
        "            pix = inputs.get(\"pixel_values\")\n",
        "            if ProbeTrainer.n == 0:\n",
        "                lrs = [(g.get('name', i), g['lr'])\n",
        "                       for i, g in enumerate(self.optimizer.param_groups)]\n",
        "                print('live param-group rates:', lrs, flush=True)\n",
        "                assert len(lrs) == (2 if VISION_LR_MULT != 1.0 else 1) or VISION_LR_MULT == 1.0, \\\n",
        "                    'vision lr multiplier was set but the optimizer has %d groups' % len(lrs)\n",
        "            print(\"micro %d IN  | seq %s | pixel_values %s | alloc %.2f GB\"\n",
        "                  % (ProbeTrainer.n,\n",
        "                     tuple(ids.shape) if ids is not None else None,\n",
        "                     tuple(pix.shape) if pix is not None else None,\n",
        "                     torch.cuda.memory_allocated() / 1e9), flush=True)\n",
        "        out = super().training_step(model, inputs, *a, **kw)\n",
        "        if first:\n",
        "            print(\"micro %d OUT | alloc %.2f GB | peak %.2f GB\"\n",
        "                  % (ProbeTrainer.n,\n",
        "                     torch.cuda.memory_allocated() / 1e9,\n",
        "                     torch.cuda.max_memory_allocated() / 1e9), flush=True)\n",
        "        ProbeTrainer.n += 1\n",
        "        return out\n",
        "\n",
        "trainer = ProbeTrainer(model=model, args=args, train_dataset=ds, eval_dataset=ds_val,\n",
        "                  data_collator=collate, callbacks=[TimeStop()])\n",
        "trainer.train(resume_from_checkpoint=resume)\n",
        "\n",
        "model.save_pretrained('/kaggle/working/lora_adapter')\n",
        "processor.save_pretrained('/kaggle/working/lora_adapter')\n",
        "print('adapter saved to /kaggle/working/lora_adapter')\n",
        "# log_history is plain data and always survives; the progress widget does not.\n",
        "print('LOSS HISTORY:', [e for e in trainer.state.log_history if 'loss' in e][-10:])\n",
        "print('EVAL LOSS   :', [(e['step'], round(e['eval_loss'], 4)) for e in trainer.state.log_history if 'eval_loss' in e])",
    ))

    cells.append(code(
        "# ---- 8. the number that matters: did it beat 60.8%? ----\n",
        "# Same harness, same 400 questions, same binned protocol as the baseline.\n",
        "# A delta measured any other way is not a delta.\n",
        "#\n",
        "# --dtype fp16, because v8 trained for 10 hours and then died HERE: the eval\n",
        "# script built a BitsAndBytesConfig, transformers looked up bitsandbytes'\n",
        "# package metadata, and the kernel has none (cell 2 never installs it).\n",
        "# The adapter was scored on the office 4060 instead - nf4, the baseline's\n",
        "# own loader: 62.0% binned vs 60.75%. PLAN section 9 has the breakdown.\n",
        "#\n",
        "# THE EVAL RUNS ON GPU 1. Kaggle gives two T4s; training is pinned to GPU 0\n",
        "# and two completed runs (cdvqa v2, ground v3) then died in the eval with\n",
        "# CUBLAS_STATUS_ALLOC_FAILED on GPU 0 even after this kernel freed and\n",
        "# asserted under 1 GB. GPU 1 is untouched by construction. The release\n",
        "# below stays, as hygiene and as a printed measurement.\n",
        "# FREE THE CARD FIRST. The eval is a subprocess, and this kernel still held\n",
        "# the trained model: rsvqa v2 finished all 1,250 steps, saved the adapter,\n",
        "# and then the eval OOMed loading a second 7.5 GB copy next to the first -\n",
        "# 'Process 23 has 8.93 GiB memory in use' was THIS process. Delete every\n",
        "# reference, collect, empty the cache, and refuse to launch if the card is\n",
        "# not actually free, because a subprocess cannot tell us why it died.\n",
        "import gc, os, zipfile\n",
        "for _n in ('trainer', 'model', '_b', '_dev', '_out'):\n",
        "    if _n in globals():\n",
        "        del globals()[_n]\n",
        "gc.collect()\n",
        "torch.cuda.empty_cache()\n",
        "_held = torch.cuda.memory_allocated() / 1e9\n",
        "print('GPU held by this kernel after release: %.2f GB' % _held)\n",
        "assert _held < 1.0, (\n",
        "    'this kernel still holds %.2f GB - something else references the model, '\n",
        "    'and the eval subprocess will OOM exactly as rsvqa v2 did' % _held)\n",
        "os.makedirs('/kaggle/working/rsvqa', exist_ok=True)\n",
        "base = 'https://zenodo.org/records/6344334/files'\n",
        "for f in ['Images_LR.zip', 'LR_split_test_questions.json', 'LR_split_test_answers.json']:\n",
        "    p = '/kaggle/working/rsvqa/' + f\n",
        "    if not os.path.exists(p) or os.path.getsize(p) < 10000:\n",
        "        !curl -sL -C - -o {p} \"{base}/{f}?download=1\"\n",
        "    print('%-34s %9.1f MB' % (f, os.path.getsize(p) / 1e6))\n",
        "\n",
        "!CUDA_VISIBLE_DEVICES=1 python /kaggle/working/src/scripts/eval_rsvqa_lr.py \\\n",
        "    --data-dir /kaggle/working/rsvqa \\\n",
        "    --model Qwen/Qwen2.5-VL-3B-Instruct \\\n",
        "    --adapter /kaggle/working/lora_adapter \\\n",
        "    --dtype fp16 --limit 400%s --out /kaggle/working/after_lora\n"
        % count_eval,
        "\n",
        "import json\n",
        "a = json.load(open('/kaggle/working/after_lora.json'))\n",
        "print('\\n' + '=' * 52)\n",
        "print('baseline (office box) : 60.8%%  vs majority 59.2%%')\n",
        "print('after LoRA            : %.1f%%  vs majority %.1f%%'\n",
        "      % (a['overall_accuracy_binned'], a['majority_baseline_binned']))\n",
        "print('delta over majority   : %+.1f  ->  %+.1f'\n",
        "      % (1.5, a['beats_majority_by_binned']))\n",
        "print('=' * 52)\n",
        "for t, d in sorted(a['per_type'].items()):\n",
        "    print('  %-12s %6.1f%%  majority %5.1f%%  %+6.1f'\n",
        "          % (t, d['accuracy_binned'], d['majority_binned'],\n",
        "             d['accuracy_binned'] - d['majority_binned']))",
    ))

    if data == "cdvqa":
        cells[-1] = _CDVQA_EVAL
    if data in ("ground", "ground-ref"):
        cells[-1] = _GROUND_EVAL
    if data in ("vrsbench-vqa", "vrsbench-ref"):
        cells[-1] = _VRSBENCH_EVAL
    if data == "joint":
        cells[-1] = _JOINT_EVAL
    if data == "fusion":
        cells[-1] = _FUSION_EVAL

    cells.append(md(
        "## If it stopped at TIMESTOP\n",
        "\n",
        "1. Output tab -> download `/kaggle/working/ckpt`, or publish it as a Kaggle\n",
        "   Dataset directly from the notebook output.\n",
        "2. Attach that dataset to this notebook (**Add Input**).\n",
        "3. Run All again. Cell 7 finds it and resumes from the last step with the\n",
        "   optimizer and LR schedule intact.\n",
        "\n",
        "## What to report\n",
        "\n",
        "* Cell 1's device line.\n",
        "* Cell 5's `non-black images` count - if that assert fired, nothing after it\n",
        "  is meaningful.\n",
        "* Cell 6's `vision tower touched: %s`.\n" % vision_lora,
        "* Cell 7's final loss history.\n",
        "* **Cell 8's table.** That is the Sprint I number: 60.8% -> what?",
    ))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            # This block is a REQUEST KAGGLE IGNORES ON A CLI PUSH. Keep it
            # (it is what the UI reads), but do not expect it to work alone.
            #
            # MEASURED 3 Sep 2026, one account, one evening, quota untouched at
            # 30:00/30 hrs the whole time:
            #     v1  CLI push  -> P100      v2  UI Save & Run All -> T4
            #     v3  CLI push  -> P100      v4  CLI push          -> P100
            # v4 carried a kaggle metadata block BYTE-IDENTICAL to the pundit
            # notebooks. Still P100. So the accelerator is a session setting
            # only the UI sets, and no API can: ApiSaveKernelRequest carries
            # enable_gpu/enable_tpu and no GPU-type field at all.
            #
            # This CORRECTS D:\punditinal.py, which concluded "it tracks
            # quota - check the Settings page FIRST". That rule cost two runs
            # here before the quota page came back full. Quota does gate
            # whether a T4 EXISTS for you; it is not what picks the card on a
            # push.
            #
            # So: push code with the CLI, START RUNS FROM THE UI
            # (Edit -> Accelerator -> GPU T4 x2 -> Save & Run All).
            "kaggle": {"accelerator": "nvidiaTeslaT4",
                       "isInternetEnabled": True,
                       "language": "python",
                       "sourceType": "notebook",
                       "isGpuEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def main():
    # --user matters because the run is split ACROSS ACCOUNTS. One teammate
    # trains until TIMESTOP, publishes the checkpoint as a dataset, and the
    # next resumes from it on their own weekly quota. Each push therefore has
    # to carry the pushing account's own id, or the CLI rejects it.
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="vinayakameta",
                    help="Kaggle username of the account that will PUSH this "
                         "notebook (not necessarily the one that started the run)")
    ap.add_argument("--resume-from", default=None,
                    help="Kaggle dataset slug holding the previous leg's "
                         "checkpoint, e.g. someuser/satquery-ckpt-leg1. It is "
                         "attached at /kaggle/input and cell 7 resumes from it.")
    ap.add_argument("--resume-kernel", default=None,
                    help="Kaggle kernel slug whose OUTPUT holds ckpt/checkpoint-*, e.g. "
                         "vinayakameta/satquery-qlora-joint. Mounted at /kaggle/input, "
                         "same place the resume glob looks; no dataset upload needed.")
    ap.add_argument("--epochs", type=int, default=1,
                    help="num_train_epochs. Use 2 with --resume-from/--resume-kernel for a "
                         "second pass: a resume with epochs=1 trains zero steps.")
    ap.add_argument("--data", choices=["ben", "rsvqa", "cdvqa", "ground", "ground-ref", "joint", "fusion",
                                       "vrsbench-vqa", "vrsbench-ref"], default="ben",
                    help="ben = BigEarthNet.txt + reBEN imagery (v8 recipe); "
                         "rsvqa = RSVQA-LR's own train split. Each variant is its "
                         "own Kaggle kernel so their versions never interleave.")
    ap.add_argument("--eval-only", default=None, metavar="USER/SLUG",
                    help="build a kernel that TRAINS NOTHING: it mounts the "
                         "lora_adapter from that kernel's output and scores it "
                         "on the full test split. Training is no longer the "
                         "scarce resource - one evaluation card is - so this "
                         "turns a spare account into a scoring machine. Slug "
                         "takes an -eval suffix.")
    ap.add_argument("--eval-limit", type=int, default=0,
                    help="rows to score with --eval-only. 0, the default, is the "
                         "whole split - which is the point of doing it here.")
    ap.add_argument("--eval-dtype", choices=["nf4", "int8", "fp16"], default="fp16",
                    help="precision for --eval-only. fp16 by default because a "
                         "Kaggle kernel has no bitsandbytes and 16 GB of card; "
                         "pass nf4 to reproduce what the demo box serves.")
    ap.add_argument("--count-answer", choices=["bucket", "integer"], default="integer",
                    help="RSVQA counting rows: 'integer' is the joint_v3 recipe, "
                         "which regresses an exact number and is then graded on the "
                         "bucket it lands in (63.0 binned, the weakest type); "
                         "'bucket' teaches the published five-bucket label under "
                         "the matching instruction, and the eval cell is run with "
                         "--count-prompt bucket to ask the same way. Its own kernel: "
                         "slug and folder take a -cnt suffix.")
    ap.add_argument("--vision-lr-mult", type=float, default=1.0,
                    help="train the vision LoRA at this multiple of the language "
                         "rate (default 1.0, one rate for both). Needs "
                         "--vision-lora. Qwen ships its ViT frozen through its own "
                         "SFT and DPO, so the vision adapters start further from "
                         "where they need to be than the language ones do. Its own "
                         "kernel: the slug takes an -lrN suffix.")
    ap.add_argument("--rsvqa-rows", type=int, default=20000,
                    help="--data rsvqa only: rows in the training subset (the whole train "
                         "split is 57,223; 20,000 is the 4 Sep recipe).")
    ap.add_argument("--ref-rows", type=int, default=CELL_REF_ROWS,
                    help="--data ground-ref only: reference rows to train on. The Lithuania/"
                         "Summer train cell holds 10,986 and that is the ref-vis recipe (it asked "
                         "for 16,000 and got the cell); above 10,986 the cut widens to all four "
                         "Lithuanian seasons. Point rows stay at 4,000.")
    ap.add_argument("--vrsbench-rows", type=int, default=30000,
                    help="--data vrsbench-*: rows in the training subset (train has 85,813 VQA "
                         "and 36,313 referring pairs; 30,000 fits one 10.5 h session at 400 tokens).")
    ap.add_argument("--visual-tokens", type=int, default=None,
                    help="processor max_pixels in 28x28 tokens. Default 256, or 400 for the "
                         "vrsbench data (512 px tiles are 334 tokens; 256 would crop).")
    ap.add_argument("--lora-rank", type=int, default=8,
                    help="LoRA r for every adapter (language and, with --vision-lora, vision). "
                         "alpha stays 32, so this rung changes the rank alone.")
    ap.add_argument("--vision-lora", action="store_true",
                    help="Put LoRA on the vision tower (every block + the patch merger) "
                         "as well as the language projections, with non-reentrant "
                         "checkpointing so the vision adapters receive gradients. Its own "
                         "kernel: slug and folder take a -vis suffix.")
    args = ap.parse_args()

    variant = args.data if args.data != "ben" else ""
    if args.count_answer == "bucket":
        variant = (variant + "-cnt") if variant else "cnt"
    if args.vision_lora:
        variant = (variant + "-vis") if variant else "vis"
    if args.eval_only:
        variant = (variant + "-eval") if variant else "eval"
    if args.vision_lr_mult != 1.0:
        if not args.vision_lora:
            ap.error("--vision-lr-mult needs --vision-lora: without it there are "
                     "no vision LoRA parameters to give a second rate to")
        variant += "-lr%g" % args.vision_lr_mult
    if args.rsvqa_rows != 20000:
        if args.data != "rsvqa":
            sys.exit("--rsvqa-rows only applies to --data rsvqa")
        variant += "-rows%d" % args.rsvqa_rows
    if args.ref_rows != CELL_REF_ROWS:
        if args.data != "ground-ref":
            sys.exit("--ref-rows only applies to --data ground-ref")
        variant += "-ref%d" % args.ref_rows
    if args.lora_rank != 8:
        variant += "-r%d" % args.lora_rank
    out_dir = OUT_DIR if not variant else os.path.join(OUT_DIR, variant)
    slug = "satquery-qlora" if not variant else "satquery-qlora-" + variant
    nb_name = "qlora-kaggle.ipynb" if not variant else "qlora-%s-kaggle.ipynb" % variant
    out = os.path.join(out_dir, nb_name)

    os.makedirs(out_dir, exist_ok=True)
    nb = build(args.data, vision_lora=args.vision_lora,
               count_answer=args.count_answer,
               vision_lr_mult=args.vision_lr_mult,
               rsvqa_rows=args.rsvqa_rows, ref_rows=args.ref_rows,
               lora_rank=args.lora_rank,
               visual_tokens=args.visual_tokens or (400 if args.data.startswith("vrsbench") else 256),
               vrsbench_rows=args.vrsbench_rows)
    if args.eval_only:
        count_eval = (" --count-prompt bucket"
                      if args.count_answer == "bucket" else "")
        nb["cells"] = eval_only_cells(
            nb["cells"], args.data, args.eval_only,
            args.eval_limit, args.eval_dtype, count_eval)
    if args.epochs != 1:
        n_hit = 0
        for cell in nb["cells"]:
            src = "".join(cell["source"])
            if "num_train_epochs=1," in src:
                cell["source"] = [ln.replace("num_train_epochs=1,", "num_train_epochs=%d," % args.epochs)
                                  for ln in cell["source"]]
                n_hit += 1
        assert n_hit == 1, "expected exactly one TrainingArguments cell, found %d" % n_hit
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)

    meta = {
        "id": "%s/%s" % (args.user, slug),
        "title": slug,
        "code_file": nb_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [args.resume_from] if args.resume_from else [],
        # An eval-only kernel mounts the adapter it scores the same way a
        # resume leg mounts its checkpoint: as the producing kernel's output
        # under /kaggle/input. No dataset upload, and the weights are
        # read-only, so nothing here can modify what is being measured.
        "kernel_sources": ([args.eval_only] if args.eval_only
                           else [args.resume_kernel] if args.resume_kernel else []),
        "competition_sources": [],
    }
    meta_path = os.path.join(out_dir, "kernel-metadata.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print("wrote %s (%.0f KB, %d cells)" % (out, os.path.getsize(out) / 1024, len(nb["cells"])))
    print("wrote %s -> %s" % (meta_path, meta["id"]))
    print("accelerator requested:", nb["metadata"]["kaggle"]["accelerator"])
    if args.resume_from:
        print("resuming from dataset:", args.resume_from)
    print("")
    print("push:  kaggle kernels push -p %s" % os.path.relpath(out_dir, HERE))
    print("Check Settings -> GPU hours FIRST. Quota exhaustion is what "
          "substitutes a P100, and 4-bit will not load on it.")


if __name__ == "__main__":
    main()
