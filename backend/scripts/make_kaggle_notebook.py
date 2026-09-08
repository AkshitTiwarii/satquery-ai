"""Build a self-contained Kaggle notebook that runs the real pipeline on a T4.

    python scripts/make_kaggle_notebook.py

Writes `akshit/satquery-live.ipynb`. Upload that ONE file to Kaggle, set the
accelerator, run. No dataset to upload, no git credentials, no /kaggle/input
paths to get wrong - the whole `satquery` package is embedded in the notebook
as base64 and written out by the second cell.

WHY THE NOTEBOOK IS BUILT RATHER THAN HAND-WRITTEN
--------------------------------------------------
Everything an earlier QLoRA project of ours lost time to on Kaggle was a
mismatch between what was on the laptop and what the kernel actually ran:
stale dataset attachments, a notebook whose code was one version behind, an
accelerator that was set in the wrong file. Generating the notebook from the
working tree removes all three - the code in the notebook IS the code in the
repo, by construction.

THE ACCELERATOR, AND WHAT ACTUALLY DECIDES IT
---------------------------------------------
GPU type is NOT in `kernel-metadata.json` (that file carries only
`enable_gpu`, a boolean, and the Kaggle CLI has no field for the type at all).
It lives in the notebook's own JSON, at `metadata.kaggle.accelerator`, and this
generator sets it to `nvidiaTeslaT4`.

That is a REQUEST, NOT A GUARANTEE. When your weekly GPU quota will not cover a
T4, Kaggle overrides it and gives you a Tesla P100 - and a P100 is sm_60, on
which every 4-bit bitsandbytes kernel dies at model load with "no kernel image
available for execution", which reads exactly like a broken CUDA install and
sends you debugging the wrong thing.

Four theories that are WRONG, each disproved by measurement rather than
argument, and each one a day lost:

  1. "The CLI push resets it."      No - CLI pushes ran fine for three
                                    consecutive versions while quota lasted.
  2. "P100 vs T4 is luck."          No - it tracks quota.
  3. "Setting the metadata fixes it." No - it was already set to nvidiaTeslaT4
                                    and Kaggle ignored it.
  4. "It is main-account-only."     No - a second account with spent quota drew
                                    a P100 too.

So: CHECK SETTINGS -> GPU HOURS FIRST. 30 h/week. If it is spent, nothing you
do in any file will get you a T4, and the only fix is to wait for the reset.
Cell 1 of the generated notebook checks the card you actually got and stops
immediately with that explanation, so a bad allocation costs you ten seconds
instead of an afternoon.
"""

import base64
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "akshit")
OUT = os.path.join(OUT_DIR, "satquery-live.ipynb")

# Everything the package needs to run. Tests are deliberately not shipped: the
# notebook's job is to exercise the REAL backend, which the suite cannot.
def _package_files():
    """Every file the package is made of, read from the tree, not from a list.

    The first build of this notebook carried a hand-typed list. Two days later
    `satquery/export.py` existed, `tools.py` imported it, and the list did not
    know - so the notebook shipped a package that could not import itself, and
    failed in the self-test cell on someone else's screen. A list that has to be
    kept in step with the tree is a list that will not be.
    """
    out = []
    for root, _, files in os.walk(os.path.join(HERE, "satquery")):
        for f in sorted(files):
            if f.endswith((".py", ".json")) and "__pycache__" not in root:
                out.append(os.path.relpath(os.path.join(root, f), HERE).replace(os.sep, "/"))
    # Akshit's visual-token study runs the SAME harness that produced the
    # 60.8% baseline. A different script would produce a number that cannot be
    # compared with it, which is the whole point of having one harness.
    out.append("scripts/eval_rsvqa_lr.py")
    return out


PACKAGE_FILES = _package_files()


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": list(lines)}


def _payload():
    blob = {}
    for rel in PACKAGE_FILES:
        with open(os.path.join(HERE, rel), "rb") as fh:
            blob[rel] = base64.b64encode(fh.read()).decode("ascii")
    return base64.b64encode(json.dumps(blob).encode("utf-8")).decode("ascii")


def build():
    cells = []

    cells.append(md(
        "# SatQuery - the real pipeline on a T4\n",
        "\n",
        "Runs the registry, gate, dispatcher and trace end to end with the **real**\n",
        "Qwen2.5-VL-3B behind `vqa.rs`, `ground.rs` and `caption.rs`.\n",
        "\n",
        "## Before you press anything\n",
        "\n",
        "1. **Settings -> GPU hours.** You get 30 a week. **If that is spent you will be\n",
        "   given a Tesla P100 no matter what this notebook asks for**, and the 4-bit\n",
        "   model will not load on it. Check this first; it is one page load.\n",
        "2. Right panel -> **Accelerator = GPU T4 x2**.\n",
        "3. Right panel -> **Internet = On**.\n",
        "4. **Save Version -> Save & Run All** and close the tab, or just run the cells.\n",
        "\n",
        "Cell 1 checks which card you actually got and stops immediately if it is the\n",
        "wrong one, so a bad allocation costs ten seconds rather than an afternoon.",
    ))

    # --- 1. PREFLIGHT. First, and it must be able to stop the whole run. ----
    cells.append(code(
        "# ---- 1. preflight: which card did we actually get? ----\n",
        "# bitsandbytes 4-bit needs sm_75 (Turing) or better. T4 is sm_75.\n",
        "# P100 is sm_60 and every 4-bit kernel dies at model load with\n",
        "# 'no kernel image available for execution', which looks like a broken\n",
        "# CUDA install and is not one.\n",
        "import torch\n",
        "\n",
        "if not torch.cuda.is_available():\n",
        "    raise RuntimeError(\n",
        "        'No GPU attached. Right panel -> Accelerator -> GPU T4 x2.')\n",
        "\n",
        "name = torch.cuda.get_device_name(0)\n",
        "major, minor = torch.cuda.get_device_capability(0)\n",
        "sm = major * 10 + minor\n",
        "total = torch.cuda.get_device_properties(0).total_memory / 1e9\n",
        "print('device      :', name)\n",
        "print('capability  : sm_%d' % sm)\n",
        "print('memory      : %.1f GB' % total)\n",
        "print('torch       :', torch.__version__)\n",
        "\n",
        "if sm < 75:\n",
        "    raise RuntimeError(\n",
        "        '\\n'\n",
        "        '=================================================================\\n'\n",
        "        'STOP. You are on a %s (sm_%d). 4-bit will not load on it.\\n'\n",
        "        '\\n'\n",
        "        'This is NOT a broken install and NOT this notebook. Kaggle gives\\n'\n",
        "        'you this card when your weekly GPU quota will not cover a T4.\\n'\n",
        "        '\\n'\n",
        "        'Check Settings -> GPU hours. If it reads 30/30, the only fix is to\\n'\n",
        "        'wait for the weekly reset. Nothing in any config file overrides it:\\n'\n",
        "        'this notebook already requests nvidiaTeslaT4 and was ignored.\\n'\n",
        "        '=================================================================' % (name, sm))\n",
        "\n",
        "print('\\nOK - this card can run 4-bit.')",
    ))

    # --- 2. unpack the package -----------------------------------------------
    cells.append(code(
        "# ---- 2. write out the satquery package (embedded, no upload needed) ----\n",
        "import base64, json, os, sys\n",
        "\n",
        "PAYLOAD = '''%s'''\n" % _payload(),
        "\n",
        "root = '/kaggle/working/satquery-src'\n",
        "for rel, b64 in json.loads(base64.b64decode(PAYLOAD)).items():\n",
        "    dst = os.path.join(root, rel)\n",
        "    os.makedirs(os.path.dirname(dst), exist_ok=True)\n",
        "    with open(dst, 'wb') as fh:\n",
        "        fh.write(base64.b64decode(b64))\n",
        "sys.path.insert(0, root)\n",
        "print('wrote', sum(len(f) for _, _, f in os.walk(root)), 'files to', root)",
    ))

    # --- 3. dependencies ------------------------------------------------------
    cells.append(code(
        "# ---- 3. dependencies ----\n",
        "!pip install -q -U transformers accelerate bitsandbytes peft jsonschema qwen-vl-utils\n",
        "import transformers; print('transformers', transformers.__version__)",
    ))

    # --- 4. data --------------------------------------------------------------
    cells.append(code(
        "# ---- 4. RSVQA-LR: the one benchmark at our own training scale ----\n",
        "# 150 MB. Zenodo 504s under load and returns a short HTML error page that\n",
        "# is not a zip, so check the SIZE before trusting the download.\n",
        "import os, zipfile\n",
        "os.makedirs('/kaggle/working/data', exist_ok=True)\n",
        "base = 'https://zenodo.org/records/6344334/files'\n",
        "for f in ['Images_LR.zip', 'LR_split_test_questions.json', 'LR_split_test_answers.json']:\n",
        "    p = '/kaggle/working/data/' + f\n",
        "    if not os.path.exists(p) or os.path.getsize(p) < 10000:\n",
        "        !curl -sL -C - -o {p} \"{base}/{f}?download=1\"\n",
        "    print('%-34s %10.1f KB' % (f, os.path.getsize(p) / 1024))\n",
        "\n",
        "with zipfile.ZipFile('/kaggle/working/data/Images_LR.zip') as z:\n",
        "    names = [n for n in z.namelist() if n.lower().endswith(('.tif', '.png'))][:8]\n",
        "    z.extractall('/kaggle/working/data/img', members=names)\n",
        "print('extracted', len(names), 'sample images')",
    ))

    # --- 5. self-test on stubs first -----------------------------------------
    cells.append(code(
        "# ---- 5. self-test: the pipeline on STUBS, before any model is loaded ----\n",
        "# If this fails, the problem is the package, not the GPU or the model.\n",
        "# Isolating that here saves debugging a model load that was never the cause.\n",
        "import glob, os\n",
        "os.environ['SATQUERY_BACKEND'] = 'stub'\n",
        "from satquery import run as sq_run, trace as sq_trace\n",
        "\n",
        "img = sorted(glob.glob('/kaggle/working/data/img/**/*.*', recursive=True))[0]\n",
        "print('image:', img)\n",
        "t = sq_run.answer('is there a water body?', [img])\n",
        "sq_trace.validate(t)\n",
        "print('task    :', t['classified_task'], '| rule', t['routing']['rule_id'])\n",
        "print('steps   :', [s['tool'] for s in t['steps']])\n",
        "print('stub    :', all(s['stub'] for s in t['steps']))\n",
        "print('answer  :', t['output']['text'])\n",
        "print('\\nSTUB PATH OK')",
    ))

    # --- 6. the real backend --------------------------------------------------
    cells.append(code(
        "# ---- 6. the REAL backend. First load is ~6 s plus the 7.5 GB download ----\n",
        "import importlib, os\n",
        "os.environ['SATQUERY_BACKEND'] = 'real'   # fail loudly rather than silently stubbing\n",
        "os.environ['SATQUERY_VISUAL_TOKENS'] = '256'\n",
        "\n",
        "from satquery import models\n",
        "importlib.reload(models)\n",
        "print('real backend available:', models.available())\n",
        "if not models.available():\n",
        "    raise RuntimeError('could not load the model: ' + models.why_not())",
    ))

    # --- 7. Gate I evidence ---------------------------------------------------
    cells.append(code(
        "# ---- 7. GATE I: VQA and grounding genuinely live, end to end ----\n",
        "import glob, importlib, json, time\n",
        "from satquery import tools, run as sq_run, trace as sq_trace\n",
        "importlib.reload(tools); importlib.reload(sq_run)\n",
        "\n",
        "imgs = sorted(glob.glob('/kaggle/working/data/img/**/*.*', recursive=True))[:4]\n",
        "for q in ['is there a water body?',\n",
        "          'how many buildings are there?',\n",
        "          'describe this image',\n",
        "          'highlight the largest road']:\n",
        "    t0 = time.time()\n",
        "    t = sq_run.answer(q, [imgs[0]])\n",
        "    sq_trace.validate(t)\n",
        "    live = [s['tool'] for s in t['steps'] if not s['stub']]\n",
        "    print('\\n%-34s %s  %5.2fs' % (q, t['routing']['rule_id'], time.time() - t0))\n",
        "    print('   live tools :', live or '(none - still stubbed)')\n",
        "    print('   answer     :', t['output']['text'][:200])",
    ))

    # --- 8. the coordinate trap ----------------------------------------------
    cells.append(code(
        "# ---- 8. THE COORDINATE TRAP - eyeball this one, do not skim it ----\n",
        "# Qwen2.5-VL returns boxes in ABSOLUTE PIXELS of the image AFTER the\n",
        "# processor's own resize. BigEarthNet.txt normalises 0-1, VRSBench 0-100.\n",
        "# Three conventions, no error if you mix them, and a box that is wrong by\n",
        "# 100x still plots - just in the wrong place.\n",
        "#\n",
        "# `box_frame` says which denominator was used. It should read 'resized'.\n",
        "# If it reads 'original', the resize is not being accounted for and every\n",
        "# box is off by the resize factor.\n",
        "import glob, json\n",
        "from satquery import run as sq_run\n",
        "from satquery.types import load_image_meta\n",
        "\n",
        "img = sorted(glob.glob('/kaggle/working/data/img/**/*.*', recursive=True))[0]\n",
        "t = sq_run.answer('highlight the largest building', [img])\n",
        "step = [s for s in t['steps'] if s['tool'] == 'ground.rs'][0]\n",
        "print('raw model answer :', t['output']['text'][:300])\n",
        "print('quantities       :', json.dumps(t['output']['quantities'], indent=2))\n",
        "print('stub             :', step['stub'])\n",
        "if t['output']['geojson']:\n",
        "    gj = json.loads(t['output']['geojson'])\n",
        "    print('geojson crs      :', gj.get('crs'))\n",
        "    print('polygon          :', gj['features'][0]['geometry']['coordinates'][0][:2])",
    ))

    # --- 9. visual-token study: the measured table ---------------------------
    cells.append(code(
        "# ---- 9. is 256 visual tokens the right number? ----\n",
        "# Nobody chose 256 carefully; it was a guess. This runs the SAME harness\n",
        "# that produced our 60.8% baseline, at four caps, so every number here is\n",
        "# comparable with that one and with the others.\n",
        "#\n",
        "# ~20 min per cap on 400 questions - about 90 min total. Start it and go\n",
        "# and do something else.\n",
        "import json, subprocess, sys\n",
        "\n",
        "HARNESS = '/kaggle/working/satquery-src/scripts/eval_rsvqa_lr.py'\n",
        "rows = []\n",
        "for n in [128, 256, 512, 1024]:\n",
        "    print('=' * 60)\n",
        "    print('RUNNING WITH', n, 'TOKENS PER IMAGE')\n",
        "    print('=' * 60)\n",
        "    subprocess.run([sys.executable, HARNESS,\n",
        "                    '--data-dir', '/kaggle/working/data',\n",
        "                    '--model', 'Qwen/Qwen2.5-VL-3B-Instruct',\n",
        "                    '--max-pixels', str(n),\n",
        "                    '--limit', '400',\n",
        "                    '--out', '/kaggle/working/tokens_%d' % n], check=True)\n",
        "    rows.append(json.load(open('/kaggle/working/tokens_%d.json' % n)))\n",
        "\n",
        "hdr = ('tokens', 'accuracy', 'majority', 'delta', 'minutes')\n",
        "print()\n",
        "print('%8s %10s %10s %8s %9s' % hdr)\n",
        "for r in rows:\n",
        "    mark = '   <- what we use now' if r['max_pixels_tokens'] == 256 else ''\n",
        "    print('%8d %9.1f%% %9.1f%% %+8.1f %9.1f%s' % (\n",
        "        r['max_pixels_tokens'],\n",
        "        r['overall_accuracy_binned'],\n",
        "        r['majority_baseline_binned'],\n",
        "        r['beats_majority_by_binned'],\n",
        "        r['elapsed_min'], mark))\n",
        "\n",
        "# ALWAYS read accuracy against the majority column. 58% on presence sounds\n",
        "# fine until you notice that answering 'yes' every time scores 84%.\n",
        "print()\n",
        "print('per question type, at 256 tokens:')\n",
        "for t, d in sorted(rows[1]['per_type'].items()):\n",
        "    print('  %-12s %6.1f%%   majority %5.1f%%   delta %+6.1f   unparsed %d' % (\n",
        "        t, d['accuracy_binned'], d['majority_binned'],\n",
        "        d['accuracy_binned'] - d['majority_binned'], d['unparsed']))",
    ))


    cells.append(md(
        "## What to send back\n",
        "\n",
        "* Cell 1's device line - proof this ran on a T4 and not a P100.\n",
        "* Cell 7's four answers, and which tools reported `live`.\n",
        "* **Cell 8's `box_frame` value.** This is the one thing that cannot be\n",
        "  checked without a GPU, so it is the one thing worth reading carefully.\n",
        "* Cell 9's table.",
    ))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            # THE accelerator field. Not kernel-metadata.json - that has only
            # enable_gpu (bool) and the CLI exposes no type at all. Even here it
            # is a request Kaggle overrides when quota is short; see the module
            # docstring and cell 1.
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "dataSources": [],
                "isInternetEnabled": True,
                "isGpuEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    nb = build()
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)

    meta = {
        "id": "vinayakameta/satquery-live",
        "title": "satquery-live",
        "code_file": "satquery-live.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    size = os.path.getsize(OUT)
    print("wrote %s (%.0f KB, %d cells)" % (OUT, size / 1024, len(nb["cells"])))
    print("accelerator requested:", nb["metadata"]["kaggle"]["accelerator"])
    print("\nUpload that one file to Kaggle - it carries the whole package.")
    print("Check Settings -> GPU hours FIRST, then set Accelerator = GPU T4 x2.")


if __name__ == "__main__":
    main()
