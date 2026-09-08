"""Compile every code cell of a generated notebook before it costs a GPU hour.

The notebook is built by string concatenation in make_train_notebook.py, so a
missing quote or a bad %-format lands as a SyntaxError forty minutes into a
Kaggle session, after the dataset download has already been paid for.

    python scripts/check_notebook.py train/qlora-kaggle.ipynb

Shell magics are not Python, so they are blanked rather than compiled - but
the two prefixes need different rules, and both false-positived on the first
drafts of this script:

  `%` is a magic only at COLUMN ZERO. An indented `% (a, b)` is the
      continuation of a %-format expression - perfectly good Python.
  `!` is a magic at ANY indentation, because IPython allows it inside a loop
      body and no Python statement begins with `!`.

A magic line becomes `pass` at its own indentation, not a blank line: blanking
an indented `!curl` empties its `for` body and raises IndentationError in code
that is fine. Continuation lines (trailing backslash) do go blank, so the line
numbers in any error still match the notebook.
"""
import json
import sys

BACKSLASH = chr(92)


def blank_magics(source):
    out, in_magic = [], False
    for line in source:
        body = line.rstrip("\n")
        magic = body[:1] == "%" or body.lstrip()[:1] == "!"
        if magic and not in_magic:
            indent = body[:len(body) - len(body.lstrip())]
            out.append(indent + "pass\n")
            in_magic = body.rstrip().endswith(BACKSLASH)
        elif in_magic:
            in_magic = body.rstrip().endswith(BACKSLASH)
            out.append("\n")
        else:
            out.append(line if line.endswith("\n") else line + "\n")
    return "".join(out)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "train/qlora-kaggle.ipynb"
    nb = json.load(open(path, encoding="utf-8"))

    bad = n_code = 0
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        n_code += 1
        try:
            compile(blank_magics(cell["source"]), "<cell %d>" % i, "exec")
        except SyntaxError as exc:
            bad += 1
            print("CELL %d: %s at line %s" % (i, exc.msg, exc.lineno))
            print("   %s" % (exc.text or "").rstrip())
    print("%d code cells, %d markdown, %d broken"
          % (n_code, len(nb["cells"]) - n_code, bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
