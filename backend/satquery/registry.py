"""Load, validate and lock the tool registry.

The registry is the interface between the two tracks. The ML pair produces
things that satisfy a row; everyone else consumes traces. Nothing in the GUI,
the report generator or the evaluation harness may reach past this into a model.

`registry.lock.json` pins the version of every row at the moment a run happens,
and its hash goes into the trace. That is what makes `--replay` an audit rather
than a re-run: if a row changed between the original run and the replay, the
lock no longer matches and the replay says so instead of quietly producing a
different answer.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY_PATH = os.path.join(HERE, "registry.json")
SCHEMA_PATH = os.path.join(HERE, "schemas", "registry.schema.json")
LOCK_PATH = os.path.join(HERE, "registry.lock.json")


class Row(dict):
    """One registry row. A dict so it serialises straight into the trace."""

    @property
    def id(self) -> str:
        return self["id"]

    @property
    def task(self) -> str:
        return self["task"]

    @property
    def served(self) -> bool:
        # Absent means served. baseline.geochat sets it false: it is a
        # development-only comparison and must never reach the answer path.
        return self.get("served", True)

    @property
    def weights_key(self) -> str:
        """Residency is keyed on WEIGHTS, not on row id.

        vqa.rs, ground.rs and caption.rs are the same Qwen weights with
        different adapters and prompts. A loader that keys on row id will
        triple-count 2.5 GB and evict things it did not need to.
        """
        return self.get("shares_weights_with", self["id"])


def load(path: str = REGISTRY_PATH, validate: bool = True) -> list:
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    if validate:
        _validate(doc)
    return [Row(r) for r in doc["rows"]]


def _validate(doc: dict) -> None:
    try:
        import jsonschema
    except ImportError:
        raise RuntimeError(
            "jsonschema is not installed, so the registry cannot be validated. "
            "Install it rather than skipping the check - an unvalidated registry "
            "is how a typo in a row id becomes a silent routing failure."
        )
    with open(SCHEMA_PATH, "r", encoding="utf-8") as fh:
        schema = json.load(fh)
    jsonschema.validate(doc, schema)

    ids = [r["id"] for r in doc["rows"]]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError("duplicate registry row ids: %s" % ", ".join(dupes))

    # A shares_weights_with target must exist, or the loader will look for a
    # model that was never declared.
    for r in doc["rows"]:
        tgt = r.get("shares_weights_with")
        if tgt and tgt not in ids:
            raise ValueError("%s shares weights with unknown row %r" % (r["id"], tgt))

    for r in doc["rows"]:
        lo, hi = r["accepts"]["gsd_range_m"]
        if lo >= hi:
            raise ValueError("%s has an empty gsd_range_m: %r" % (r["id"], [lo, hi]))
        b = r["accepts"]["bands"]
        if b["min"] > b["max"]:
            raise ValueError("%s has an empty bands range: %r" % (r["id"], b))


def served(rows: list) -> list:
    return [r for r in rows if r.served]


def by_id(rows: list, row_id: str) -> Row:
    for r in rows:
        if r.id == row_id:
            return r
    raise KeyError("no registry row %r" % row_id)


def candidates(rows: list, task: str, images: list) -> list:
    """Rows whose declared preconditions the validated input satisfies.

    This is the only place a tool is chosen by its declared capability rather
    than by name, which is what lets a new capability be added as a row plus a
    module with no dispatcher change.
    """
    out = []
    for r in served(rows):
        if r.task != task:
            continue
        ok, _ = check_accepts(r, images)
        if ok:
            out.append(r)
    return out


def check_accepts(row: Row, images: list):
    """Does this row's DECLARED contract cover these images? (ok, reason).

    The dispatcher picks a row by rule, but a rule knows about image count,
    modality and dates - not about band counts, formats or the resolution range
    a component was actually built for. Without this check the registry
    documents preconditions that nothing enforces, and a tool gets handed input
    it never claimed to handle. The contract is explicit that the dispatcher
    executes rows "whose declared preconditions are satisfied by the validated
    input"; this is that sentence, in code.
    """
    a = row["accepts"]
    if len(images) > a["n_images"]:
        return False, "%s accepts at most %d image(s), got %d" % (
            row.id, a["n_images"], len(images))

    mods = {im.modality for im in images}
    declared = set(a["modality"])
    if len(mods) > 1:
        if "mixed" not in declared:
            return False, "%s does not accept a mixed-modality pair (%s)" % (
                row.id, ", ".join(sorted(mods)))
    else:
        only = next(iter(mods))
        if only not in declared and "mixed" not in declared:
            return False, "%s accepts %s, not %s" % (
                row.id, "/".join(sorted(declared)), only)

    for im in images:
        nm = os.path.basename(im.path)
        if im.fmt not in a["formats"]:
            return False, "%s accepts %s, and %s is %s" % (
                row.id, "/".join(a["formats"]), nm, im.fmt)
        if not a["bands"]["min"] <= im.n_bands <= a["bands"]["max"]:
            return False, "%s accepts %d-%d bands, and %s has %d" % (
                row.id, a["bands"]["min"], a["bands"]["max"], nm, im.n_bands)
        if im.gsd_m is not None:
            lo, hi = a["gsd_range_m"]
            if not lo <= im.gsd_m <= hi:
                return False, (
                    "%s is declared for %g-%g m ground sample distance, and %s "
                    "is %.4g m" % (row.id, lo, hi, nm, im.gsd_m))
    return True, None


def write_lock(rows: list, path: str = LOCK_PATH) -> str:
    """Pin every row's version and return the lock's own hash."""
    doc = {
        "rows": {r.id: {"version": r["version"], "weights_sha256": r["weights_sha256"]} for r in rows},
    }
    blob = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(blob)
    return lock_hash(path)


def lock_hash(path: str = LOCK_PATH) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def ensure_lock(rows: list, path: str = LOCK_PATH) -> str:
    if not os.path.exists(path):
        return write_lock(rows, path)
    return lock_hash(path)
