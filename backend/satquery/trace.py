"""Build, validate and write the graded artefact.

Four rules, straight from the contract:

  1. The GUI consumes traces and nothing else. Anything the front end can
     display is by construction something the trace recorded, so the demo and
     the audit log can never disagree.
  2. `input_check` is a peer of `steps`, not one of them - the gate is a
     standing duty of the controller, not a tool it chose to call.
  3. Every emitted trace validates against the schema, INCLUDING refusals. A
     rejected query still produces a complete trace with `abstained: true`.
  4. `--replay trace.json` reproduces byte-identical output.

On (4): the replay compares the `output` block, not the whole trace.
`duration_ms` is wall-clock and will never repeat exactly; asserting on it
would make the audit fail for a reason that has nothing to do with the answer.
Everything that determines the answer - seed, input hashes, the registry lock -
is compared, and those are what "auditable" has to mean.
"""

from __future__ import annotations

import json
import os
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(HERE, "schemas", "trace.schema.json")

_SCHEMA = None


def _schema():
    global _SCHEMA
    if _SCHEMA is None:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as fh:
            _SCHEMA = json.load(fh)
    return _SCHEMA


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def build(query, classified_task, routing, input_check, steps, output, abstained, replay, trace_id=None) -> dict:
    # input_check carries scratch keys the orchestrator uses between stages
    # (plan_mode, needs_reproject). They are working state, not part of the
    # graded record, so they are dropped here rather than widening the schema.
    ic = {k: v for k, v in input_check.items() if k not in ("plan_mode", "needs_reproject", "pair_declaration")}
    return {
        "trace_id": trace_id or new_id(),
        "query": query,
        "classified_task": classified_task,
        "routing": routing,
        "input_check": ic,
        "steps": steps,
        "output": output,
        "abstained": bool(abstained),
        "replay": replay,
    }


def validate(trace: dict) -> None:
    """Raise if a trace does not validate. Called on every emission.

    A trace that does not validate is worse than no trace: the GUI reads this
    and only this, so a malformed one is a demo that shows nothing with no
    error to explain why.
    """
    try:
        import jsonschema
    except ImportError:
        raise RuntimeError(
            "jsonschema is not installed, so traces cannot be validated. The "
            "trace is the graded artefact - do not run without the check."
        )
    jsonschema.validate(trace, _schema())


def dumps(trace: dict) -> str:
    return json.dumps(trace, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write(trace: dict, path: str) -> str:
    validate(trace)
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(dumps(trace))
    return path


def read(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
