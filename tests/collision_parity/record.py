"""Record the golden collision traces, or check the current code against them.

Usage::

    python -m tests.collision_parity.record            # record every scene
    python -m tests.collision_parity.record crossing_lanes attach_detach
    python -m tests.collision_parity.record --check    # compare, exit 1 on a difference

Record on the revision the parity test is meant to compare against, and say
so in the commit: the trace's ``reference`` field holds the revision.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List, Optional

from .scenes import SCENES
from .trace import CollisionTrace, compare_traces, golden_path, record_scene


def current_revision() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenes", nargs="*", help="scene names (default: every scene)")
    parser.add_argument("--check", action="store_true", help="compare against the goldens instead of writing them")
    args = parser.parse_args(argv)

    names = args.scenes or list(SCENES)
    unknown = [n for n in names if n not in SCENES]
    if unknown:
        parser.error(f"unknown scene(s): {', '.join(unknown)}; known: {', '.join(SCENES)}")

    reference = current_revision()
    failed = False
    for name in names:
        trace = record_scene(SCENES[name], reference)
        path = golden_path(name)
        if args.check:
            if not path.exists():
                print(f"{name}: no golden trace at {path}")
                failed = True
                continue
            golden = CollisionTrace.load(path)
            differences = compare_traces(golden, trace)
            status = "same" if not differences else f"{len(differences)} difference(s)"
            print(f"{name}: {status} (golden recorded on {golden.reference})")
            for difference in differences:
                print(f"  {difference}")
            failed = failed or bool(differences)
        else:
            trace.save(path)
            active = sum(len(s.pairs) for s in trace.steps)
            events = sum(len(s.events) for s in trace.steps)
            print(f"{name}: {len(trace.steps)} steps, {events} events, {active} pair samples -> {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
