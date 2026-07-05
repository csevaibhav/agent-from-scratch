"""
Companion to step7_observability.py -- reads agent_logs.jsonl and answers
real questions about agent behavior across one or many runs.

Run this AFTER running step7_observability.py at least once (ideally a
few times, so there's more than one session to compare):

    python analyze_logs.py
"""

import json
from collections import defaultdict
from pathlib import Path

LOG_FILE = Path("./agent_logs.jsonl")


def load_events():
    if not LOG_FILE.exists():
        print(f"No log file found at {LOG_FILE}. Run step7_observability.py first.")
        return []
    events = []
    with open(LOG_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def main():
    events = load_events()
    if not events:
        return

    sessions = defaultdict(list)
    for e in events:
        sessions[e["session_id"]].append(e)

    print(f"Found {len(events)} log entries across {len(sessions)} session(s).\n")

    # --- Question 1: how long did each session take, and how successful was it? ---
    print("=== Session overview ===")
    for sid, evs in sessions.items():
        finished = next((e for e in evs if e["event"] == "session_finished"), None)
        started = next((e for e in evs if e["event"] == "session_started"), None)
        if finished and started:
            print(
                f"[{sid}] \"{started['request'][:60]}...\" "
                f"-> {finished['succeeded']}/{finished['total_subtasks']} succeeded "
                f"in {finished['duration_seconds']}s"
            )

    # --- Question 2: which tools were called, how often, and how fast? ---
    print("\n=== Tool call stats (across all sessions) ===")
    tool_calls = [e for e in events if e["event"] == "tool_call"]
    by_tool = defaultdict(list)
    for e in tool_calls:
        by_tool[e["tool"]].append(e)

    for tool, calls in by_tool.items():
        successes = [c for c in calls if c["success"]]
        failures = [c for c in calls if not c["success"]]
        durations = [c["duration_seconds"] for c in calls if "duration_seconds" in c]
        avg_duration = round(sum(durations) / len(durations), 4) if durations else 0
        print(f"{tool}: {len(calls)} calls, {len(successes)} succeeded, "
              f"{len(failures)} failed, avg {avg_duration}s")

    # --- Question 3: what actually failed, and why? ---
    print("\n=== All failures (across all sessions) ===")
    failures = [e for e in events if e["event"] == "tool_call" and not e.get("success", True)]
    if not failures:
        print("No tool failures logged.")
    for f in failures:
        print(f"  [{f['session_id']}] {f['tool']}({f['args']}) "
              f"-> {f.get('failure_type', '?')}: {f.get('error', '?')}")

    # --- Question 4: which subtasks needed retries? ---
    print("\n=== Subtasks that needed retries ===")
    retries = [e for e in events if e["event"] == "subtask_retry"]
    if not retries:
        print("No retries logged.")
    for r in retries:
        print(f"  [{r['session_id']}] subtask {r['index']} ({r['task'][:50]}...) attempt {r['attempt']}")


if __name__ == "__main__":
    main()