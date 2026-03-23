#!/usr/bin/env python3
"""
Claude Code UserPromptSubmit hook.
Runs auto_tip.py when the user's prompt mentions NRL tips,
then injects the output as context for Claude.
"""
import json
import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUTO_TIP   = os.path.join(SCRIPT_DIR, "s6_tips.py")

TIP_PATTERN = re.compile(
    r"\b(tip(s|ping)?|nrl\s+pick|pick(s)?\s+this\s+week|who\s+(do\s+i\s+)?pick)\b",
    re.IGNORECASE,
)

data   = json.load(sys.stdin)
prompt = data.get("prompt", "")

if not TIP_PATTERN.search(prompt):
    sys.exit(0)   # not a tips request — do nothing

result = subprocess.run(
    [sys.executable, AUTO_TIP],
    capture_output=True,
    text=True,
    timeout=90,
)

output = result.stdout.strip()
if result.returncode != 0 and result.stderr:
    output += f"\n\n[auto_tip errors]\n{result.stderr.strip()}"

if not output:
    output = "[auto_tip.py produced no output — check that dependencies are installed]"

response = {
    "hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit",
        "additionalContext": (
            "The NRL tipping model has just run. Here are the raw results — "
            "please present them clearly to the user:\n\n" + output
        ),
    }
}
print(json.dumps(response))
