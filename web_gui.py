#!/usr/bin/env python3
"""NRL Tips — Flask web interface. Run with: python3 web_gui.py"""

import json
import os
import queue
import subprocess
import sys
import threading

from flask import Flask, Response, render_template, request

SCRIPT_DIR         = os.path.dirname(os.path.abspath(__file__))
AUTO_TIP           = os.path.join(SCRIPT_DIR, "s6_tips.py")
TRAIN_CMD          = os.path.join(SCRIPT_DIR, "m5_nrl.py")
COLLECT_CMD        = os.path.join(SCRIPT_DIR, "s0_collect.py")
PERF_CMD           = os.path.join(SCRIPT_DIR, "s9_performance.py")
MODEL_INFO_ODDS    = os.path.join(SCRIPT_DIR, "nrl_model_info.json")
MODEL_INFO_NO_ODDS = os.path.join(SCRIPT_DIR, "nrl_model_no_odds_info.json")

app = Flask(__name__)

_process: subprocess.Popen | None = None
_lock = threading.Lock()


def _build_cmd(action: str, round_val: str) -> tuple[list, str]:
    rnd_args = (
        ["--round", round_val]
        if round_val and round_val.lower() != "auto" and round_val.isdigit()
        else []
    )
    match action:
        case "tips":
            return [sys.executable, AUTO_TIP] + rnd_args, "Fetching tips (with odds)..."
        case "tips_no_odds":
            return [sys.executable, AUTO_TIP, "--no-odds"] + rnd_args, "Fetching tips (no odds)..."
        case "compare":
            return [sys.executable, PERF_CMD, "--compare"] + rnd_args, "Comparing models..."
        case "collect_new":
            return [sys.executable, COLLECT_CMD, "--new-only"], "Collecting new data..."
        case "collect_all":
            return [sys.executable, COLLECT_CMD], "Collecting all data..."
        case "retrain":
            return [sys.executable, TRAIN_CMD, "--train"], "Training model (with odds)..."
        case "retrain_no_odds":
            return [sys.executable, TRAIN_CMD, "--train", "--no-odds"], "Training model (no odds)..."
        case _:
            return [], "Unknown action"


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/model_info")
def model_info():
    def load(path, label):
        if not os.path.exists(path):
            return [f"[{label}] No model info found — run Retrain first."]
        with open(path) as f:
            d = json.load(f)
        lines = [
            f"=== {label} ===",
            f"  Trained:           {d.get('trained_at', 'unknown')}",
            f"  CV accuracy:       {d['cv_accuracy']:.3f} ± {d['cv_std']:.3f}",
            f"  Training accuracy: {d['train_accuracy']:.3f}",
            f"  Brier score:       {d['brier_score']:.4f}  (lower = better; random ≈ 0.25)",
            "",
            "  Top features:",
        ]
        for feat in d.get("features", []):
            bar = "█" * int(feat["importance"] * 200)
            lines.append(f"    {feat['name']:<35} {bar} ({feat['importance']:.4f})")
        return lines

    odds_lines    = load(MODEL_INFO_ODDS,    "With Odds")
    no_odds_lines = load(MODEL_INFO_NO_ODDS, "No Odds")

    col_width = max((len(l) for l in odds_lines), default=40)
    col_width = max(col_width, 40)
    n   = max(len(odds_lines), len(no_odds_lines))
    sep = "─" * col_width + "  │  " + "─" * 40

    lines = [f"{'── With Odds ──':<{col_width}}  │  ── No Odds ──", sep]
    for i in range(n):
        left  = odds_lines[i]    if i < len(odds_lines)    else ""
        right = no_odds_lines[i] if i < len(no_odds_lines) else ""
        lines.append(f"{left:<{col_width}}  │  {right}")

    return {"lines": lines}


@app.post("/cancel")
def cancel():
    global _process
    with _lock:
        if _process and _process.poll() is None:
            _process.kill()
    return "", 204


@app.get("/stream")
def stream():
    action    = request.args.get("action", "")
    round_val = request.args.get("round", "auto").strip()

    cmd, label = _build_cmd(action, round_val)

    def generate():
        global _process

        if not cmd:
            yield _evt({"type": "error", "text": "Unknown action"})
            yield _evt({"type": "done", "exit": 1, "status": "Error"})
            return

        with _lock:
            if _process and _process.poll() is None:
                yield _evt({"type": "error", "text": "[busy — already running, please wait]"})
                yield _evt({"type": "done", "exit": 1, "status": "Busy"})
                return

            env = os.environ.copy()
            _process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=SCRIPT_DIR,
                env=env,
                text=True,
                bufsize=1,
            )

        yield _evt({"type": "status", "text": label})

        out_q: queue.Queue = queue.Queue()

        def reader(pipe, stream_type):
            for line in pipe:
                out_q.put((stream_type, line.rstrip()))
            out_q.put((stream_type, None))

        threading.Thread(target=reader, args=(_process.stdout, "stdout"), daemon=True).start()
        threading.Thread(target=reader, args=(_process.stderr, "stderr"), daemon=True).start()

        done_count = 0
        while done_count < 2:
            try:
                stream_type, line = out_q.get(timeout=60)
            except queue.Empty:
                break
            if line is None:
                done_count += 1
                continue
            yield _evt({"type": stream_type, "text": line})

        _process.wait()
        code = _process.returncode
        if code in (-2, -9, 9, 15):
            status = "Cancelled"
        elif code == 0:
            status = "Done"
        else:
            status = f"Finished with errors (exit {code})"

        yield _evt({"type": "done", "exit": code, "status": status})

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _evt(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


if __name__ == "__main__":
    print("Starting NRL Tips web server...")
    print("Open http://localhost:5000 in your browser")
    print("On your phone (same Wi-Fi): http://<your-linux-ip>:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
