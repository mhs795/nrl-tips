#!/usr/bin/env python3

"""
NRL Tips — standalone desktop GUI (PyQt6)
Run with:  python gui.py
"""

import os
import re
import sys

from PyQt6.QtCore import QProcess, Qt
from PyQt6.QtGui import QColor, QFont, QIcon, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QProgressBar, QPushButton, QStatusBar, QTextEdit, QVBoxLayout, QWidget,
)

import subprocess as _subprocess

SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
AUTO_TIP          = os.path.join(SCRIPT_DIR, "s6_tips.py")
TRAIN_CMD         = os.path.join(SCRIPT_DIR, "m5_nrl.py")
COLLECT_CMD       = os.path.join(SCRIPT_DIR, "s0_collect.py")
PERF_CMD          = os.path.join(SCRIPT_DIR, "s9_performance.py")
EXPORT_CMD        = os.path.join(SCRIPT_DIR, "export_android.py")
MODEL_INFO_ODDS   = os.path.join(SCRIPT_DIR, "nrl_model_info.json")
MODEL_INFO_NO_ODDS = os.path.join(SCRIPT_DIR, "nrl_model_no_odds_info.json")

# ── Palette ──────────────────────────────────────────────────────────────────
BG      = "#1a1a2e"
PANEL   = "#16213e"
ACCENT  = "#0f3460"
GREEN   = "#00b894"
YELLOW  = "#fdcb6e"
RED     = "#d63031"
WHITE   = "#dfe6e9"
GREY    = "#636e72"


class NRLTipsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NRL Tips")
        self.setWindowIcon(QIcon(os.path.join(SCRIPT_DIR, "rugby_ball.svg")))
        self.resize(800, 700)
        self._process: QProcess | None = None
        self._pending_no_odds = False  # set True if we still need to run the no-odds model
        self._pending_export  = False  # set True after retrain to trigger export+push
        self._web_server = _subprocess.Popen(
            [sys.executable, os.path.join(SCRIPT_DIR, "web_gui.py")],
            cwd=SCRIPT_DIR,
        )
        self._build_ui()
        self._centre()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setStyleSheet(f"background-color: {BG}; color: {WHITE};")

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = QWidget()
        hdr.setStyleSheet(f"background-color: {ACCENT};")
        hdr_layout = QHBoxLayout(hdr)
        hdr_layout.setContentsMargins(20, 10, 20, 10)
        title = QLabel("🏉  NRL Tips")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {WHITE};")
        hdr_layout.addWidget(title)
        hdr_layout.addStretch()
        layout.addWidget(hdr)

        # ── Toolbar ───────────────────────────────────────────────────────────
        bar = QWidget()
        bar.setStyleSheet(f"background-color: {PANEL};")
        bar_outer = QVBoxLayout(bar)
        bar_outer.setContentsMargins(16, 8, 16, 8)
        bar_outer.setSpacing(6)

        self.btn_tips          = self._make_button("Tips (with Odds)", "#1565c0", WHITE, self._get_tips)
        self.btn_tips_no_odds  = self._make_button("Tips (No Odds)", "#1e88e5", "#000", self._get_tips_no_odds)
        self.btn_compare       = self._make_button("Compare Models", "#64b5f6", "#000", self._compare_models)
        self.btn_show_model        = self._make_button("Show Models", "#004d40", WHITE, self._show_model_both)
        self.btn_collect_new   = self._make_button("Collect New Data", "#00796b", WHITE, self._collect_new_data)
        self.btn_collect_all   = self._make_button("Collect All Data", "#00897b", "#000", self._collect_all_data)
        self.btn_train_all     = self._make_button("Retrain and Export All", "#26a69a", "#000", self._retrain_all)
        self.btn_cancel        = self._make_button("Cancel", RED, WHITE, self._cancel)
        self.btn_clear         = self._make_button("Clear", PANEL, GREY, self._clear)
        self.btn_cancel.hide()

        # Row 1: tips + perf + round input
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        row1.addWidget(self.btn_tips)
        row1.addWidget(self.btn_tips_no_odds)
        row1.addWidget(self.btn_compare)
        row1.addStretch()
        row1.addWidget(QLabel("Round:"))
        self.round_input = QLineEdit("auto")
        self.round_input.setFixedWidth(60)
        self.round_input.setStyleSheet(
            f"background:{ACCENT}; color:{WHITE}; border:none; padding:4px; border-radius:4px;"
        )
        row1.addWidget(self.round_input)

        # Row 2: data + train + cancel/clear
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        row2.addWidget(self.btn_show_model)
        row2.addWidget(self.btn_collect_new)
        row2.addWidget(self.btn_collect_all)
        row2.addWidget(self.btn_train_all)
        row2.addWidget(self.btn_cancel)
        row2.addWidget(self.btn_clear)
        row2.addStretch()

        bar_outer.addLayout(row1)
        bar_outer.addLayout(row2)
        layout.addWidget(bar)

        # ── Output ────────────────────────────────────────────────────────────
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Courier New", 11))
        self.output.setStyleSheet(
            f"background-color: {PANEL}; color: {WHITE}; "
            f"border: none; padding: 12px;"
        )
        layout.addWidget(self.output)

        # ── Status bar ────────────────────────────────────────────────────────
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(f"background-color: {ACCENT}; color: {WHITE};")
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(140)
        self.progress.setMaximumHeight(14)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet(
            f"QProgressBar {{ background:{ACCENT}; border:none; border-radius:3px; }}"
            f"QProgressBar::chunk {{ background:{GREEN}; border-radius:3px; }}"
        )
        self.progress.hide()
        self.status_bar.addPermanentWidget(self.progress)

    def _make_button(self, text, bg, fg, handler) -> QPushButton:
        btn = QPushButton(text)
        btn.setFont(QFont("Segoe UI", 11))
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(
            f"QPushButton {{ background:{bg}; color:{fg}; border:none; "
            f"padding:7px 16px; border-radius:5px; }}"
            f"QPushButton:hover {{ opacity:0.85; }}"
            f"QPushButton:disabled {{ background:{GREY}; color:#aaa; }}"
        )
        btn.clicked.connect(handler)
        return btn

    def closeEvent(self, event):
        self._web_server.terminate()
        self._web_server.wait()
        super().closeEvent(event)

    def _centre(self):
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width()  - self.width())  // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _get_tips(self):
        args = []
        rnd = self.round_input.text().strip()
        if rnd and rnd.lower() != "auto" and rnd.isdigit():
            args = ["--round", rnd]
        self._run(sys.executable, [AUTO_TIP] + args, "Fetching tips (with odds)...")

    def _get_tips_no_odds(self):
        args = ["--no-odds"]
        rnd = self.round_input.text().strip()
        if rnd and rnd.lower() != "auto" and rnd.isdigit():
            args += ["--round", rnd]
        self._run(sys.executable, [AUTO_TIP] + args, "Fetching tips (no odds)...")

    def _collect_new_data(self):
        """Fetch only new rounds + missing stats/squads — fast incremental update."""
        self._run(sys.executable, [COLLECT_CMD, "--new-only"], "Collecting new data...")

    def _collect_all_data(self):
        """Full data rebuild from scratch — slow, use for first setup or repairs."""
        self._run(sys.executable, [COLLECT_CMD], "Collecting all data...")

    def _check_performance(self):
        args = []
        rnd = self.round_input.text().strip()
        if rnd and rnd.lower() != "auto" and rnd.isdigit():
            args = ["--round", rnd]
        self._run(sys.executable, [PERF_CMD] + args, "Checking results (with odds)...")

    def _check_performance_no_odds(self):
        args = ["--no-odds"]
        rnd = self.round_input.text().strip()
        if rnd and rnd.lower() != "auto" and rnd.isdigit():
            args += ["--round", rnd]
        self._run(sys.executable, [PERF_CMD] + args, "Checking results (no odds)...")

    def _compare_models(self):
        args = ["--compare"]
        rnd = self.round_input.text().strip()
        if rnd and rnd.lower() != "auto" and rnd.isdigit():
            args += ["--round", rnd]
        self._run(sys.executable, [PERF_CMD] + args, "Comparing models...")

    def _show_model_both(self):
        import json
        self._clear()

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
        self._append_line(f"{'── With Odds ──':<{col_width}}  │  ── No Odds ──")
        self._append_line(sep)
        for i in range(n):
            left  = odds_lines[i]    if i < len(odds_lines)    else ""
            right = no_odds_lines[i] if i < len(no_odds_lines) else ""
            self._append_line(f"{left:<{col_width}}  │  {right}")

    def _retrain_all(self):
        self._pending_no_odds = True
        self._pending_export  = True
        self._run(sys.executable, [TRAIN_CMD, "--train"], "Training model (with odds)...")

    def _run(self, program: str, args: list, label: str, clear: bool = True):
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            self._append("[busy — already running, please wait]\n", RED)
            return
        if clear:
            self._clear()
        self._set_busy(label)

        env = self._process_env()
        self._process = QProcess(self)
        self._process.setWorkingDirectory(SCRIPT_DIR)
        self._process.setProcessEnvironment(env)
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.readyReadStandardError.connect(self._on_stderr)
        self._process.finished.connect(self._on_finished)
        self._process.start(program, args)

    def _process_env(self):
        from PyQt6.QtCore import QProcessEnvironment
        env = QProcessEnvironment.systemEnvironment()
        key = os.environ.get("ODDS_API_KEY", "")
        if key:
            env.insert("ODDS_API_KEY", key)
        return env

    def _on_stdout(self):
        data = self._process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        for line in data.splitlines():
            self._append_line(line)

    def _on_stderr(self):
        data = self._process.readAllStandardError().data().decode("utf-8", errors="replace")
        for line in data.splitlines():
            self._append(line + "\n", RED)

    def _cancel(self):
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.kill()
            self._append("\n[Cancelled]\n", RED)

    def _on_finished(self, exit_code, _):
        if exit_code in (-2, 9, 15):
            self._pending_no_odds = False
            self._pending_export  = False
            self._set_idle("Cancelled")
            return

        if exit_code != 0:
            self._pending_no_odds = False
            self._pending_export  = False
            self._set_idle(f"Finished with errors (exit {exit_code})")
            return

        # Successful finish — check if we still need to run no-odds model
        if self._pending_no_odds:
            self._pending_no_odds = False
            self._append("\n" + "─" * 50 + "\n", "#636e72")
            self._run(sys.executable, [TRAIN_CMD, "--train", "--no-odds"],
                      "Training model (no odds)...", clear=False)
            return

        # Successful finish — check if we should export + push to Android
        if self._pending_export:
            self._pending_export = False
            self._append("\n" + "─" * 50 + "\n", "#636e72")
            self._run(sys.executable, [EXPORT_CMD],
                      "Exporting models to Android & pushing to GitHub...",
                      clear=False)
        else:
            self._set_idle("Done")

    # ── Output helpers ────────────────────────────────────────────────────────

    def _append_line(self, line: str):
        colour = self._classify(line)
        bold   = any(x in line for x in ("NRL ", "TIPS", "===", "TIP:"))
        self._append(line + "\n", colour, bold=bold)

    def _classify(self, line: str) -> str:
        l = line.strip()
        if l.startswith("NRL ") or "TIPS" in l or l.startswith("==="):
            return YELLOW
        if l.startswith("TIP:"):
            return GREEN
        if "⚡" in l:
            return YELLOW
        if l.startswith("[") or "Could not" in l or re.search(r'\berror\b', l, re.IGNORECASE):
            return RED if re.search(r'\berror\b', l, re.IGNORECASE) else GREY
        return WHITE

    def _append(self, text: str, colour: str = WHITE, bold: bool = False):
        cursor = self.output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(colour))
        if bold:
            fmt.setFontWeight(QFont.Weight.Bold)
        cursor.setCharFormat(fmt)
        cursor.insertText(text)
        self.output.setTextCursor(cursor)
        self.output.ensureCursorVisible()

    def _clear(self):
        self.output.clear()

    def _set_busy(self, msg: str):
        self.status_label.setText(msg)
        for btn in (self.btn_tips, self.btn_tips_no_odds,
                    self.btn_collect_new, self.btn_collect_all,
                    self.btn_train_all):
            btn.setEnabled(False)
        self.btn_cancel.show()
        self.progress.setRange(0, 0)
        self.progress.show()

    def _set_idle(self, msg: str = "Ready"):
        self.status_label.setText(msg)
        for btn in (self.btn_tips, self.btn_tips_no_odds,
                    self.btn_collect_new, self.btn_collect_all,
                    self.btn_train_all):
            btn.setEnabled(True)
        self.btn_cancel.hide()
        self.progress.hide()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("nrl-tips")
    app.setStyle("Fusion")
    win = NRLTipsWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
