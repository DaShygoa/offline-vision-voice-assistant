"""Text command input — for machines without a microphone.

Two layers:
  1) In-frame bar (OpenCV) — always drawn, shows live text / last command
  2) Optional Tkinter window (default) — real Chinese IME on Windows

OpenCV-only fallback: press `t` / `/` / Enter to focus the in-frame box,
type ASCII, Backspace deletes, Enter submits, Esc cancels.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Optional

log = logging.getLogger(__name__)


class TextInputWindow:
    """Always-on-top Tk Entry. Safe to call submit from the Tk thread."""

    def __init__(self, on_submit: Callable[[str], None], title: str = "文本指令（无麦克风）"):
        self.on_submit = on_submit
        self.title = title
        self._root = None
        self._entry = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._echo = ""          # last committed text for OpenCV overlay
        self._live = ""          # what user is typing now
        self._lock = threading.Lock()
        self.available = False

    # ---------------------------------------------------------------- echo
    @property
    def display_text(self) -> str:
        with self._lock:
            return self._live or self._echo

    @property
    def is_typing(self) -> bool:
        with self._lock:
            return bool(self._live)

    @property
    def has_focus_content(self) -> bool:
        return bool(self.display_text)

    def set_echo(self, text: str) -> None:
        with self._lock:
            self._echo = text
            self._live = ""

    # ------------------------------------------------------------ lifecycle
    def start(self) -> bool:
        try:
            import tkinter as tk
            from tkinter import font as tkfont
        except Exception as e:
            log.warning("Tkinter unavailable (%s) — use OpenCV keys t/Enter to type", e)
            self.available = False
            return False

        self._running = True
        ready = threading.Event()
        err: list[BaseException] = []

        def _run():
            try:
                root = tk.Tk()
                self._root = root
                root.title(self.title)
                root.attributes("-topmost", True)
                root.resizable(False, False)
                # compact size, bottom-right-ish
                root.update_idletasks()
                sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
                ww, wh = 460, 78
                root.geometry(f"{ww}x{wh}+{max(0, sw - ww - 40)}+{max(0, sh - wh - 120)}")

                frame = tk.Frame(root, padx=8, pady=6)
                frame.pack(fill="both", expand=True)

                hint = tk.Label(
                    frame,
                    text="输入中文指令后回车 例：这是什么 / 记住，这个叫… / 帮我找…",
                    anchor="w",
                    font=("Microsoft YaHei UI", 9),
                )
                hint.pack(anchor="w", fill="x")

                entry = tk.Entry(
                    frame,
                    font=("Microsoft YaHei UI", 12),
                    relief="solid",
                    bd=1,
                )
                entry.pack(fill="x", pady=(4, 0), ipady=4)
                self._entry = entry

                def _on_key(event):
                    # live echo for OpenCV bar
                    with self._lock:
                        self._live = entry.get()
                    if event.keysym == "Return":
                        text = entry.get().strip()
                        entry.delete(0, "end")
                        with self._lock:
                            self._live = ""
                            if text:
                                self._echo = text
                        if text:
                            try:
                                self.on_submit(text)
                            except Exception as ex:
                                log.error("text submit failed: %s", ex)
                        return "break"
                    if event.keysym == "Escape":
                        entry.delete(0, "end")
                        with self._lock:
                            self._live = ""
                        return "break"
                    return None

                def _on_modified(_event=None):
                    with self._lock:
                        self._live = entry.get()

                entry.bind("<Key>", _on_key)
                entry.bind("<KeyRelease>", _on_modified)
                entry.focus_set()

                # hide button row: clear only
                btns = tk.Frame(frame)
                btns.pack(anchor="e", pady=(4, 0))

                def _clear():
                    entry.delete(0, "end")
                    with self._lock:
                        self._live = ""

                def _send():
                    text = entry.get().strip()
                    entry.delete(0, "end")
                    with self._lock:
                        self._live = ""
                        if text:
                            self._echo = text
                    if text:
                        self.on_submit(text)

                tk.Button(btns, text="清空", command=_clear, width=6, font=("Microsoft YaHei UI", 9)).pack(side="right", padx=(6, 0))
                tk.Button(btns, text="发送", command=_send, width=6, font=("Microsoft YaHei UI", 9)).pack(side="right")

                def _on_close():
                    self._running = False
                    try:
                        root.destroy()
                    except Exception:
                        pass

                root.protocol("WM_DELETE_WINDOW", _on_close)
                self.available = True
                ready.set()
                while self._running:
                    root.update()
                    root.after(20)
                try:
                    root.destroy()
                except Exception:
                    pass
            except BaseException as e:  # noqa: BLE001 — report any Tk failure
                err.append(e)
                ready.set()

        self._thread = threading.Thread(target=_run, name="text-input", daemon=True)
        self._thread.start()
        if not ready.wait(timeout=5.0):
            log.warning("Text input window did not open in time")
            self.available = False
            return False
        if err:
            log.warning("Text input window failed: %s", err[0])
            self.available = False
            return False
        log.info("Text input window ready (IME Chinese)")
        return True

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
