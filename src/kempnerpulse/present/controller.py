"""Keyboard command handling and the raw-terminal context manager.

:class:`CommandController` owns the view state the dashboard reads (focused GPU,
plot/job mode, fleet scroll offset, the ``:``-command buffer) and parses raw
stdin bytes — including CSI arrow/page escape sequences — into state changes.
:func:`cbreak_stdin` puts the terminal into cbreak mode so single keystrokes are
delivered without a newline; outside a TTY it is a no-op.
"""
from __future__ import annotations

import select
import sys
from contextlib import contextmanager
from typing import Optional, Set

# termios / tty are POSIX-only; the interactive paths are no-ops without a TTY,
# so tolerate their absence (e.g. on Windows / restricted runtimes).
try:
    import termios
    import tty
except ImportError:  # pragma: no cover - non-POSIX fallback
    termios = None  # type: ignore[assignment]
    tty = None  # type: ignore[assignment]

# Card-rows moved per PgUp / PgDn.
SCROLL_PAGE = 3


class CommandController:
    """View-mode and input state shared between the input loop and the renderer."""

    def __init__(self, initial_focus: Optional[str] = None):
        self.focus_gpu: Optional[str] = initial_focus
        self.line_mode = False
        self.jobs_mode = False
        self.command_mode = False
        self.buffer = ""
        self.should_exit = False
        self.last_message = ""
        self.fleet_scroll_offset = 0

    def hint(self) -> str:
        if self.command_mode:
            return f":{self.buffer}"
        return "Type :focus <gpu>, :plot, :job, :q, or :exit"

    def scroll_fleet(self, delta: int) -> None:
        """Adjust the fleet scroll offset; the renderer clamps the upper bound."""
        self.fleet_scroll_offset = max(0, self.fleet_scroll_offset + delta)

    def handle_input(self, available_gpu_ids: Set[str]) -> None:
        """Drain any pending stdin and apply it; a no-op without a TTY."""
        if not sys.stdin.isatty():
            return
        chunk = ""
        while True:
            rlist, _, _ = select.select([sys.stdin], [], [], 0)
            if not rlist:
                break
            ch = sys.stdin.read(1)
            if not ch:
                break
            chunk += ch
        if chunk:
            self._process_chunk(chunk, available_gpu_ids)

    def _process_chunk(self, chunk: str, available_gpu_ids: Set[str]) -> None:
        # Parse char by char, recognizing CSI escape sequences (arrows / page keys)
        # which arrive as ESC [ <code>. Everything else goes to _process_char.
        i, n = 0, len(chunk)
        while i < n:
            ch = chunk[i]
            if ch == "\x1b" and chunk[i + 1:i + 2] == "[":
                code = chunk[i + 2:i + 3]
                if not self.command_mode:
                    paged = chunk[i + 3:i + 4] == "~"   # PgUp/PgDn require the trailing '~'
                    if code == "A":            # up arrow
                        self.scroll_fleet(-1)
                    elif code == "B":          # down arrow
                        self.scroll_fleet(1)
                    elif code == "5" and paged:        # PgUp  (ESC[5~)
                        self.scroll_fleet(-SCROLL_PAGE)
                    elif code == "6" and paged:        # PgDn  (ESC[6~)
                        self.scroll_fleet(SCROLL_PAGE)
                # advance past the sequence (4 bytes for ESC[5~ / ESC[6~, else 3)
                i += 4 if code in ("5", "6") and chunk[i + 3:i + 4] == "~" else 3
                continue
            self._process_char(ch, available_gpu_ids)
            i += 1

    def _process_char(self, ch: str, available_gpu_ids: Set[str]) -> None:
        if not self.command_mode:
            if ch == ":":
                self.command_mode = True
                self.buffer = ""
                self.last_message = ""
            elif ch in ("j", "J"):
                self.scroll_fleet(1)
            elif ch in ("k", "K"):
                self.scroll_fleet(-1)
            return

        if ch in ("\r", "\n"):
            self._execute_command(self.buffer.strip(), available_gpu_ids)
            self.command_mode = False
            self.buffer = ""
            return
        if ch in ("\x1b",):
            self.command_mode = False
            self.buffer = ""
            self.last_message = ""
            return
        if ch in ("\x7f", "\b"):
            self.buffer = self.buffer[:-1]
            return
        if ch == "\x03":
            self.should_exit = True
            return
        if ch.isprintable():
            self.buffer += ch

    def _execute_command(self, cmd: str, available_gpu_ids: Set[str]) -> None:
        if not cmd:
            return
        lower = cmd.lower()
        if lower in {"q", "quit"}:
            if self.line_mode:
                self.line_mode = False
                self.last_message = "Returned to fleet view"
            elif self.jobs_mode:
                self.jobs_mode = False
                self.last_message = "Returned to fleet view"
            elif self.focus_gpu is not None:
                self.focus_gpu = None
                self.last_message = "Returned to fleet view"
            else:
                self.should_exit = True
            return
        if lower == "exit":
            self.should_exit = True
            return
        if lower == "plot":
            self.line_mode = True
            self.jobs_mode = False
            self.focus_gpu = None
            self.last_message = "Plot view"
            return
        if lower == "job":
            self.jobs_mode = True
            self.line_mode = False
            self.focus_gpu = None
            self.last_message = "Job view"
            return
        if lower.startswith("focus"):
            parts = cmd.split()
            if len(parts) != 2:
                self.last_message = "Usage: :focus <gpu_id>"
                return
            gpu_id = parts[1]
            if gpu_id not in available_gpu_ids:
                self.last_message = f"GPU {gpu_id} is not visible"
                return
            self.line_mode = False
            self.jobs_mode = False
            self.focus_gpu = gpu_id
            self.last_message = f"Focused GPU {gpu_id}"
            return
        self.last_message = f"Unknown command: {cmd}"


@contextmanager
def cbreak_stdin(enabled: bool):
    """Put stdin into cbreak mode for single-keystroke input; no-op off-TTY."""
    if not enabled or termios is None or tty is None or not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
