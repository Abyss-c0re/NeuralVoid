# src/utils/rendering.py
import asyncio
from typing import Optional, Tuple


class Rendering:
    """
    Centralized helper to print into the TUI chat interface.
    Supports streaming, auto-scroll, and cancelable messages.
    """

    def __init__(self, app=None, message_class=None):
        self.app = app
        self.message_class = message_class
        self.queue: asyncio.Queue[Tuple[str, str]] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._active_streams: set[asyncio.Task] = set()

    # --------------------------
    # Worker for queued messages
    # --------------------------
    async def start_worker(self):
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def _worker_loop(self):
        while True:
            role, text = await self.queue.get()
            if text is None:
                break
            if self.app and self.message_class:
                msg = self.message_class(role, text)
                self.app.chat.add(msg)
                self._maybe_scroll()
            else:
                print(f"[{role}] {text}")

    # --------------------------
    # Enqueue messages
    # --------------------------
    async def enqueue(self, text: str, role: str = "system", stream: bool = False):
        if stream:
            task = asyncio.create_task(self._stream_to_chat(role, text))
            self._active_streams.add(task)
            task.add_done_callback(lambda t: self._active_streams.discard(t))
        else:
            await self.queue.put((role, text))

    # --------------------------
    # Streaming helper
    # --------------------------
    async def _stream_to_chat(
        self, role: str, text: str, batch_size: int = 50, delay: float = 0.02
    ):
        if not self.app or not self.message_class:
            print(f"[{role}] {text}")
            return

        msg = self.message_class(role, "")
        self.app.chat.add(msg)
        buffer = ""

        for i in range(0, len(text), batch_size):
            if any(t.cancelled() for t in self._active_streams):
                break
            buffer += text[i : i + batch_size]
            msg.buffer = buffer
            msg.update(msg.render_markdown())
            self._maybe_scroll()
            await asyncio.sleep(delay)

        if msg.buffer != "":
            msg.update(msg.render_markdown())

    # --------------------------
    # Public streaming method
    # --------------------------
    async def stream_message(
        self,
        content: str,
        role: str = "assistant",
        batch_size: int = 50,
        delay: float = 0.02,
    ):
        """
        Gradually stream a long message to the chat.
        """
        if not self.app or not self.message_class:
            print(f"[{role}] {content}")
            return

        task = asyncio.create_task(
            self._stream_to_chat(role, content, batch_size=batch_size, delay=delay)
        )
        self._active_streams.add(task)
        task.add_done_callback(lambda t: self._active_streams.discard(t))
        await task

    # --------------------------
    # Auto-scroll helper
    # --------------------------
    def _maybe_scroll(self):
        if self.app:
            try:
                if self.app.chat.scroll_end_offset < 3:
                    self.app.chat.scroll_end()
            except Exception:
                self.app.chat.scroll_end()

    # --------------------------
    # Simple sync printers
    # --------------------------
    def printer(self, content: str, system: bool = False):
        role = "system" if system else "assistant"
        if self.app:
            try:
                asyncio.get_running_loop()
                asyncio.create_task(self.enqueue(content, role=role))
            except RuntimeError:
                asyncio.run(self.enqueue(content, role=role))
        else:
            print(f"[{role}] {content}")

    def printer_stream(
        self,
        content: str,
        system: bool = False,
        batch_size: int = 50,
        delay: float = 0.02,
    ):
        role = "system" if system else "assistant"
        if self.app:
            try:
                asyncio.get_running_loop()
                asyncio.create_task(
                    self.stream_message(
                        content, role=role, batch_size=batch_size, delay=delay
                    )
                )
            except RuntimeError:
                asyncio.run(
                    self.stream_message(
                        content, role=role, batch_size=batch_size, delay=delay
                    )
                )
        else:
            print(f"[{role}] {content}")


# --------------------------
# Singleton instance
# --------------------------
_renderer: Optional[Rendering] = None


def get_renderer() -> Rendering:
    global _renderer
    if _renderer is None:
        _renderer = Rendering()
    return _renderer


def set_renderer_app(app, message_class):
    global _renderer
    if _renderer is None:
        _renderer = Rendering(app=app, message_class=message_class)
    else:
        _renderer.app = app
        _renderer.message_class = message_class
