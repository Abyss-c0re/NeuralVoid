import asyncio
import json
import time

from typing import Optional, Union, List, Any, Dict

from textual.app import App, ComposeResult
from textual.widgets import Input, Markdown
from textual.containers import VerticalScroll
from textual.binding import Binding

from neuralcore.agents.core import Agent
from neuralcore.actions.actions import ActionSet
from neuralcore.actions.manager import DynamicActionManager


from neuralvoid.ui.rendering import set_renderer_app, get_renderer
from neuralvoid.ui.helpers import _build_tool_markdown, _format_block, _format_text


from neuralcore.utils.logger import Logger

ToolProvider = Union[ActionSet, DynamicActionManager, list[dict[str, Any]]]

logger = Logger.get_logger()


# ============================================================
# Message widget – clean status footer (no iteration spam)
# ============================================================


class Message(Markdown):
    """Markdown-rendered chat message with optional status footer."""

    def __init__(self, role: str, content: str = ""):
        self.role = role
        self.buffer = content
        self.status_line: str = ""
        super().__init__(self.render_markdown())

    def update_status(self, text: str) -> None:
        """Update only the tiny status line at the bottom."""
        self.status_line = text.strip()
        self.update(self.render_markdown())

    def clear_status(self) -> None:
        self.status_line = ""
        self.update(self.render_markdown())

    def render_markdown(self) -> str:
        if self.role == "user":
            prefix = "🧑 **You**: "
        elif self.role == "assistant":
            prefix = "🤖 **Assistant**: "
        elif self.role == "system":
            prefix = "💻 **System**: "
        else:
            prefix = ""

        main = prefix + self.buffer

        if self.status_line:
            # Tiny dimmed status (never clogs the main content)
            status = f"\n\n<span style='dim'>└─ {self.status_line}</span>"
            return main + status
        return main


# ============================================================
# Chat container
# ============================================================


class ChatView(VerticalScroll):
    def add(self, widget):
        self.mount(widget)
        self.scroll_end()


# ============================================================
# Main App
# ============================================================


class LLMChatApp(App):
    agent: Agent  # ← required by new queue-based Agent
    system_prompt: Optional[str]
    waiting_for_confirmation: bool = False
    pending_confirmation: Optional[dict] = None

    # Persistent agent runner (one single run() that lives forever)
    _agent_task: asyncio.Task | None = None
    _current_assistant_msg: Optional[Message] = None

    # Per-turn buffers (reset on every new assistant message)
    _current_pure_text: str = ""
    _current_tool_buffer: str = ""
    _last_stream_update: float = 0.0
    _last_finished: bool = True

    # UI constants
    UPDATE_INTERVAL = 0.08
    SPINNERS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    BINDINGS = [
        Binding("ctrl+l", "clear_chat", "Clear"),
        Binding("ctrl+c", "quit", "Quit"),
        Binding("escape", "stop_stream", "Stop generation", show=True, priority=True),
    ]

    CSS = """
    Screen {
        layout: vertical;
    }

    #chat {
        height: 1fr;
        padding: 1;
    }

    Input {
        dock: bottom;
    }
    """

    def __init__(
        self,
        agent: Agent,
        system_prompt: Optional[str] = None,
        tool_rendering: Optional[str] = "off",
        max_iterations: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_info_level: Optional[str] = "compact",
    ):
        super().__init__()
        self.agent = agent

        self.client = agent.client
        self.registry = agent.registry
        self.system_prompt = system_prompt
        self.context_manager = agent.context_manager
        self.conversation = []

        self.rendering = get_renderer()
        self.tool_rendering = tool_rendering
        self.max_iterations = max_iterations or getattr(
            agent.client, "max_iterations", 20
        )
        self.temperature = temperature or getattr(agent.client, "temperature", 0.7)
        self.max_tokens = max_tokens or getattr(agent.client, "max_tokens", 32000)
        self.tool_info_level = tool_info_level

        self._spinner_idx = 0

    def compose(self) -> ComposeResult:
        self.chat = ChatView(id="chat")
        yield self.chat
        yield Input(placeholder="Ask something...")

    async def on_mount(self):
        set_renderer_app(self, Message)
        await self.rendering.start_worker()

        self.chat.add(
            Message(
                "assistant",
                f"Connected to **{self.client.model}**\n\n"
                "Type a message and press **Enter**.\n"
                "Commands: **stop** / **cancel** → stop current stream\n"
                "**exit** → close app",
            )
        )

        # === NEW: Start the persistent queue-based agent runner once ===
        self._agent_task = asyncio.create_task(
            self._run_agent_forever(), name="agent-runner"
        )

        self.query_one(Input).focus()

    # In LLMChatApp class

    _agent_task: asyncio.Task | None = None
    _current_assistant_msg: Optional[Message] = None

    async def _run_agent_forever(self):
        """Single persistent consumer that creates a NEW assistant message for EVERY LLM turn."""
        try:
            async for event_type, payload in self.agent.run(
                user_prompt=None,
                system_prompt=self.system_prompt or "",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                chat_mode=True
            ):
                # Every time we see a new "llm_stream" or "content_delta" after a finish,
                # we start a fresh assistant bubble (important for chat loop)
                if event_type in ("phase_changed", "content_delta", "tool_start") and (
                    self._current_assistant_msg is None
                    or getattr(self, "_last_finished", False)
                ):
                    assistant_msg = Message("assistant", "")
                    self.chat.add(assistant_msg)
                    self._current_assistant_msg = assistant_msg
                    self._current_pure_text = ""
                    self._current_tool_buffer = ""
                    self._last_finished = False

                if self._current_assistant_msg is None:
                    continue

                await self._process_agent_event(
                    event_type, payload, self._current_assistant_msg
                )

                if event_type == "finish":
                    self._last_finished = True
                    self._current_assistant_msg = None  # allow new message on next turn

        except asyncio.CancelledError:
            logger.debug("Agent runner cancelled")
        except Exception as e:
            logger.exception("Agent runner crashed")
            self.chat.add(Message("system", f"❌ Runner error: {e}"))

    async def _ui_update(self, message: Message, immediate: bool = False) -> None:
        """Shared UI refresh helper (was _ui_update inside old stream_llm)."""
        now = time.time()
        if not immediate and now - self._last_stream_update < self.UPDATE_INTERVAL:
            return

        display = (self._current_pure_text + self._current_tool_buffer).replace(
            "[FINAL_ANSWER_COMPLETE]", ""
        )

        if message.buffer != display:
            message.buffer = display
            message.update(message.render_markdown())
            self.chat.scroll_end(animate=False)
            self._last_stream_update = now

    async def _process_agent_event(
        self, event_type: str, payload: Any, message: Message
    ) -> None:
        """All event handling logic from the old stream_llm, now running inside the persistent runner."""
        level = self.tool_info_level or "compact"

        # ── Phase changes ───────────────────────
        if event_type == "phase_changed":
            phase = payload.get("phase", "unknown")
            message.update_status(
                f"{self.SPINNERS[self._spinner_idx % len(self.SPINNERS)]} {phase.upper()}"
            )
            self._spinner_idx = (self._spinner_idx + 1) % len(self.SPINNERS)
            return

        # ── Planning phase ─────────────────────────────────────
        elif event_type == "planning_complete":
            steps = payload.get("steps", [])
            goal = payload.get("goal", "")
            self._current_pure_text += (
                f"\n**Planning complete**\nGoal: {goal}\n\n**Steps:**\n"
            )
            for i, step in enumerate(steps, 1):
                self._current_pure_text += f"{i}. {step}\n"
            self._current_pure_text += "\n"
            await self._ui_update(message, immediate=True)
            return

        # ── Streaming assistant content ──────────────────────────────
        elif event_type == "content_delta":
            self._current_pure_text += payload
            await self._ui_update(message)
            return


        elif event_type == "llm_response":
            full_reply = payload.get("full_reply", "").strip()
            if full_reply:
                self._current_pure_text += full_reply
                await self._ui_update(message, immediate=True)

                # Also clear any lingering status
                message.clear_status()
            return

        # ── Tool call delta (streaming args) ─────────────────────────
        elif event_type == "tool_call_delta":
            func = payload.get("function", {})
            name = func.get("name") or payload.get("name") or "unknown"
            args_str = func.get("arguments") or payload.get("arguments_delta") or ""

            try:
                args_dict = json.loads(args_str) if args_str.strip() else {}
            except Exception:
                args_dict = {"_partial": args_str}

            md = _build_tool_markdown(
                name=name,
                args=args_dict,
                level=level,
                result=None,
                confirmation=None,
                error=False,
            )
            self._current_tool_buffer += md
            await self._ui_update(message, immediate=True)

            message.update_status(
                f"{self.SPINNERS[self._spinner_idx % len(self.SPINNERS)]} using **{name}**"
            )
            self._spinner_idx = (self._spinner_idx + 1) % len(self.SPINNERS)
            return

        # ── Tool batch start ─────────────────────────────────────────
        elif event_type == "tool_calls":
            count = len(payload) if isinstance(payload, (list, tuple)) else "?"
            message.update_status(f"Executing {count} tool(s)...")
            return

        # ── Tool start ─────────────────────
        elif event_type == "tool_start":
            name = payload.get("name", "unknown")
            args = payload.get("args", {})
            md = _build_tool_markdown(
                name=name,
                args=args,
                level=level,
                result=None,
                confirmation=None,
                error=False,
            )
            self._current_tool_buffer += md
            await self._ui_update(message, immediate=True)
            return

        # ── Tool result ──────────────────────────────────────────────
        elif event_type == "tool_result":
            name = payload.get("name", "unknown")
            result = str(payload.get("result", ""))
            error = payload.get("error", False)

            md = _build_tool_markdown(
                name=name,
                args=payload.get("args", {}),
                result=result,
                level=level,
                error=error,
            )
            self._current_tool_buffer += md
            await self._ui_update(message, immediate=True)

            status = "❌ failed" if error else "completed"
            message.update_status(f"Tool **{name}** {status}")
            return

        # ── Confirmation required ──────────────────
        elif event_type == "needs_confirmation":
            tool_name = payload.get("name", "unknown")
            md = _build_tool_markdown(
                name=tool_name,
                args=payload.get("args", {}),
                confirmation=payload.get("preview", ""),
                level=level,
            )
            self._current_tool_buffer += md
            await self._ui_update(message, immediate=True)

            message.update_status("⏳ Waiting for your confirmation")

            # Attach the current UI message so _handle_confirmation_response can update it
            self.pending_confirmation = {**payload, "assistant_msg": message}
            self.waiting_for_confirmation = True
            return

        # ── Reflection ───────────────────────────────────────────────
        elif event_type == "reflection_triggered":
            self._current_pure_text += (
                f"\n\n🤔 **Self-Reflection**\n{payload.strip()}\n\n"
            )
            await self._ui_update(message, immediate=True)
            return

        # ── Final summary report ─────────────────────────────────────
        elif event_type == "final_summary":
            self._current_pure_text += f"\n\n{payload}\n"
            await self._ui_update(message, immediate=True)
            return

        # ── Finish signal ────────────────────────────────────────────
        elif event_type == "finish":
            reason = payload.get("reason", "unknown")

            if reason == "casual_complete":
                self._current_pure_text = self._current_pure_text.strip()
            else:
                if self._current_tool_buffer.strip():
                    self._current_tool_buffer = (
                        "\n\n─── 🔧 Tool usage history ───\n\n"
                        + self._current_tool_buffer
                    )

            await self._ui_update(message, immediate=True)

            final_content = (
                self._current_pure_text + self._current_tool_buffer
            ).strip()
            if final_content:
                self.conversation.append(
                    {"role": "assistant", "content": final_content}
                )

            message.clear_status()

            # Reset for next turn
            self._current_assistant_msg = None
            self._current_pure_text = ""
            self._current_tool_buffer = ""
            return

        # ── Errors / cancellations / warnings ────────────────────────
        elif event_type in ("cancelled", "error", "warning"):
            icon = {"cancelled": "🛑", "error": "❌", "warning": "⚠️"}.get(
                event_type, "⚠️"
            )
            self._current_pure_text += (
                f"\n\n{icon} **{event_type.capitalize()}**\n{payload}\n"
            )
            await self._ui_update(message, immediate=True)

            if event_type in ("cancelled", "error"):
                message.clear_status()
                self._current_assistant_msg = None
                self._current_pure_text = ""
                self._current_tool_buffer = ""
            return

    # ==================== CONFIRMATION (still needed for dangerous tools) ====================
    async def _handle_confirmation_response(self, user_input: str) -> bool:
        if not self.waiting_for_confirmation or not self.pending_confirmation:
            return False

        approved = user_input.strip().upper() in {"YES", "Y", "OK", "CONFIRM"}
        info = self.pending_confirmation

        tool_call_id = info["tool_call_id"]
        name = info["name"]
        args = info["args"]
        action = info["action"]
        assistant_msg = info["assistant_msg"]
        original_tool_calls = info.get("tool_calls", [])

        if approved:
            try:
                real_executor = action.executor
                if asyncio.iscoroutinefunction(real_executor):
                    result = await real_executor(**args)
                else:
                    result = real_executor(**args)
            except Exception as exc:
                result = f"Error during confirmed execution: {exc}"
            content = str(result)
        else:
            content = "User denied the action."

        # === NEW: Tell the running workflow via control instead of restarting a stream ===
        await self.agent.post_control(
            {
                "event": "needs_confirmation",  # workflow recognises this as confirmation response
                "approved": approved,
                "content": content,
                "tool_call_id": tool_call_id,
                "name": name,
            }
        )

        feedback = f"→ Action **{name}** was {'approved' if approved else 'denied'}.\nResult: {content}"
        assistant_msg.buffer += f"\n\n{feedback}"
        assistant_msg.update(assistant_msg.render_markdown())
        self.chat.scroll_end()

        self.waiting_for_confirmation = False
        self.pending_confirmation = None

        # No more stream_llm restart – the persistent runner continues automatically
        return True

    # ==================== USER INPUT ====================
    async def on_input_submitted(self, event: Input.Submitted):
        value = event.value.strip()
        event.input.value = ""
        if not value:
            return

        cmd = value.lower()
        if cmd in ("stop", "cancel"):
            await self.action_stop_stream()
            return

        if cmd == "exit":
            self.chat.add(Message("system", "👋 Exiting..."))
            await asyncio.sleep(0.3)
            self.exit()
            return

        if await self._handle_confirmation_response(value):
            return

        # Always show the user message immediately
        self.chat.add(Message("user", value))
        self.conversation.append({"role": "user", "content": value})

        # Create fresh assistant message for this turn
        assistant_msg = Message("assistant", "")
        self.chat.add(assistant_msg)

        # Reset per-turn buffers
        self._current_assistant_msg = assistant_msg
        self._current_pure_text = ""
        self._current_tool_buffer = ""

        # === NEW: Communicate via queue instead of calling run() with a prompt ===
        await self.agent.post_message(value)

    # ==================== STOP / CANCEL ====================
    async def action_stop_stream(self) -> None:
        if self._current_assistant_msg is None:
            self.notify("No active generation to stop.", timeout=2.5)
            return

        # Tell the running workflow via control (Workflow._drain_control will pick it up)
        await self.agent.post_control({"event": "cancelled"})
        self.notify("🛑 Generation cancelled", severity="warning", timeout=4)

    # ==================== CLEAR CHAT ====================
    async def action_clear_chat(self):
        self.chat.remove_children()
        self.conversation.clear()
        self.waiting_for_confirmation = False
        self.pending_confirmation = None
        self._current_assistant_msg = None
        self._current_pure_text = ""
        self._current_tool_buffer = ""
