import asyncio
import json

from typing import Optional, Union, List, Any, Dict


from textual.app import App, ComposeResult
from textual.widgets import Input, Markdown
from textual.containers import VerticalScroll
from textual.binding import Binding

from neuralcore.core.client import LLMClient
from neuralcore.actions.actions import ActionSet
from neuralcore.actions.manager import DynamicActionManager

from neuralvoid.ui.rendering import set_renderer_app, get_renderer

from neuralcore.cognition.memory import ContextManager
from neuralcore.agents.agent_core import AgentRunner
from neuralcore.utils.logger import Logger

ToolProvider = Union[ActionSet, DynamicActionManager, list[dict[str, Any]]]

logger = Logger.get_logger()

# ============================================================
# Message widget
# ============================================================


class Message(Markdown):
    """Markdown-rendered chat message."""

    def __init__(self, role: str, content: str = ""):
        self.role = role
        self.buffer = content
        super().__init__(self.render_markdown())

    def render_markdown(self) -> str:

        if self.role == "user":
            prefix = "🧑 **You**: "
        elif self.role == "assistant":
            prefix = "🤖 **Assistant**: "
        elif self.role == "system":
            prefix = "💻 **System**: "
        else:
            prefix = ""

        return prefix + self.buffer


# ============================================================
# Chat container
# ============================================================


class ChatView(VerticalScroll):
    """Scrollable chat history."""

    def add(self, widget):
        self.mount(widget)
        self.scroll_end()


# ============================================================
# Main App
# ============================================================


class LLMChatApp(App):
    client: LLMClient
    system_prompt: Optional[str]
    waiting_for_confirmation: bool = False
    pending_confirmation: Optional[dict] = None
    UPDATE_INTERVAL = 0.01

    BINDINGS = [
        Binding("ctrl+l", "clear_chat", "Clear"),
        Binding("ctrl+c", "quit", "Quit"),
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
        client: LLMClient,
        system_prompt: Optional[str] = None,
        tools: Optional[
            Union[ActionSet, "DynamicActionManager", List[Dict[str, Any]]]
        ] = None,
        context_manager: Optional[ContextManager] = None,
        tool_rendering: Optional[str] = "off",
        max_iterations: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        super().__init__()

        self.client = client
        self.system_prompt = system_prompt
        self.tools = tools
        self.context_manager = context_manager
        self.conversation = []

        self.rendering = get_renderer()
        self.tool_rendering = tool_rendering  # off | info | full
        self.max_iterations = (
            max_iterations
            if max_iterations is not None
            else getattr(client, "max_iterations", 100)
        )
        self.temperature = (
            temperature
            if temperature is not None
            else getattr(client, "temperature", 0.3)
        )
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else getattr(client, "max_tokens", 32000)
        )

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

        # ← NEW: Auto-focus input on startup
        self.query_one(Input).focus()

    async def _handle_confirmation_response(self, user_input: str) -> bool:
        """
        Returns True if this input was treated as confirmation answer
        """
        if not self.waiting_for_confirmation:
            return False

        approved = user_input.strip().upper() in {"YES", "Y", "OK", "CONFIRM"}

        info = self.pending_confirmation
        if not info:
            self.waiting_for_confirmation = False
            return False

        tool_call_id = info["tool_call_id"]
        name = info["name"]
        args = info["args"]
        action = info["action"]  # ← the Action object
        assistant_msg = info["assistant_msg"]
        original_tool_calls = info["tool_calls"]

        if approved:
            try:
                real_executor = action.executor  # ← this is the callable we want

                if asyncio.iscoroutinefunction(real_executor):
                    result = await real_executor(**args)
                else:
                    result = real_executor(**args)

            except Exception as exc:
                result = f"Error during confirmed execution: {exc}"

            content = str(result)
        else:
            content = "User denied the action."

        # Now we can finally send the tool result back to the model
        tool_result_msg = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content,
        }

        # Append the assistant's tool call request + our result
        self.conversation.append(
            {"role": "assistant", "tool_calls": original_tool_calls}
        )
        self.conversation.append(tool_result_msg)

        # Show feedback in UI
        feedback = f"→ Action **{name}** was {'approved' if approved else 'denied'}.\nResult: {content}"
        assistant_msg.buffer += f"\n\n{feedback}"
        assistant_msg.update(assistant_msg.render_markdown())
        self.chat.scroll_end()

        # Reset confirmation state
        self.waiting_for_confirmation = False
        self.pending_confirmation = None

        # Continue LLM generation with the new tool result in history
        asyncio.create_task(
            self.stream_llm(
                prompt="",  # we continue from history
                message=assistant_msg,  # continue in same bubble or new one
                tools=self.tools,
            )
        )

        return True

    async def on_input_submitted(self, event: Input.Submitted):
        value = event.value.strip()
        event.input.value = ""  # clear immediately
        # self.query_one("#input", Input).focus()  # keep focus for next input

        if not value:
            return

        cmd = value.lower()

        # ── Special commands ─────────────────────────────────────
        if cmd in ("stop", "cancel"):
            if self.client.stop_stream():
                self.chat.add(
                    Message(
                        "system",
                        "🛑 **Stop signal sent** — current stream will end soon.",
                    )
                )
            else:
                self.chat.add(Message("system", "ℹ️ No active stream to stop."))
            return  # do NOT treat as user message

        elif cmd == "exit":
            self.chat.add(Message("system", "👋 Exiting..."))
            await asyncio.sleep(0.3)  # let the message appear
            self.exit()
            return

        # ── Confirmation handling (unchanged) ────────────────────
        if await self._handle_confirmation_response(value):
            return

        # ── Normal user message flow (unchanged) ─────────────────
        self.chat.add(Message("user", value))
        self.conversation.append({"role": "user", "content": value})

        assistant_msg = Message("assistant", "")
        self.chat.add(assistant_msg)

        asyncio.create_task(self.stream_llm(value, assistant_msg, self.tools))

    def _build_tool_markdown(
        self,
        name: str,
        args: dict,
        result: Optional[str] = None,
        confirmation: Optional[str] = None,
        error: bool = False,
        error_message: Optional[str] = None,
    ) -> str:
        """Pure function: returns markdown to append. No UI calls."""
        level = getattr(self, "tool_rendering", "off")
        if level == "off":
            return ""

        parts: list[str] = []

        if result is None and confirmation is None:
            if level in ("info", "full"):
                parts.append(
                    f"\n\n🔧 **Calling tool:** `{name}`\n"
                    f"```json\n{json.dumps(args, indent=2)}\n```"
                )

        if confirmation is not None:
            parts.append(
                f"\n⚠ **Confirmation required**\n{confirmation}\n\n"
                f"Type **YES** to approve."
            )

        if result is not None:
            if error:
                if level in ("info", "full"):
                    msg = error_message or result
                    parts.append(f"\n❌ **Tool `{name}` failed**\n```\n{msg}\n```")
            else:
                if level == "full":
                    parts.append(f"\n✅ **Result from `{name}`:**\n```\n{result}\n```")

        return "".join(parts)

    async def stream_llm(
        self,
        prompt: str,
        message: Message,
        tools: Optional[ToolProvider] = None,
    ) -> None:
        import time

        logger.info("🚀 stream_llm START | prompt=%s", prompt[:200])

        assistant_msg = message
        text_buffer: str = ""
        self._last_stream_update = time.time() - 0.1  # force first update

        # Prepare messages (unchanged)
        messages = [
            {"role": "system", "content": self.system_prompt or ""}
        ] + self.conversation[-15:]

        # Context enrichment (unchanged)
        if self.context_manager:
            try:
                context_history = await self.context_manager.generate_prompt(
                    prompt, num_messages=0
                )
                if context_history and context_history[-1].get("content"):
                    last_msg = context_history[-1]["content"]
                    if "\nUser query:" in last_msg:
                        ctx_part = last_msg.split("\nUser query:")[0].strip()
                        if ctx_part:
                            messages.insert(
                                1,
                                {
                                    "role": "system",
                                    "content": f"Relevant context:\n{ctx_part}",
                                },
                            )
            except Exception as e:
                logger.error("❌ History enrichment failed: %s", e, exc_info=True)

        runner = AgentRunner(
            client=self.client,
            max_iterations=self.max_iterations,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        async def _ui_update(new_buffer: str, immediate: bool = False):
            """Single place that decides when to render."""
            now = time.time()

            # Debounce normal streaming text
            if not immediate and now - self._last_stream_update < self.UPDATE_INTERVAL:
                return

            # Prevent duplicate renders
            if assistant_msg.buffer != new_buffer:
                assistant_msg.buffer = new_buffer
                assistant_msg.update(assistant_msg.render_markdown())
                self.chat.scroll_end(animate=False)
                self._last_stream_update = now

        try:
            async for event_type, payload in runner.run(
                user_prompt=prompt,
                messages_so_far=messages,
                tools=tools or [],
                system_prompt=self.system_prompt or "",
                context_manager=self.context_manager,
            ):
                if event_type == "content_delta":
                    text_buffer += payload
                    await _ui_update(text_buffer, immediate=False)

                elif event_type == "tool_start":
                    tool_md = self._build_tool_markdown(
                        name=payload["name"],
                        args=payload.get("args", {}),
                    )
                    text_buffer += tool_md
                    await _ui_update(text_buffer, immediate=True)

                elif event_type == "tool_result":
                    tool_md = self._build_tool_markdown(
                        name=payload["name"],
                        args=payload.get("args", {}),
                        result=str(payload.get("result", "")),
                        error=payload.get("error", False),
                    )
                    text_buffer += tool_md
                    await _ui_update(text_buffer, immediate=True)

                elif event_type == "needs_confirmation":
                    tool_md = self._build_tool_markdown(
                        name=payload["name"],
                        args=payload["args"],
                        confirmation=payload.get("preview", ""),
                    )
                    text_buffer += tool_md
                    await _ui_update(text_buffer, immediate=True)

                    self.waiting_for_confirmation = True
                    self.pending_confirmation = {...}  # your existing dict (unchanged)
                    return  # pause stream

                elif event_type == "final_answer":
                    final_text = str(payload).strip()
                    # FINAL OVERRIDE → guarantees no double text
                    text_buffer = final_text
                    await _ui_update(text_buffer, immediate=True)
                    self.conversation.append({"role": "assistant", "content": payload})

                elif event_type == "cancelled":
                    text_buffer += f"\n\n🛑 **Stream cancelled** — {payload or 'user requested stop'}"
                    await _ui_update(text_buffer, immediate=True)
                    return

                elif event_type == "error":
                    text_buffer += f"\n\n❌ **Error:** {payload}"
                    await _ui_update(text_buffer, immediate=True)
                    return

                elif (
                    event_type == "finish"
                    and payload.get("reason") == "max_iterations_reached"
                ):
                    text_buffer += "\n\n⚠️ Max iterations reached."
                    await _ui_update(text_buffer, immediate=True)

                # Optional fallback
                if (
                    event_type in ("tool_start", "tool_result")
                    and not text_buffer.strip()
                ):
                    text_buffer = "Thinking…\n\n"
                    await _ui_update(text_buffer, immediate=True)

        except asyncio.CancelledError:
            text_buffer += "\n\n🛑 **Agent loop cancelled**"
            await _ui_update(text_buffer, immediate=True)

        except Exception as exc:
            logger.exception("🔥 Unexpected crash in stream_llm")
            text_buffer += f"\n\n❌ **Unexpected error:** {exc}"
            await _ui_update(text_buffer, immediate=True)

        logger.info("✅ stream_llm END")

    async def action_clear_chat(self):
        self.chat.remove_children()
