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

    def render_tool_call(
        self,
        message: Message,
        name: str,
        args: dict,
        result: Optional[str] = None,
        confirmation: Optional[str] = None,
        error: bool = False,
        error_message: Optional[str] = None,
    ):
        level = getattr(self, "tool_rendering", "off")
        if level == "off":
            return

        parts = []

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

        if parts:
            # Append instead of full replace → smoother
            message.buffer += "".join(parts)
            # If inside streaming content, better to use stream if available
            # but for simplicity we do full update here (tool events are rare)
            message.update(message.render_markdown())
            self.chat.scroll_end(animate=False)

    async def stream_llm(
        self,
        prompt: str,
        message: Message,
        tools: Optional[ToolProvider] = None,
    ) -> None:
        import time
        import asyncio

        logger.info("🚀 stream_llm START | prompt=%s", prompt[:200])

        assistant_msg = message
        text_buffer: str = ""
        last_update: float = time.time() - 0.1
        UPDATE_INTERVAL: float = 0.03

        logger.debug("Initial state | buffer_len=%d", len(text_buffer))

        # Prepare messages
        messages = [
            {"role": "system", "content": self.system_prompt or ""}
        ] + self.conversation[-15:]

        logger.debug(
            "Messages prepared | system_prompt_len=%d | history_count=%d",
            len(self.system_prompt or ""),
            len(self.conversation[-15:])
        )

        # Context enrichment
        if self.context_manager:
            logger.debug("Context manager detected, enriching context...")
            try:
                context_history = await self.context_manager.generate_prompt(
                    prompt, num_messages=0
                )

                logger.debug(
                    "Context history received | items=%d",
                    len(context_history) if context_history else 0
                )

                if context_history and context_history[-1].get("content"):
                    last_msg = context_history[-1]["content"]

                    logger.debug("Last context message length=%d", len(last_msg))

                    if "\nUser query:" in last_msg:
                        ctx_part = last_msg.split("\nUser query:")[0].strip()

                        logger.debug("Extracted ctx_part length=%d", len(ctx_part))

                        if ctx_part:
                            messages.insert(
                                1,
                                {
                                    "role": "system",
                                    "content": f"Relevant context:\n{ctx_part}",
                                },
                            )
                            logger.info("Context successfully injected into messages")

            except Exception as e:
                logger.error("❌ History enrichment failed: %s", e, exc_info=True)

        logger.debug("Final message count sent to runner=%d", len(messages))

        # Runner setup
        runner = AgentRunner(
            client=self.client,
            max_iterations=self.max_iterations,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        logger.info(
            "AgentRunner initialized | max_iterations=%s | temperature=%s",
            self.max_iterations,
            self.temperature,
        )

        # ── MAIN LOOP ─────────────────────────────────────────────
        try:
            logger.info("Starting AgentRunner loop...")

            async for event_type, payload in runner.run(
                user_prompt=prompt,
                messages_so_far=messages,
                tools=tools or [],
                system_prompt=self.system_prompt or "",
                context_manager=self.context_manager,
            ):
                logger.debug(
                    "Event received | type=%s | payload_preview=%s",
                    event_type,
                    str(payload)[:200],
                )

                if event_type == "content_delta":
                    delta = payload
                    text_buffer += delta

                    logger.debug(
                        "content_delta | delta_len=%d | buffer_len=%d",
                        len(delta),
                        len(text_buffer),
                    )

                    assistant_msg.buffer = text_buffer

                    current_time = time.time()
                    if current_time - last_update >= UPDATE_INTERVAL:
                        logger.debug("UI update triggered")
                        assistant_msg.update(assistant_msg.render_markdown())
                        last_update = current_time
                        self.chat.scroll_end(animate=False)

                elif event_type == "tool_start":
                    logger.info(
                        "Tool start | name=%s | args=%s",
                        payload.get("name"),
                        payload.get("args"),
                    )

                    if text_buffer:
                        assistant_msg.update(assistant_msg.render_markdown())
                        self.chat.scroll_end(animate=False)

                    self.render_tool_call(
                        assistant_msg,
                        name=payload["name"],
                        args=payload.get("args", {}),
                    )

                elif event_type == "tool_result":
                    logger.info(
                        "Tool result | name=%s | error=%s",
                        payload.get("name"),
                        payload.get("error"),
                    )

                    self.render_tool_call(
                        assistant_msg,
                        name=payload["name"],
                        args=payload.get("args", {}),
                        result=str(payload.get("result", "")),
                        error=payload.get("error", False),
                    )

                elif event_type == "needs_confirmation":
                    logger.warning(
                        "Confirmation required | tool=%s",
                        payload.get("name"),
                    )

                    if text_buffer:
                        assistant_msg.update(assistant_msg.render_markdown())
                        self.chat.scroll_end(animate=False)

                    preview = payload.get("preview", "")

                    self.render_tool_call(
                        assistant_msg,
                        name=payload["name"],
                        args=payload["args"],
                        confirmation=preview,
                    )

                    self.waiting_for_confirmation = True
                    self.pending_confirmation = {
                        "tool_call_id": payload["tool_call_id"],
                        "name": payload["name"],
                        "args": payload["args"],
                        "action": payload.get("action"),
                        "assistant_msg": assistant_msg,
                        "tool_calls": payload.get("tool_calls"),
                    }

                    logger.info("Paused for user confirmation")
                    return

                elif event_type == "cancelled":
                    logger.warning("Stream cancelled | reason=%s", payload)

                    text_buffer += (
                        f"\n\n🛑 **Stream cancelled** — {payload or 'user requested stop'}"
                    )

                    assistant_msg.buffer = text_buffer
                    assistant_msg.update(assistant_msg.render_markdown())
                    self.chat.scroll_end(animate=False)
                    return

                elif event_type == "final_answer":
                    logger.info("Final answer received | length=%d", len(str(payload)))

                    if text_buffer:
                        text_buffer += "\n\n"

                    text_buffer += str(payload).strip()
                    assistant_msg.buffer = text_buffer

                    assistant_msg.update(assistant_msg.render_markdown())
                    self.chat.scroll_end(animate=False)

                    self.conversation.append({"role": "assistant", "content": payload})

                    logger.info(
                        "Conversation updated | total_messages=%d",
                        len(self.conversation),
                    )

                elif event_type == "error":
                    logger.error("Agent error event: %s", payload)

                    text_buffer += f"\n\n❌ **Error:** {payload}"
                    assistant_msg.buffer = text_buffer

                    assistant_msg.update(assistant_msg.render_markdown())
                    self.chat.scroll_end(animate=False)
                    return

                elif event_type in ("log", "warning"):
                    logger.debug("Agent internal log: %s", payload)

                elif event_type == "finish":
                    logger.info("Finish event | reason=%s", payload.get("reason"))

                    if payload.get("reason") == "max_iterations_reached":
                        text_buffer += "\n\n⚠️ Max iterations reached."
                        assistant_msg.buffer = text_buffer

                        assistant_msg.update(assistant_msg.render_markdown())
                        self.chat.scroll_end(animate=False)

                # Safety net
                if event_type in ("tool_start", "tool_result") and not text_buffer.strip():
                    logger.debug("Injecting fallback 'Thinking…'")
                    text_buffer = "Thinking…\n\n"
                    assistant_msg.buffer = text_buffer

                    assistant_msg.update(assistant_msg.render_markdown())
                    self.chat.scroll_end(animate=False)

            logger.info("AgentRunner loop completed normally")

        except asyncio.CancelledError:
            logger.warning("Agent loop cancelled via asyncio")

            text_buffer += "\n\n🛑 **Agent loop cancelled**"
            assistant_msg.buffer = text_buffer

            assistant_msg.update(assistant_msg.render_markdown())
            self.chat.scroll_end(animate=False)

        except Exception as exc:
            logger.exception("🔥 Unexpected crash in stream_llm")

            error_msg = f"\n\n❌ **Unexpected error:** {exc}"
            text_buffer += error_msg

            assistant_msg.buffer = text_buffer
            assistant_msg.update(assistant_msg.render_markdown())
            self.chat.scroll_end(animate=False)

        # Final flush
        logger.debug("Final UI flush | buffer_len=%d", len(text_buffer))

        assistant_msg.update(assistant_msg.render_markdown())
        self.chat.scroll_end(animate=False)

        logger.info("✅ stream_llm END")

    async def action_clear_chat(self):
        self.chat.remove_children()
