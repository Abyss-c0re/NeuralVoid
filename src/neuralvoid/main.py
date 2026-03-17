import os
import sys

import asyncio

from pathlib import Path
from neuralcore.core.client import LLMClient
from neuralcore.actions.registry import ActionRegistry

from neuralcore.actions.manager import DynamicActionManager
from neuralcore.utils.tool_browser import ToolBrowser

from neuralcore.utils.llm_tools import InternalTools
from neuralcore.cognition.memory import ContextManager

from neuralvoid.tools.terminal_set import get_terminal_actions
from neuralvoid.tools.file_set import get_file_actions

from neuralvoid.ui.chat import LLMChatApp
from neuralvoid.ui.rendering import get_renderer

from neuralvoid.cli.arg_parser import CLIParser
from neuralcore.utils.logger import Logger


logger = Logger.get_logger(renderer=get_renderer())

def main():
    args = CLIParser().parse()

    # ─────────────────────────────────────────────────────────────
    # LLM & components setup (common to both modes)
    # ─────────────────────────────────────────────────────────────
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:1212/v1")
    model = os.getenv("LLM_MODEL", "qwen3.5-9b")
    system_prompt = "You are terminal chat assistant, use provided tools if needed"

    client = LLMClient(
        base_url=base_url,
        model=model,
        tokenizer="Qwen/Qwen3.5-9B",
        api_key= "not-needed",
        extra_body={
            "presence_penalty": 1.5,
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    if not client.tokenizer:
        from neuralcore.utils.text_tokenizer import TextTokenizer
        tokenizer = TextTokenizer("Qwen/Qwen3.5-9B")
    else:
        tokenizer = client.tokenizer

    reasoner = LLMClient(
        base_url=base_url,
        model=model,
        tokenizer=tokenizer,
        extra_body={
            "presence_penalty": 1.2,
            "top_k": 30,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )

    embeddings = LLMClient(
        base_url=base_url,
        model="embedding-gemma-300m",
        tokenizer=tokenizer,
    )

    # Registry & tools setup
    registry = ActionRegistry()
    registry.register_set("terminal set", get_terminal_actions())
    registry.register_set("file set", get_file_actions())

    reasoning_tools = InternalTools(
        client=reasoner,
        description=(
            "Powerful reasoning model optimized for step-by-step thinking, "
            "complex math, planning, code writing, and long chains of thought."
        ),
        methods=[client.ask, client.stream_chat, client.chat],
    )
    registry.register_set("HeavyReasoning", reasoning_tools.as_action_set("HeavyReasoning"))

    dynamic_manager = DynamicActionManager(registry)
    ToolBrowser(registry, dynamic_manager)

    context_manager = ContextManager(client=embeddings, tokenizer=tokenizer)

    # ─────────────────────────────────────────────────────────────
    # DEPLOY AGENT MODE
    # ─────────────────────────────────────────────────────────────
    if args.deploy:
        from neuralvoid.cli.headless_agent import HeadlessAgentRunner

        runner = HeadlessAgentRunner(
            status_file=args.status_file,
            pid_file=args.pid_file,
            status_update_throttle_sec=args.throttle_sec,
        )

        prompt = args.deploy.strip()
        print(f"🚀 Deploying headless agent")
        print(f"   Prompt       : {prompt}")
        print(f"   Status file  : {Path(args.status_file).resolve()}")
        print(f"   PID file     : {Path(args.pid_file).resolve()}")
        print(f"   Throttle     : {args.throttle_sec} s")
        print("-" * 60)

        success = asyncio.run(
            runner.run(
                client=client,
                prompt=prompt,
                dynamic_manager=dynamic_manager,
                system_prompt=system_prompt,
                context_manager=context_manager,
                max_iterations=args.max_iterations,
                max_tokens=12000,
            )
        )

        sys.exit(0 if success else 1)

    # ─────────────────────────────────────────────────────────────
    # NORMAL INTERACTIVE CHAT MODE
    # ─────────────────────────────────────────────────────────────
    app = LLMChatApp(
        client=client,
        system_prompt=system_prompt,
        tools=dynamic_manager,
        context_manager=context_manager,
        tool_rendering="info",
    )
    app.run()


if __name__ == "__main__":
    main()