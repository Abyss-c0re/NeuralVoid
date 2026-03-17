import sys
import asyncio
from pathlib import Path

from neuralcore.actions.registry import ActionRegistry
from neuralcore.actions.manager import DynamicActionManager

from neuralcore.utils.tool_browser import ToolBrowser
from neuralcore.cognition.memory import ContextManager

from neuralvoid.tools.terminal_set import get_terminal_actions
from neuralvoid.tools.file_set import get_file_actions

from neuralvoid.ui.chat import LLMChatApp
from neuralvoid.ui.rendering import get_renderer

from neuralvoid.cli.arg_parser import CLIParser
from neuralcore.utils.logger import Logger

from neuralcore.utils.config import ConfigLoader
from neuralcore.core.client_factory import ClientFactory

logger = Logger.get_logger(renderer=get_renderer())


def main():
    # ───────────────────────────── CLI ─────────────────────────────
    args = CLIParser().parse()

    # ───────────────────────────── CONFIG ──────────────────────────
    loader = ConfigLoader(cli_path=args.config)
    system_prompt = loader.get_system_prompt()

    # ───────────────────────────── CLIENTS ─────────────────────────
    factory = ClientFactory(loader)
    clients = factory.build()

    client = clients.get("main")
    embeddings = clients.get("embeddings")

    # Ensure tokenizer
    if not client.tokenizer:
        from neuralcore.utils.text_tokenizer import TextTokenizer

        main_cfg = loader.get_client_config("main")
        tokenizer_name = main_cfg.get("tokenizer", "Qwen/Qwen3.5-9B")
        tokenizer = TextTokenizer(tokenizer_name)
    else:
        tokenizer = client.tokenizer

    # ───────────────────────────── REGISTRY & TOOLS ─────────────────
    registry = ActionRegistry()

    # register standard tool sets
    enabled_sets = loader.config.get("tools", {}).get("enabled_sets", [])
    if "terminal" in enabled_sets:
        registry.register_set("terminal set", get_terminal_actions())
    if "file" in enabled_sets:
        registry.register_set("file set", get_file_actions())

    # register LLM clients marked as tools in config
    factory.register_tool_clients(registry, client)

    dynamic_manager = DynamicActionManager(registry)
    ToolBrowser(registry, dynamic_manager)

    context_manager = ContextManager(client=embeddings, tokenizer=tokenizer)
    # Headless Agent Mode
    if args.deploy:
        from neuralvoid.cli.headless_agent import HeadlessAgentRunner

        runner = HeadlessAgentRunner(
            status_file=args.status_file,
            pid_file=args.pid_file,
            status_update_throttle_sec=args.throttle_sec,
        )

        prompt = args.deploy.strip()
        agent_cfg = loader.get_agent_config("headless")  # <-- NEW

        max_iterations = args.max_iterations or agent_cfg.get("max_iterations", 10)
        max_tokens = agent_cfg.get("max_tokens", 12000)

        print("    Deploying headless agent")
        print(f"   Prompt       : {prompt}")
        print(f"   Status file  : {Path(args.status_file).resolve()}")
        print(f"   PID file     : {Path(args.pid_file).resolve()}")
        print(f"   Max iterations: {max_iterations}")
        print("-" * 60)

        success = asyncio.run(
            runner.run(
                client=client,
                prompt=prompt,
                dynamic_manager=dynamic_manager,
                system_prompt=system_prompt,
                context_manager=context_manager,
                max_iterations=max_iterations,
                max_tokens=max_tokens,
            )
        )

        sys.exit(0 if success else 1)

    # ───────────────────────────── INTERACTIVE CHAT ────────────────

    app_cfg = loader.get_app_config()
    max_iterations = getattr(args, "max_iterations", None) or app_cfg.get("max_iterations", 10)
    max_tokens = app_cfg.get("max_tokens", 12000)
    app = LLMChatApp(
        client=client,
        system_prompt=system_prompt,
        tools=dynamic_manager,
        context_manager=context_manager,
        tool_rendering="info",
        max_iterations=max_iterations,
        max_tokens = max_tokens
    )
    app.run()


if __name__ == "__main__":
    main()
