import os
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

from neuralcore.utils.config import get_loader
from neuralcore.core.client_factory import get_clients, get_client_factory

logger = Logger.get_logger(renderer=get_renderer())


def main():
    # ───────────────────────────── CLI ─────────────────────────────
    args = CLIParser().parse()

    if args.config:  # Check if a specific config path is passed via CLI
        os.environ["NEURALCORE_CONFIG"] = args.config
    else:
        # If no config path is passed, set a default config path (optional)
        os.environ["NEURALCORE_CONFIG"] = str(
            Path.home() / ".neuralcore" / "config.yaml"
        )

    # ───────────────────────────── CONFIG ──────────────────────────
    loader = get_loader(cli_path=args.config)
    system_prompt = loader.get_system_prompt()

    # ───────────────────────────── CLIENTS ─────────────────────────

    clients = get_clients()
    client = clients.get("main")

    if client:
        # ───────────────────────────── REGISTRY & TOOLS ─────────────────
        registry = ActionRegistry()

        # register standard tool sets
        enabled_sets = loader.config.get("tools", {}).get("enabled_sets", [])
        if "terminal" in enabled_sets:
            registry.register_set("terminal set", get_terminal_actions())
        if "file" in enabled_sets:
            registry.register_set("file set", get_file_actions())

        # register LLM clients marked as tools in config
        factory = get_client_factory()
        factory.register_tool_clients(registry, client)

        dynamic_manager = DynamicActionManager(registry)
        ToolBrowser(registry, dynamic_manager)

        context_manager = ContextManager()

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
        max_iterations = getattr(args, "max_iterations", None) or app_cfg.get(
            "max_iterations", 10
        )
        tool_info_level = app_cfg.get("tool_info_level", "compact")
        max_tokens = app_cfg.get("max_tokens", 12000)
        app = LLMChatApp(
            client=client,
            system_prompt=system_prompt,
            tools=dynamic_manager,
            context_manager=context_manager,
            tool_rendering="info",
            max_iterations=max_iterations,
            max_tokens=max_tokens,
            tool_info_level = tool_info_level
        )
        app.run()
    else:
        print("""Export NEURALCORE_CONFIG to config.yaml or specify --config via command line.
            Default paths:
            - $HOME/.neuralcore/config.yaml
            - Inside the app folder""")


if __name__ == "__main__":
    main()
