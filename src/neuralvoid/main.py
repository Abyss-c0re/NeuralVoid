import os
import sys
import asyncio
from pathlib import Path

from neuralvoid.cli.arg_parser import CLIParser
from neuralvoid.ui.chat import LLMChatApp

from neuralvoid.ui.rendering import get_renderer

from neuralcore.utils.config import get_loader

from neuralcore.core.client_factory import get_clients


from neuralcore.utils.logger import Logger

logger = Logger.get_logger(renderer=get_renderer())


def main():
    # ───────────────────────────── CLI ─────────────────────────────
    args = CLIParser().parse()

    if args.config:
        os.environ["NEURALCORE_CONFIG"] = args.config
    else:
        os.environ["NEURALCORE_CONFIG"] = str(
            Path.home() / ".neuralcore" / "config.yaml"
        )

    # ───────────────────────────── CONFIG ──────────────────────────
    loader = get_loader(cli_path=args.config, app_root=Path(__file__).parent)

    system_prompt = loader.get_system_prompt()

    # ───────────────────────────── CLIENTS ─────────────────────────
    clients = get_clients()
    client = clients.get("main")

    if not client:
        print("""Export NEURALCORE_CONFIG to config.yaml or specify --config via command line.
Default paths:
- $HOME/.neuralcore/config.yaml
- Inside the app folder""")
        sys.exit(1)

    agent_id = args.agent or "agent_002"  # default to casual chat agent
    agent = loader.load_agent_from_config(agent_id)

    # ── Headless mode ─────────────────────────────────────────────
    if args.deploy:
        from neuralvoid.cli.headless_agent import HeadlessAgentRunner

        runner = HeadlessAgentRunner(
            agent=agent,
            status_file=args.status_file,
            pid_file=args.pid_file,
            status_update_throttle_sec=args.throttle_sec,
        )

        prompt = args.deploy.strip()
        agent_cfg = loader.get_agent_config(agent_id)

        max_iterations = args.max_iterations or agent_cfg.get("max_iterations", 10)
        max_tokens = args.max_tokens or agent_cfg.get("max_tokens", 12000)

        print(f"   Deploying headless agent '{agent.name}'")
        print(f"   Prompt       : {prompt}")
        print(f"   Status file  : {Path(args.status_file).resolve()}")
        print(f"   PID file     : {Path(args.pid_file).resolve()}")
        print(f"   Max iterations: {max_iterations}")
        print("-" * 60)

        success = asyncio.run(
            runner.run(
                prompt=prompt,
                system_prompt=loader.get_system_prompt(),
                max_tokens=max_tokens,
            )
        )
        sys.exit(0 if success else 1)

    # ── Interactive UI mode ───────────────────────────────────────

    app_cfg = loader.get_app_config()
    max_iterations = getattr(args, "max_iterations", None) or app_cfg.get(
        "max_iterations", 10
    )
    tool_info_level = app_cfg.get("tool_info_level", "compact")
    max_tokens = app_cfg.get("max_tokens", 12000)

    app = LLMChatApp(
        agent=agent,
        system_prompt=system_prompt,
        tool_rendering="info",
        max_iterations=max_iterations,
        max_tokens=max_tokens,
        tool_info_level=tool_info_level,
    )
    app.run()


if __name__ == "__main__":
    main()
