import os
import sys
import signal
import atexit
import asyncio
from pathlib import Path

from neuralvoid.cli.arg_parser import CLIParser
from neuralvoid.ui.chat import LLMChatApp

from neuralvoid.utils import find_app_root

from neuralcore import ConfigLoader, get_clients, AgentFactory
from neuralvoid.workflows.default_flow import AgentFlow


_active_agents: list = []


def _shutdown_all_agents():
    """Best-effort shutdown of any agents that were created in this process."""
    for agent in _active_agents:
        try:
            if hasattr(agent, "shutdown"):
                asyncio.run(agent.shutdown())
        except Exception:
            pass
    _active_agents.clear()


def _setup_top_level_signal_handlers():
    """Install process-wide signal handlers so Ctrl+C always triggers full background cleanup."""
    def _handler(sig, frame):
        print("\n[NeuralVoid] Received shutdown signal, cleaning up background work...")
        _shutdown_all_agents()
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handler)

    # Extra safety net for normal process exit
    atexit.register(_shutdown_all_agents)


def main():
    _setup_top_level_signal_handlers()

    # ───────────────────────────── CLI ─────────────────────────────
    args = CLIParser().parse()

    if args.config:
        os.environ["NEURALCORE_CONFIG"] = args.config
    else:
        os.environ["NEURALCORE_CONFIG"] = str(
            Path.home() / ".neuralcore" / "config.yaml"
        )

    app_root = find_app_root()
    loader = ConfigLoader(cli_path=args.config, app_root=app_root)

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

    # ───────────────────────────── AGENT ───────────────────────────
    if args.agent:
        # User-provided --agent always takes highest priority
        agent_id = args.agent
    elif args.deploy is not None:
        # Headless mode default
        agent_id = "agent_001"
    else:
        # UI / Interactive mode default
        agent_id = "agent_002"

    # Updated agent loading (load_agent_from_config removed)
    # Uses AgentFactory pattern from the integration test reference.
    # No domain-specific tool loading added here – kept modular and client-side.
    agent_config = loader.get_agent_config(agent_id)
    factory = AgentFactory(loader)
    agent = factory.create_agent(
        agent_id=agent_id,
        config=agent_config,
        app_root=app_root,
    )
    _active_agents.append(agent)
    AgentFlow(agent)  # loading default workflows.

    # ── Headless mode ─────────────────────────────────────────────
    if args.deploy is not None:
        # Optional prompt support (None = task-ready / autonomous mode)
        prompt = args.deploy.strip() if args.deploy else None

        # ────────────────────────────────────────────────────────
        # [NEW] Multi-agent deployment via AgentHub
        # When --agents "id1,id2" is supplied, we spin up the Hub,
        # register every requested agent, and deploy them in parallel
        # with full WebSocket inter-agent communication.
        # ────────────────────────────────────────────────────────
        if getattr(args, "agents", None):
            from neuralhub import AgentHub

            agent_ids = [aid.strip() for aid in args.agents.split(",") if aid.strip()]
            hub_port = getattr(args, "hub_port", 8770)

            hub = AgentHub(host="127.0.0.1", hub_port=hub_port)

            for aid in agent_ids:
                a_cfg = loader.get_agent_config(aid)
                if not a_cfg:
                    print(f"[ERROR] Agent '{aid}' not found in config — skipping")
                    continue
                a = factory.create_agent(
                    agent_id=aid,
                    config=a_cfg,
                    app_root=app_root,
                )
                _active_agents.append(a)
                AgentFlow(a)  # register default workflows
                hub.register_agent(a)

            if not hub.agents:
                print("[ERROR] No valid agents to deploy")
                sys.exit(1)

            max_tokens = args.max_tokens or 12000

            print(f"\n   Multi-agent deploy: {list(hub.agents.keys())}")
            print(f"   Hub port     : {hub_port}")
            print(f"   Prompt       : {prompt or '<none - listening mode>'}")
            print("-" * 60)

            try:
                results = asyncio.run(
                    hub.deploy_all(
                        prompt=prompt,
                        system_prompt=loader.get_system_prompt(),
                        max_tokens=max_tokens,
                    )
                )
            finally:
                # Best-effort full shutdown of all agents' background managers
                for a in list(hub.agents.values()):
                    try:
                        asyncio.run(a.shutdown())
                    except Exception:
                        pass
            # Exit 0 only if every agent succeeded
            sys.exit(0 if all(results.values()) else 1)

        # ────────────────────────────────────────────────────────
        # Single-agent deployment (original path, unchanged)
        # ────────────────────────────────────────────────────────
        from neuralvoid.cli.headless_agent import HeadlessAgentRunner

        runner = HeadlessAgentRunner(
            agent=agent,
            status_file=args.status_file,
            pid_file=args.pid_file,
            status_update_throttle_sec=args.throttle_sec,
        )

        # Reuse pre-fetched agent_config (avoids redundant loader call)
        agent_cfg = agent_config

        max_iterations = args.max_iterations or agent_cfg.get("max_iterations", 10)
        max_tokens = args.max_tokens or agent_cfg.get("max_tokens", 12000)

        print(f"   Deploying headless agent '{agent.name}'")
        print(f"   Prompt       : {prompt or '<none - task-ready / autonomous mode>'}")
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
