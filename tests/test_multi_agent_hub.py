"""
Multi-agent Hub integration test.

Deploys two agents using either MockLLMServer or the **real** LLM server
from config.yaml (e.g. localhost:1212 + the extra hub_alpha/hub_beta agents).

The core goal: prove that when a message arrives via AgentHub, the *receiving
agent's own LLM* (real server) actually runs the normal classify_intent +
casual chat path and emits a real generated reply, and that this works
bidirectionally (Agent 1's LLM responds to Agent 2 and vice versa).

Run:
    uv run python tests/test_multi_agent_hub.py
    # or: python -m pytest ... (if pytest-ified later)

The test:
  1. Loads config.yaml (or a *_test_config.yaml if present).
  2. Creates two agents via AgentFactory (hub_alpha / hub_beta preferred).
  3. Registers them with AgentHub (WebSocket bridges + central router).
  4. Starts the hub + both agents in listening mode (chat_tool_loop).
  5. Sends a message Alpha → Beta via the hub.
  6. Waits for Beta's *real LLM* (on the configured server) to classify intent + generate a reply.
  7. Sends a reply back Beta → Alpha via the hub.
  8. Waits for Alpha's *real LLM* to also generate a response.
  9. Asserts that we saw actual LLM-generated text in BOTH directions.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# ── Framework imports ──
import neuralcore.utils.config as config_module
import neuralcore.clients.factory as cfactory_module
from neuralcore.utils.config import ConfigLoader
from neuralcore.agents.factory import AgentFactory
from neuralcore.clients.factory import ClientFactory

# ── Client imports ──
from neuralvoid.workflows.default_flow import AgentFlow
from neuralvoid.hub import AgentHub

# ── Make sure imports work from project root ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# Note: NeuralCore (dependency) is resolved via uv / site-packages.
# No "client/src" insert needed for NeuralVoid layout.

os.chdir(PROJECT_ROOT)


async def run_test():
    print("=" * 70)
    print("  MULTI-AGENT HUB TEST — Alpha ↔ Beta via WebSocket Hub")
    print("=" * 70)

    # ──────────────────────────────────────────────────────────────
    # 1. Load config (real server from config.yaml or TEST mock if used)
    #    Reset global singletons so the test config takes effect.
    # ──────────────────────────────────────────────────────────────
    config_module.loader = None
    cfactory_module._factory = None

    # Prefer local config.yaml; fall back to data/ test config if present
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        config_path = PROJECT_ROOT / "data" / "multi_agent_test_config.yaml"
    loader = ConfigLoader(cli_path=str(config_path), app_root=PROJECT_ROOT)
    # Install as global singleton so get_clients() / get_loader() use it
    config_module.loader = loader

    # If any client uses base_url=TEST the loader starts an internal MockLLMServer
    mock_server = loader.get_test_server()
    if mock_server is not None:
        print(f"[OK] Using MockLLMServer at {mock_server.base_url}")
    else:
        # Real server - show first client's url
        try:
            first_client = list(loader.get_clients().values())[0] if loader.get_clients() else None
            url = getattr(first_client, "base_url", "unknown")
            print(f"[OK] Using REAL LLM server at {url}")
        except Exception:
            print("[OK] Using real clients from config (no mock)")

    # ──────────────────────────────────────────────────────────────
    # 2. Build clients + agents (set factory as global singleton too)
    # ──────────────────────────────────────────────────────────────
    client_factory = ClientFactory(loader)
    client_factory.build()
    cfactory_module._factory = client_factory

    agent_factory = AgentFactory(loader)

    # Use the dedicated hub test agents (added to config.yaml) with fallbacks
    # to the built-in ones so the test works out-of-the-box with real config.
    alpha_id = "hub_alpha"
    beta_id = "hub_beta"
    alpha_cfg = loader.get_agent_config(alpha_id) or loader.get_agent_config("agent_002")
    beta_cfg = loader.get_agent_config(beta_id) or loader.get_agent_config("agent_001")
    if not alpha_cfg:
        alpha_id = "agent_002"
        alpha_cfg = loader.get_agent_config(alpha_id)
    if not beta_cfg:
        beta_id = "agent_002"
        beta_cfg = loader.get_agent_config(beta_id)

    alpha = agent_factory.create_agent(
        agent_id=alpha_id, config=alpha_cfg, app_root=PROJECT_ROOT
    )
    beta = agent_factory.create_agent(
        agent_id=beta_id, config=beta_cfg, app_root=PROJECT_ROOT
    )

    # Register default workflows (chat_tool_loop, goal_driven_loop, etc.)
    AgentFlow(alpha)
    AgentFlow(beta)

    print(f"[OK] Agent Alpha: {alpha.name} ({alpha.agent_id})")
    print(f"[OK] Agent Beta:  {beta.name} ({beta.agent_id})")

    # ──────────────────────────────────────────────────────────────
    # 3. Wire up the Hub
    # ──────────────────────────────────────────────────────────────
    hub = AgentHub(host="127.0.0.1", hub_port=8770, bridge_base_port=8771)
    alpha_port = hub.register_agent(alpha)
    beta_port = hub.register_agent(beta)
    print(
        f"[OK] Hub registered Alpha (bridge :{alpha_port}), Beta (bridge :{beta_port})"
    )

    # ──────────────────────────────────────────────────────────────
    # 4. Start the Hub (bridges + hub WS server)
    # ──────────────────────────────────────────────────────────────
    await hub.start()
    # Small delay so bridges bind
    await asyncio.sleep(0.5)
    print(f"[OK] Hub started on ws://127.0.0.1:8770")

    # ──────────────────────────────────────────────────────────────
    # 5. Start BOTH agents in real listening mode (chat_tool_loop).
    #    For real LLM server this means:
    #      - Each agent will call the real server (localhost:1212) for
    #        intent classification + casual reply when a hub message arrives.
    #      - We collect "llm_response" events from both directions.
    # ──────────────────────────────────────────────────────────────
    use_mock = mock_server is not None
    alpha_stop = asyncio.Event()
    beta_stop = asyncio.Event()
    alpha_events: list = []
    beta_events: list = []
    alpha_task = None
    beta_task = None

    async def _run_agent(agent_obj, stop_ev, event_list, label):
        """Run one agent in listening mode and collect interesting events."""
        try:
            async for event_type, payload in agent_obj.run(
                user_prompt=None,
                system_prompt=f"You are {label} in the hub round-trip test.",
                max_tokens=1024,
                chat_mode=False,
                stop_event=stop_ev,
            ):
                event_list.append((event_type, payload))
                if event_type == "llm_response":
                    reply = (payload.get("full_reply", "") if isinstance(payload, dict) else str(payload))
                    print(f"\n  [{label.upper()} LLM] {reply[:220]}")
                elif event_type in ("phase_changed", "error"):
                    print(f"  [{label.upper()}] {event_type}: {payload}")
                elif event_type not in ("step_start_detail", "step_start", "step_completed",
                                        "loop_steps_completed", "wait_progress", "thinking"):
                    print(f"  [{label.upper()}] {event_type}: {str(payload)[:110]}")
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            print(f"  [{label.upper()}] runner error: {exc}")

    # Launch both agents (they will block in wait_for_incoming_message until hub messages arrive)
    alpha_task = asyncio.create_task(_run_agent(alpha, alpha_stop, alpha_events, "Alpha"), name="alpha_runner")
    beta_task = asyncio.create_task(_run_agent(beta, beta_stop, beta_events, "Beta"), name="beta_runner")

    # Give both agents time to enter their chat loops and hit the message wait point
    await asyncio.sleep(2.8 if not use_mock else 1.2)
    print("[OK] Both agents are listening for hub-relayed messages (real LLM path active)" if not use_mock
          else "[OK] Both agents listening (mock path)")

    # ──────────────────────────────────────────────────────────────
    # 6. Bidirectional real-LLM round-trip via the Hub
    #    Alpha's LLM must reply to a message that came from Beta (and vice versa).
    # ──────────────────────────────────────────────────────────────
    test_message = "Hello Beta, this is just a friendly hello from Alpha via the hub. No action needed, please reply briefly so I know your LLM is working."
    print(f"\n>>> [STEP 1] Alpha → Beta (real LLM on server must classify + generate reply)")
    print(f'    "{test_message}"')

    ok1 = await hub.send_to_agent(
        target_id=beta_id,
        content=test_message,
        source_id=alpha_id,
    )
    print(f"    Hub relay 1 success: {ok1}")

    # ──────────────────────────────────────────────────────────────
    # 7. Wait for Beta's real LLM to produce a reply (CASUAL path)
    # ──────────────────────────────────────────────────────────────
    print("\n    Waiting for Beta's LLM to respond (this calls the real server)...")
    deadline = time.time() + (180.0 if not use_mock else 20.0)
    beta_replies = []
    while time.time() < deadline:
        for evt_type, evt_payload in beta_events:
            if evt_type == "llm_response":
                reply = (evt_payload.get("full_reply", "") if isinstance(evt_payload, dict) else str(evt_payload))
                if reply and reply not in beta_replies:
                    beta_replies.append(reply)
        if beta_replies:
            break
        await asyncio.sleep(0.35 if not use_mock else 0.2)

    # ──────────────────────────────────────────────────────────────
    # 8. Now the other direction: Beta's LLM replied → send a reply back
    #    so that Alpha's LLM also has to respond via the real server.
    # ──────────────────────────────────────────────────────────────
    beta_reply_text = beta_replies[0] if beta_replies else "I got your message."
    reply_back = f"Hi Alpha, friendly hello back from Beta. Just acknowledging your relayed message (no tasks). My LLM reply was: {beta_reply_text[:100]}"

    print(f"\n>>> [STEP 2] Beta → Alpha (real LLM on server must respond to the reply)")
    print(f'    "{reply_back[:160]}..."')

    ok2 = await hub.send_to_agent(
        target_id=alpha_id,
        content=reply_back,
        source_id=beta_id,
    )
    print(f"    Hub relay 2 success: {ok2}")

    # Wait for Alpha's real LLM reply
    print("\n    Waiting for Alpha's LLM to respond (second real generation)...")
    deadline2 = time.time() + (120.0 if not use_mock else 15.0)
    alpha_replies = []
    while time.time() < deadline2:
        for evt_type, evt_payload in alpha_events:
            if evt_type == "llm_response":
                reply = (evt_payload.get("full_reply", "") if isinstance(evt_payload, dict) else str(evt_payload))
                if reply and reply not in alpha_replies:
                    alpha_replies.append(reply)
        if alpha_replies:
            break
        await asyncio.sleep(0.35 if not use_mock else 0.2)

    got_response = bool(beta_replies) and bool(alpha_replies)

    # ──────────────────────────────────────────────────────────────
    # 9. Verify results — we want real LLM generations in BOTH directions
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS — Real LLM Server Round-Trip via AgentHub")
    print("=" * 70)

    # Hub log (two messages for bidirectional)
    print(f"\n  Hub message log entries: {len(hub.message_log)}")
    for entry in hub.message_log:
        print(f"    {entry.get('source')} → {entry.get('target')}: {entry.get('content','')[:90]}")

    # Real LLM replies we captured from the event streams
    print(f"\n  Beta LLM replies captured: {len(beta_replies)}")
    for r in beta_replies:
        print(f"    Beta said: {r[:220]}")

    print(f"\n  Alpha LLM replies captured: {len(alpha_replies)}")
    for r in alpha_replies:
        print(f"    Alpha said: {r[:220]}")

    # Show final context on both sides (proves the relayed messages + replies were recorded)
    try:
        beta_ctx = beta.context_manager.get_context_summary(max_messages=8, max_chars=1200)
        print(f"\n  Beta context (last messages):\n    {beta_ctx[:350]}...")
    except Exception:
        pass
    try:
        alpha_ctx = alpha.context_manager.get_context_summary(max_messages=8, max_chars=1200)
        print(f"\n  Alpha context (last messages):\n    {alpha_ctx[:350]}...")
    except Exception:
        pass

    # Final verdict — the key requirement is that the *real LLM server*
    # was used by each agent to generate a reply to the message that
    # arrived from the other agent via the hub.
    both_directions = bool(beta_replies) and bool(alpha_replies)
    success = ok1 and ok2 and len(hub.message_log) >= 2 and both_directions

    print("\n" + "=" * 70)
    if success:
        print("  ✅ TEST PASSED — Agent 1's LLM (real server) replied to Agent 2 and vice versa!")
        print("     Both directions used the LLM via the normal chat_tool_loop + hub relay.")
    else:
        print("  ❌ TEST FAILED — full LLM round-trip did not complete")
        if not (ok1 and ok2):
            print("    - One or both hub.send_to_agent calls failed")
        if len(hub.message_log) < 2:
            print("    - Expected at least 2 relayed messages in the log")
        if not beta_replies:
            print("    - Beta never produced an llm_response (real LLM did not reply to Alpha)")
        if not alpha_replies:
            print("    - Alpha never produced an llm_response (real LLM did not reply to Beta)")
    print("=" * 70)

    # ──────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────
    alpha_stop.set()
    beta_stop.set()
    for t, name in [(alpha_task, "alpha"), (beta_task, "beta")]:
        if t is not None and not t.done():
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
    await hub.stop()

    return success


def main():
    success = asyncio.run(run_test())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
