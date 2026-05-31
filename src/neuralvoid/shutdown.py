"""
Central shutdown coordination for NeuralVoid.

This module breaks circular imports and provides a single place
for all shutdown-related state and the "purge everything" logic.

The contract the user wants:
- As soon as the TUI app closes (via "exit" command or Ctrl+C),
  we aggressively purge the main agent + every background agent
  + their BackgroundManagers.
"""

import asyncio
import signal
from typing import Any

_shutdown_event = asyncio.Event()
_current_tui_app: Any = None
_active_agents: list = []


def _shutdown_all_agents():
    """Best-effort shutdown of any agents that were created in this process."""
    for agent in list(_active_agents):
        try:
            if hasattr(agent, "shutdown"):
                asyncio.run(agent.shutdown())
        except Exception:
            pass
    _active_agents.clear()
    print("[NeuralVoid] All background agents purged (sync path). Forcing process exit...")
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.raise_signal(signal.SIGINT)


async def purge_everything():
    """
    Aggressively shut down the main TUI agent + all background agents
    and their BackgroundManagers.

    This should be called as soon as the TUI decides to close.
    """
    print("[NeuralVoid] Purging all background work...")

    # Shutdown the TUI's main agent (if the TUI set the reference)
    try:
        if _current_tui_app and hasattr(_current_tui_app, "agent") and _current_tui_app.agent:
            await asyncio.wait_for(_current_tui_app.agent.shutdown(), timeout=6.0)
    except asyncio.TimeoutError:
        print("[NeuralVoid] Main agent shutdown timed out.")
    except Exception as e:
        print(f"[NeuralVoid] Error shutting down main agent: {e}")

    # Shutdown every background agent we ever tracked
    for agent in list(_active_agents):
        try:
            await asyncio.wait_for(agent.shutdown(), timeout=6.0)
        except asyncio.TimeoutError:
            print(f"[NeuralVoid] Background agent shutdown timed out.")
        except Exception as e:
            print(f"[NeuralVoid] Error shutting down background agent: {e}")

    _active_agents.clear()
    print("[NeuralVoid] All background agents purged. Forcing process exit...")
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.raise_signal(signal.SIGINT)


_sigint_count = [0]


def _setup_top_level_signal_handlers():
    """Install process-wide signal handlers that cooperate with asyncio/Textual."""
    def _handler(sig, frame):
        _sigint_count[0] += 1

        print("\n[NeuralVoid] Ctrl+C received — initiating clean shutdown of background agents...")
        _shutdown_event.set()

        global _current_tui_app
        if _current_tui_app is not None:
            try:
                _current_tui_app.call_from_thread(_current_tui_app.exit)
            except Exception:
                pass

        if _sigint_count[0] >= 2:
            print("[NeuralVoid] Second Ctrl+C — force killing process")
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.raise_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

