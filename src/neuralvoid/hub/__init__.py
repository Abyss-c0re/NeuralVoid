"""
AgentHub package for multi-agent WebSocket coordination.

Re-exports the main AgentHub class so that
    from neuralvoid.hub import AgentHub
continues to work after the hub module was turned into a package.
"""

from .hub import AgentHub

__all__ = ["AgentHub"]
