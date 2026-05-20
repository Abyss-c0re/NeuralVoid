"""
NeuralVoid UI

Textual-based terminal chat interface and rendering helpers.
"""

from .chat import LLMChatApp
from .rendering import get_renderer, set_renderer_app

__all__ = ["LLMChatApp", "get_renderer", "set_renderer_app"]
