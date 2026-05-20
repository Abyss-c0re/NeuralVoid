"""
NeuralVoid - Terminal-first AI Agent Client

Built on top of NeuralCore. Provides interactive chat UI, headless deployment,
multi-agent hub coordination, and rich tool integrations.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("neuralvoid")
except PackageNotFoundError:
    __version__ = "0.1.0"

# High-level public API
from neuralhub import AgentHub
from .ui.chat import LLMChatApp
from .cli.arg_parser import CLIParser

__all__ = [
    "__version__",
    "AgentHub",
    "LLMChatApp",
    "CLIParser",
]
