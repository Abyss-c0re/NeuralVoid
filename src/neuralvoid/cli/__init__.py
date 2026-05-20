"""
NeuralVoid CLI

Command-line argument parsing and headless agent runner.

The rich HeadlessAgentRunner is a subclass of the canonical implementation
provided by NeuralHub (neuralhub.runners.headless_runner).
"""

from .arg_parser import CLIParser
from .headless_agent import HeadlessAgentRunner

__all__ = ["CLIParser", "HeadlessAgentRunner"]
