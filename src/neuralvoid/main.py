import os

from neuralcore.core.client import LLMClient
from neuralcore.actions.registry import ActionRegistry

from neuralcore.actions.manager import DynamicActionManager
from neuralcore.utils.tool_browser import ToolBrowser

from neuralcore.utils.llm_tools import InternalTools
from neuralcore.cognition.memory import ContextManager

from neuralvoid.tools.terminal_set import get_terminal_actions
from neuralvoid.tools.file_set import get_file_actions

from neuralvoid.ui.chat import LLMChatApp
from neuralvoid.ui.rendering import get_renderer
from neuralcore.utils.logger import Logger


logger = Logger.get_logger(renderer=get_renderer())

def main():
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:1212/v1")
    model = os.getenv("LLM_MODEL", "qwen3.5-9b")
    system_prompt = "You are terminal chat assistant, use provided tools if needed"

    client = LLMClient(
        base_url=base_url,
        model=model,
        tokenizer= "Qwen/Qwen3.5-9B",
        extra_body={
            "presence_penalty": 1.5,
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    if not client.tokenizer:
        from neuralcore.utils.text_tokenizer import TextTokenizer
        tokenizer = TextTokenizer("Qwen/Qwen3.5-9B")
    else:
        tokenizer = client.tokenizer

  

    reasoner = LLMClient(
        base_url=base_url,
        model=model,
        tokenizer = tokenizer,
        extra_body={
            "presence_penalty": 1.2,
            "top_k": 30,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )

    embeddings = LLMClient(
        base_url=base_url,
        model="embedding-gemma-300m",
        tokenizer = tokenizer
    )

    # ─────────────────────────────────────────────────────────────
    # Registry — add as many sets as you want here
    # ─────────────────────────────────────────────────────────────
    registry = ActionRegistry()
    registry.register_set("terminal set", get_terminal_actions())
    registry.register_set("file set", get_file_actions())
    # Add more sets here later...
    reasoning_tools = InternalTools(
        client=reasoner,
        description=(
            "Powerful reasoning model optimized for step-by-step thinking, "
            "complex math, planning, code writing, and long chains of thought. "
            "Use when the main model is struggling with depth or precision."
        ),
        methods=[
            client.ask,  # quick single-turn
            client.stream_chat,  # streaming long reasoning traces
            client.chat,  # non-streaming full response
        ],
    )
    reasoning_set = reasoning_tools.as_action_set("HeavyReasoning")
    registry.register_set("HeavyReasoning", reasoning_set)

    dynamic_manager = DynamicActionManager(registry)
    ToolBrowser(registry, dynamic_manager)  # auto-adds itself

   
    
    context_manager = ContextManager(client=embeddings,tokenizer = tokenizer)

    app = LLMChatApp(
        client=client,
        system_prompt=system_prompt,
        tools=dynamic_manager,  # ← works now
        context_manager=context_manager,
        tool_rendering="info",
    )
    app.run()


if __name__ == "__main__":
    main()
