from neuralcore.actions.manager import tool
from neuralcore.agents.core import Agent


@tool(
    "ContextManager",
    name="Get Context",
    description="use this tool to search your own memory",
)
async def provide_context(agent: Agent, query):
    return await agent.context_manager.provide_context(query)
