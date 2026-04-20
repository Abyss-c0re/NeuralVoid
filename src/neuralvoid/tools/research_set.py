import asyncio
from neuralcore.actions.registry import tool
from neuralcore.utils.prompt_builder import PromptBuilder


@tool(
    "ResearchTools",
    name="SearchToolResults",
    description="Search and retrieve the most relevant content from previous tool outcomes and executed commands (including file contents, analysis results, logs, and other tool-generated data). Use this when you need to research what has already been discovered or processed by tools.",
    tags=["context", "research", "investigate", "tool_results", "memory"],
)
async def provide_context(agent, query: str):
    return await agent.provide_context(
        query=query,
        research_mode=True,
        return_as_string=True,
        lightweight_agentic=True,
    )


@tool(
    "ResearchTools",
    name="PerformAnalysis",
    description="Perform deep analysis on accumulated knowledge from tool outcomes. "
    "Provide a clear analysis query (e.g. 'analyze the payment processor integration issues and findings'). "
    "This tool will generate multiple targeted searches, retrieve relevant tool results, "
    "and synthesize a comprehensive structured report.",
    tags=["context", "research", "investigate", "analysis", "report", "tool_results"],
)
async def perform_analysis(agent, query: str):
    if not query or not query.strip():
        return "Error: Analysis query cannot be empty."

    # Step 1: Use PromptBuilder to generate multiple diverse search queries
    multi_query_prompt = PromptBuilder.analysis_multi_query_generation(query)
    multi_query_response = await agent.client.ask(multi_query_prompt)

    # Parse the response into a list of queries (expecting JSON array or numbered list)
    try:
        import json

        queries = json.loads(multi_query_response)
        if not isinstance(queries, list):
            queries = [q.strip() for q in multi_query_response.split("\n") if q.strip()]
    except Exception:
        # Fallback: simple line split
        queries = [
            q.strip()
            for q in multi_query_response.split("\n")
            if q.strip() and not q.startswith("#")
        ]

    # Limit to reasonable number (3-6) to avoid token explosion
    queries = queries[:6]
    if not queries:
        queries = [query]  # fallback to original

    # Step 2: Accumulate results from SearchToolResults for each generated query
    all_research = []
    for sub_query in queries:
        try:
            result = await agent.dynamic_manager.execute_direct(
                "SearchToolResults",
                query=sub_query,
            )
            if result and str(result).strip():
                all_research.append(f"--- Search for: {sub_query} ---\n{result}\n")
        except Exception as e:
            all_research.append(
                f"--- Search for: {sub_query} ---\n[Error retrieving: {e}]\n"
            )

    combined_research = (
        "\n".join(all_research) if all_research else "No relevant tool outcomes found."
    )

    # Step 3: Instruct LLM to generate a structured report
    report_prompt = PromptBuilder.analysis_report_synthesis(query, combined_research)
    final_report = await agent.client.ask(report_prompt)

    return final_report


@tool(
    "ResearchTools",
    name="ResearchWeb",
    description="Perform deep web research on a topic: search the web, index relevant pages into the knowledge base, "
    "then analyze everything using accumulated tool outcomes (including new web content). "
    "Returns a comprehensive structured analysis report. "
    "Best used when you need up-to-date external knowledge combined with previous tool findings.",
    tags=["web", "research", "investigate", "analysis", "report"],
)
async def research_web(agent, query: str, max_results: int = 5):
    """Research a topic on the web and deliver a synthesized analysis report."""
    if not query or not query.strip():
        return "Error: Research query cannot be empty."

    # Step 1: Perform web search (do NOT use results directly — let them go into context via indexing)
    search_text = await agent.dynamic_manager.execute_direct(
        "search_web",
        query=query,
        max_results=max_results,
    )

    # Step 2: Index the top results into the knowledge base (so they become tool outcomes)
    if (
        not search_text.startswith("search_web error")
        and "(no results)" not in search_text.lower()
    ):
        lines = search_text.split("\n\n")
        indexed_count = 0
        for line in lines[:max_results]:
            try:
                if "http" in line.lower():
                    parts = line.split("\n")
                    if len(parts) > 1:
                        url = parts[1].strip()
                        if url.startswith("http"):
                            await agent.dynamic_manager.execute_direct(
                                "index_web_page",
                                url=url,
                            )
                            indexed_count += 1
            except Exception:
                continue  # silent fail on individual indexing

        if indexed_count > 0:
            # Small pause to allow indexing to settle into KB
            await asyncio.sleep(0.3)

    # Step 3: Perform deep analysis using the new indexed web content + existing tool outcomes
    analysis_report = await agent.dynamic_manager.execute_direct(
        "PerformAnalysis",
        query=f"Web research on: {query}. Analyze all recent tool outcomes including newly indexed web pages.",
    )

    return analysis_report
