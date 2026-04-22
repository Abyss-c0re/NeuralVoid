import json
import asyncio
from neuralcore.actions.registry import tool
from neuralcore.utils.prompt_builder import PromptBuilder
from neuralcore.utils.logger import Logger

logger = Logger.get_logger()


@tool(
    "ResearchTools",
    name="GetContext",
    description=(
        "PRIMARY TOOL for searching the KnowledgeBase. "
        "Use this FIRST whenever the task requires: "
        "- Retrieving relevant content from indexed documents, PDFs, or previous tool outcomes "
        "- Researching what has already been discovered or processed "
        "- Gathering context for analysis or report writing. "
        "This is the main research/retrieval tool. Always prefer this over generic file reading when the query involves 'knowledge base', 'indexed documents', 'neuroscience', 'analysis', or 'research'."
    ),
    tags=[
        "context",
        "investigate",
        "tool_results",
        "memory",
        "knowledgebase",
    ],
    record_to_context=False
)
async def provide_context(agent, query: str):
    results = await agent.context_manager.provide_context(
        query=query,
        research_mode=True,
        return_as_string=True,
    )
    return results


@tool(
    "ResearchTools",
    name="PerformAnalysis",
    description=(
        "PRIMARY TOOL for deep analysis and report synthesis. "
        "Use this when the task requires: "
        "- Performing structured analysis on accumulated knowledge "
        "- Generating comprehensive reports with Theoretical Basis, Validation, Gap Analysis, and Recommendations "
        "- Synthesizing insights from multiple sources in the knowledgebase. "
        "This tool automatically generates multiple targeted searches and produces a professional structured report. "
        "Call this directly for any 'analyze', 'compare', 'theoretical alignment', or 'report' task."
    ),
    tags=[
        "context",
        "research",
        "investigate",
        "analysis",
        "report",
        "knowledgebase",
        "tool_results",
    ],
)
async def perform_analysis(agent, query: str):
    if not query or not query.strip():
        logger.warning("PerformAnalysis called with empty query")
        return "Error: Analysis query cannot be empty."

    logger.info(f"PerformAnalysis started | query='{query[:100]}...'")

    # Step 1: Use PromptBuilder to generate multiple diverse search queries
    logger.debug("Generating multiple diverse search queries via PromptBuilder")
    multi_query_prompt = PromptBuilder.analysis_multi_query_generation(query)
    multi_query_response = await agent.client.chat(messages=multi_query_prompt)
    logger.debug(f"Received multi-query response (length={len(multi_query_response)})")

    # Parse the response into a list of queries (expecting JSON array or numbered list)
    try:
        queries = json.loads(multi_query_response)
        if not isinstance(queries, list):
            queries = [q.strip() for q in multi_query_response.split("\n") if q.strip()]
        logger.debug(f"Parsed {len(queries)} queries from JSON response")
    except Exception as parse_err:
        logger.warning(f"Failed to parse JSON, falling back to line split: {parse_err}")
        queries = [
            q.strip()
            for q in multi_query_response.split("\n")
            if q.strip() and not q.startswith("#")
        ]

    # Limit to reasonable number (3-6) to avoid token explosion
    queries = queries[:6]
    if not queries:
        queries = [query]  # fallback to original
        logger.debug("No queries parsed, using original query as fallback")

    logger.info(f"Generated {len(queries)} sub-queries for analysis")

    # Step 2: Accumulate results from GetContext for each generated query
    all_research = []
    for sub_query in queries:
        logger.debug(f"Executing GetContext for sub-query: {sub_query[:80]}...")
        try:
            result = await agent.manager.execute_direct(
                "GetContext",
                query=sub_query,
            )
            if result and str(result).strip():
                all_research.append(f"--- Search for: {sub_query} ---\n{result}\n")
                logger.debug(f"Retrieved {len(str(result))} chars for sub-query")
            else:
                logger.debug(f"No results returned for sub-query: {sub_query[:50]}...")
        except Exception as e:
            logger.error(f"Failed to execute GetContext for '{sub_query[:50]}...': {e}")
            all_research.append(
                f"--- Search for: {sub_query} ---\n[Error retrieving: {e}]\n"
            )

    combined_research = (
        "\n".join(all_research) if all_research else "No relevant tool outcomes found."
    )
    logger.debug(f"Combined research context length: {len(combined_research)} chars")

    # Step 3: Instruct LLM to generate a structured report
    logger.debug("Synthesizing final structured report via LLM")
    report_prompt = PromptBuilder.analysis_report_synthesis(query, combined_research)
    final_report = await agent.client.chat(messages=report_prompt)
    logger.info(
        f"PerformAnalysis completed successfully | report_length={len(final_report)} chars"
    )

    return final_report


@tool(
    "ResearchTools",
    name="ResearchWeb",
    description=(
        "Use for external web research when the task requires up-to-date information not present in the local knowledgebase. "
        "Automatically searches the web, indexes results, and then performs deep analysis using PerformAnalysis. "
        "Only use when the query explicitly needs current external knowledge (e.g. latest papers, documentation, or real-time data)."
    ),
    tags=["web", "research", "investigate", "analysis", "report"],
)
async def research_web(agent, query: str, max_results: int = 5):
    """Research a topic on the web and deliver a synthesized analysis report."""
    if not query or not query.strip():
        logger.warning("ResearchWeb called with empty query")
        return "Error: Research query cannot be empty."

    logger.info(
        f"ResearchWeb started | query='{query[:100]}...' | max_results={max_results}"
    )

    # Step 1: Perform web search (do NOT use results directly — let them go into context via indexing)
    logger.debug("Performing web search via search_web tool")
    search_text = await agent.manager.execute_direct(
        "search_web",
        query=query,
        max_results=max_results,
    )
    logger.debug(f"Web search completed | results_length={len(search_text)} chars")

    # Step 2: Index the top results into the knowledge base (so they become tool outcomes)
    indexed_count = 0
    if (
        not search_text.startswith("search_web error")
        and "(no results)" not in search_text.lower()
    ):
        lines = search_text.split("\n\n")
        for line in lines[:max_results]:
            try:
                if "http" in line.lower():
                    parts = line.split("\n")
                    if len(parts) > 1:
                        url = parts[1].strip()
                        if url.startswith("http"):
                            logger.debug(f"Indexing web page: {url[:80]}...")
                            await agent.manager.execute_direct(
                                "index_web_page",
                                url=url,
                            )
                            indexed_count += 1
            except Exception as idx_err:
                logger.error(f"Failed to index URL from search results: {idx_err}")
                continue  # silent fail on individual indexing

        if indexed_count > 0:
            logger.info(f"Indexed {indexed_count} web pages into knowledge base")
            # Small pause to allow indexing to settle into KB
            await asyncio.sleep(0.3)
    else:
        logger.warning(
            f"Web search failed or returned no usable results: {search_text[:100]}..."
        )

    # Step 3: Perform deep analysis using the new indexed web content + existing tool outcomes
    logger.debug("Triggering PerformAnalysis on web research results + existing KB")
    analysis_report = await agent.manager.execute_direct(
        "PerformAnalysis",
        query=f"Web research on: {query}. Analyze all recent tool outcomes including newly indexed web pages.",
    )
    logger.info(
        f"ResearchWeb completed successfully | final_report_length={len(analysis_report)} chars"
    )
