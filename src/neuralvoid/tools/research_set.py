import json
import asyncio
from typing import Optional
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
    record_to_context=True,
)
async def provide_context(agent, query: str):
    results = await agent.context_manager.provide_context(
        query=query,
        research_mode=True,
        return_as_string=True,
        max_input_tokens=agent.max_tokens * 0.65,
        reserved_for_output=agent.client.max_tokens * 0.35,
    )
    return results


@tool(
    "ResearchTools",
    name="ConductResearch",
    description=(
        "PRIMARY RESEARCH TOOL for deep analysis and professional report generation. "
        "Expands the topic into multiple targeted sub-queries, retrieves a rich context chunk from the local knowledgebase "
        "(GetContext when local=True) and/or the web (search_web + indexing when web=True), then synthesizes a structured report via LLM. "
        "If you provide a file_output path, the tool will automatically save the final report to that file using the write_file tool "
        "and append a confirmation note to the returned text. "
        "Use this for any 'analyze', 'research', 'compare', 'theoretical alignment', or 'generate report' task. "
        "Set local=True (default) for knowledgebase-only, web=True for up-to-date external information, or both=True for hybrid research."
    ),
    tags=[
        "research",
        "analysis",
        "report",
        "knowledgebase",
        "web",
        "context",
        "investigate",
    ],
)
async def conduct_research(
    agent,
    topic: str,
    local: bool = True,
    web: bool = False,
    file_output: Optional[str] = None,
):
    """Unified research conductor: local KB + optional web search → multi-query retrieval → LLM synthesis → optional file write."""
    if not topic or not topic.strip():
        logger.warning("ConductResearch called with empty topic")
        return "Error: topic cannot be empty."

    logger.info(
        f"ConductResearch started | topic='{topic[:100]}...' | local={local} | web={web} | file_output={file_output}"
    )

    # Step 1: Generate multiple diverse sub-queries from the main topic
    logger.debug("Generating multiple diverse search queries via PromptBuilder")
    multi_query_prompt = PromptBuilder.analysis_multi_query_generation(topic)
    multi_query_response = await agent.client.chat(messages=multi_query_prompt)
    logger.debug(f"Received multi-query response (length={len(multi_query_response)})")

    # Parse queries (JSON array or fallback to lines)
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

    queries = queries[:6]  # limit to avoid token explosion
    if not queries:
        queries = [topic]
        logger.debug("No queries parsed — using original topic as fallback")

    logger.info(f"Generated {len(queries)} sub-queries for research")

    all_research = []

    # Local knowledgebase retrieval (GetContext)
    if local:
        logger.info(
            f"Collecting local knowledgebase context via GetContext ({len(queries)} queries)"
        )
        for sub_query in queries:
            logger.debug(f"Executing GetContext for: {sub_query[:80]}...")
            try:
                result = await agent.manager.execute_direct(
                    "GetContext", query=sub_query
                )
                if result and str(result).strip():
                    all_research.append(f"--- Local KB: {sub_query} ---\n{result}\n")
                    logger.debug(f"Retrieved {len(str(result))} chars from GetContext")
                else:
                    logger.debug(f"No local results for sub-query: {sub_query[:50]}...")
            except Exception as e:
                logger.error(f"GetContext failed for '{sub_query[:50]}...': {e}")
                all_research.append(f"--- Local KB: {sub_query} ---\n[Error: {e}]\n")

    # Web research path
    if web:
        logger.info("Performing web search + indexing for external knowledge")
        search_text = await agent.manager.execute_direct(
            "search_web", query=topic, max_results=8
        )
        logger.debug(f"Web search completed | results_length={len(search_text)} chars")

        indexed_count = 0
        if (
            not search_text.startswith("search_web error")
            and "(no results)" not in search_text.lower()
        ):
            lines = search_text.split("\n\n")
            for line in lines[:8]:
                try:
                    if "http" in line.lower():
                        parts = line.split("\n")
                        if len(parts) > 1:
                            url = parts[1].strip()
                            if url.startswith("http"):
                                logger.debug(f"Indexing: {url[:80]}...")
                                await agent.manager.execute_direct(
                                    "index_web_page", url=url
                                )
                                indexed_count += 1
                except Exception as idx_err:
                    logger.error(f"Failed to index URL: {idx_err}")
                    continue

            if indexed_count > 0:
                logger.info(f"Indexed {indexed_count} web pages into knowledgebase")
                await asyncio.sleep(0.4)  # allow indexing to settle

        # Re-query GetContext so newly indexed web content is included in the research chunk
        logger.debug("Re-querying GetContext to pull in newly indexed web content")
        for sub_query in queries:
            try:
                result = await agent.manager.execute_direct(
                    "GetContext", query=sub_query
                )
                if result and str(result).strip():
                    all_research.append(
                        f"--- Web-enhanced KB: {sub_query} ---\n{result}\n"
                    )
            except Exception as e:
                logger.error(
                    f"Post-index GetContext failed for '{sub_query[:50]}...': {e}"
                )

    combined_research = (
        "\n".join(all_research)
        if all_research
        else "No research data collected (both local and web disabled or empty)."
    )
    logger.debug(
        f"Final combined research context length: {len(combined_research)} chars"
    )

    # Step 3: LLM synthesizes the structured report
    logger.debug("Synthesizing professional structured report via LLM")
    report_prompt = PromptBuilder.analysis_report_synthesis(topic, combined_research)
    final_report = await agent.client.chat(
        messages=report_prompt, max_tokens=agent.max_tokens
    )
    logger.info(f"LLM report synthesis completed | length={len(final_report)} chars")

    # Optional file output via write_file tool
    if file_output:
        try:
            write_result = await agent.manager.execute_direct(
                "write_file",
                file_path=file_output,
                content=final_report,
                append=False,
            )
            logger.info(
                f"Report saved to file | path={file_output} | result={write_result}"
            )
            final_report += f"\n\n[Report also written to: {file_output}]"
        except Exception as e:
            logger.error(f"Failed to write report to {file_output}: {e}")
            final_report += f"\n\n[ERROR: Could not save to file — {e}]"

    logger.info("ConductResearch finished successfully")
    return final_report
