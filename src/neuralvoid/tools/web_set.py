from neuralcore.actions.registry import tool, sequenced
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from ddgs import DDGS


# ─────────────────────────────────────────────────────────────
# WEB TOOLS (Free • No API • DuckDuckGo + lightweight scraper)
# ─────────────────────────────────────────────────────────────


@tool(
    "WebTools",
    tags=["web", "search"],
    name="search_web",
    description="Perform free web search using DuckDuckGo.",
)
async def search_web(query: str, max_results: int = 5) -> str:
    """Free web search via DuckDuckGo (no API key)."""
    try:

        def _search_sync():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        results = await asyncio.to_thread(_search_sync)
        if not results:
            return "(no results)"
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}\n   {r['href']}\n   {r['body'][:200]}...")
        return "\n\n".join(lines)
    except Exception as e:
        return f"search_web error: {str(e)}"


@tool(
    "WebTools",
    tags=["web", "browse", "scrape"],
    name="fetch_web_page",
    description="Fetch any webpage and return clean text content.",
)
async def fetch_web_page(url: str) -> str:
    """Fetch and clean webpage (with browser headers to avoid blocks)."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return f"HTTP {resp.status}"
                html = await resp.text()
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return text[:15000] if text else "(empty page)"
    except Exception as e:
        return f"fetch_web_page error: {str(e)}"


@tool(
    "WebTools",
    tags=["web", "index", "kb"],
    name="index_web_page",
    description="Fetch webpage and add its content to knowledge base.",
)
async def index_web_page(agent, url: str) -> str:
    """Fetch page and index into knowledge base."""
    content = await fetch_web_page(url)
    if content.startswith(("HTTP", "fetch_web_page error")):
        return content
    await agent.context_manager.add_external_content(
        source_type="indexed_web",
        content=content,
        metadata={"url": url, "type": "webpage"},
    )
    return f"✅ Indexed webpage '{url}'"


@tool(
    "WebTools",
    tags=["web", "search", "index", "kb"],
    name="search_web_and_index",
    description="Search the web then automatically index the top result pages.",
)
async def search_web_and_index(agent, query: str, max_results: int = 3) -> str:
    """Search web then index top pages."""
    search_text = await search_web(query, max_results=max_results)
    if "(no results)" in search_text or "error" in search_text.lower():
        return search_text
    lines = search_text.split("\n\n")
    indexed = 0
    for line in lines[:max_results]:
        if "http" in line.lower():
            url = line.split("\n")[1].strip()
            if url.startswith("http"):
                await index_web_page(agent, url)
                indexed += 1
    return f"✅ Searched '{query}' and indexed {indexed} pages"


@sequenced(
    name="research_topic",
    description="Quick & reliable research: search + auto-index + summary.",
    set_name="WebTools",
    tags=["web", "research", "index", "kb", "workflow"],
    propagate=True,
    output_from="GetContext",
    dependencies={
        "search_web_and_index": {"query": "input", "max_results": "max_results"},
        "GetContext": {
            "query": "input",
        },
    },
    steps=["search_web_and_index", "GetContext"],
)
def research_topic():
    pass
