from duckduckgo_search import DDGS

def google_search(query, num_results=5):
    """
    Fetches search snippets using DuckDuckGo Search API (ddgs).
    Returns a list of (title, snippet).
    """
    results = []
    with DDGS() as ddgs:
        search_results = ddgs.text(query, max_results=num_results)
        for res in search_results:
            results.append((res.get("title", ""), res.get("body", "")))
    return results
