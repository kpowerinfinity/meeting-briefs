import requests
from bs4 import BeautifulSoup
from typing import List, Dict

def web_search(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Perform a web search and return a list of results with title, url, and snippet."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    url = f"https://www.bing.com/search?q={requests.utils.quote(query)}"
    resp = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(resp.text, "lxml")
    results = []
    for item in soup.select("li.b_algo"):
        title_el = item.select_one("h2")
        link_el = title_el.find("a") if title_el else None
        snippet_el = item.select_one("p")
        if link_el and link_el.get("href"):
            results.append({
                "title": title_el.get_text(strip=True) if title_el else "",
                "url": link_el["href"],
                "snippet": snippet_el.get_text(strip=True) if snippet_el else ""
            })
        if len(results) >= max_results:
            break
    return results 