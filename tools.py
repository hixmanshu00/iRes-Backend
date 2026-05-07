from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from rich import print

from dotenv import load_dotenv

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def scrape_web(query: str) -> str:
    """Search the web for information related to the query. Returns titles, urls, dates, and snippets."""
    try:
        search_results = tavily.search(query, max_results=7, search_depth="advanced")
        out = []
        for result in search_results["results"]:
            pub = result.get("published_date", "")
            date_line = f"Date: {pub}\n" if pub else ""
            out.append(
                f"Title: {result['title']}\n"
                f"URL: {result['url']}\n"
                f"{date_line}"
                f"Snippet: {result['content'][:500]}\n"
            )
        return "\n".join(out)
    except Exception as e:
        return f"An error occurred while searching the web: {str(e)}"
    

@tool
def scrape_url(url: str) -> str:
    """Scrape the content of a specific URL. Returns the text content of the page."""
    try:
        response = requests.get(url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style", "noscript", "nav", "header", "footer", "aside"]):
            script.decompose()  # Remove JavaScript and CSS
        return soup.get_text()[:2500]  # Return the first 2000 characters to avoid overload
    except Exception as e:
        return f"An error occurred while scraping the URL: {str(e)}"