from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import scrape_web, scrape_url
from dotenv import load_dotenv

load_dotenv()

PRIMARY_MODEL  = "mistral-small-latest"
FALLBACK_MODEL = "open-mistral-7b"

llm          = ChatMistralAI(model=PRIMARY_MODEL,  temperature=0)
fallback_llm = ChatMistralAI(model=FALLBACK_MODEL, temperature=0)


def build_search_agent(use_fallback: bool = False):
    model = fallback_llm if use_fallback else llm
    return create_agent(model=model, tools=[scrape_web])


def build_url_scraper_agent(use_fallback: bool = False):
    model = fallback_llm if use_fallback else llm
    return create_agent(model=model, tools=[scrape_url])


# ── Writer ────────────────────────────────────────────────────────────────────

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain          = writer_prompt | llm          | StrOutputParser()
fallback_writer_chain = writer_prompt | fallback_llm | StrOutputParser()


# ── Critic ────────────────────────────────────────────────────────────────────

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain          = critic_prompt | llm          | StrOutputParser()
fallback_critic_chain = critic_prompt | fallback_llm | StrOutputParser()


# ── Refiner ───────────────────────────────────────────────────────────────────

refiner_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Improve reports based on specific critic feedback."),
    ("human", """A research report on "{topic}" was reviewed by a critic.

Original Report:
{report}

Critic Feedback:
{feedback}

Rewrite the report to directly address every point in "Areas to Improve".
Keep the same structure. Make it more accurate, deeper, and better sourced.
Do NOT add a note saying this is a revision — just output the improved report."""),
])

refiner_chain          = refiner_prompt | llm          | StrOutputParser()
fallback_refiner_chain = refiner_prompt | fallback_llm | StrOutputParser()


def get_model_params() -> dict:
    return {
        "primary_model":  PRIMARY_MODEL,
        "fallback_model": FALLBACK_MODEL,
        "temperature":    0,
    }
