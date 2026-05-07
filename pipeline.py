from __future__ import annotations

from typing import Any, Dict, Iterator, Tuple

from research_runtime import ResearchRuntime, new_run_id, parse_sources

# Step ids match UI pipeline labels
StepId = str


def iter_research_pipeline(topic: str) -> Iterator[Tuple[StepId, Dict[str, Any]]]:
    """
    Run the multi-agent pipeline step by step.

    Yields ``(step_id, accumulated_state)`` after each step completes so the UI
    can update progress without duplicating agent logic.
    """
    topic = (topic or "").strip()
    runtime = ResearchRuntime()
    run_id = new_run_id()
    state: Dict[str, Any] = {}

    print("\n" + " =" * 50)
    print("step 1 - search agent is working ...")
    print("=" * 50)

    search_result = runtime.run_search(run_id=run_id, topic=topic)
    state["run_id"] = run_id
    state["search_results"] = search_result.output
    state["sources"] = parse_sources(state["search_results"])
    state["metrics"] = {"search_latency_s": round(search_result.latency_s, 3)}
    state["errors"] = {}
    if search_result.error:
        state["errors"]["search"] = search_result.error
    print("\n search result ", state["search_results"])
    yield "search", dict(state)

    print("\n" + " =" * 50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("=" * 50)

    reader_result = runtime.run_reader(run_id=run_id, topic=topic, search_results=state["search_results"])
    state["scraped_content"] = reader_result.output
    state["metrics"]["reader_latency_s"] = round(reader_result.latency_s, 3)
    if reader_result.error:
        state["errors"]["reader"] = reader_result.error
    print("\nscraped content: \n", state["scraped_content"])
    yield "reader", dict(state)

    print("\n" + " =" * 50)
    print("step 3 - Writer is drafting the report ...")
    print("=" * 50)

    research_combined = f"SEARCH RESULTS :\n{state['search_results']}\n\nDETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    writer_result = runtime.run_writer(run_id=run_id, topic=topic, research_text=research_combined)
    state["report"] = writer_result.output
    state["metrics"]["writer_latency_s"] = round(writer_result.latency_s, 3)
    if writer_result.fallback_used:
        state["metrics"]["writer_fallback_used"] = True
    if writer_result.error:
        state["errors"]["writer"] = writer_result.error
    print("\n Final Report\n", state["report"])
    yield "writer", dict(state)

    print("\n" + " =" * 50)
    print("step 4 - critic is reviewing the report ")
    print("=" * 50)

    critic_result = runtime.run_critic(run_id=run_id, report=state["report"])
    state["feedback"] = critic_result.output
    state["critic_score"] = None
    state["metrics"]["critic_latency_s"] = round(critic_result.latency_s, 3)
    if critic_result.fallback_used:
        state["metrics"]["critic_fallback_used"] = True
    if critic_result.error:
        state["errors"]["critic"] = critic_result.error

    from research_runtime import score_from_feedback
    state["critic_score"] = score_from_feedback(state["feedback"])
    print("\n critic report \n", state["feedback"])
    yield "critic", dict(state)

    print("\n" + " =" * 50)
    print("step 5 - Refiner is improving the report ...")
    print("=" * 50)

    refiner_result = runtime.run_refiner(
        run_id=run_id,
        topic=topic,
        report=state["report"],
        feedback=state["feedback"],
        score=state["critic_score"],
    )
    state["refined_report"] = refiner_result.output
    state["was_refined"] = not refiner_result.from_cache and refiner_result.latency_s > 0.01
    state["metrics"]["refiner_latency_s"] = round(refiner_result.latency_s, 3)
    state["metrics"]["total_latency_s"] = round(
        state["metrics"].get("search_latency_s", 0.0)
        + state["metrics"].get("reader_latency_s", 0.0)
        + state["metrics"].get("writer_latency_s", 0.0)
        + state["metrics"].get("critic_latency_s", 0.0)
        + state["metrics"].get("refiner_latency_s", 0.0),
        3,
    )
    if refiner_result.fallback_used:
        state["metrics"]["refiner_fallback_used"] = True
    if refiner_result.error:
        state["errors"]["refiner"] = refiner_result.error
    print("\n refined report \n", state["refined_report"])

    approved_source_ids = [s["id"] for s in state.get("sources", []) if s.get("approved")]
    runtime.save_run(
        run_id=run_id,
        topic=topic,
        sources=state.get("sources", []),
        approved_source_ids=approved_source_ids,
        search_results=state.get("search_results", ""),
        scraped_content=state.get("scraped_content", ""),
        report=state.get("report", ""),
        refined_report=state.get("refined_report", ""),
        feedback=state.get("feedback", ""),
        metrics=state.get("metrics", {}),
        errors=state.get("errors", {}),
    )
    yield "refiner", dict(state)


def run_research_pipeline(topic: str) -> dict:
    """Run the full pipeline and return the final accumulated state."""
    final: Dict[str, Any] = {}
    for _, partial in iter_research_pipeline(topic):
        final = partial
    return final



if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    run_research_pipeline(topic)
