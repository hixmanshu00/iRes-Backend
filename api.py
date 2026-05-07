from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pipeline import iter_research_pipeline
from research_runtime import ResearchRuntime

app = FastAPI(title="iRes API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173", "http://127.0.0.1:5173",
                   "http://localhost:8080", "http://127.0.0.1:8080", "https://ires.hiowner00.workers.dev"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Per-run queues bridge the sync pipeline thread → async SSE generator
_run_queues: dict[str, asyncio.Queue[dict[str, Any] | None]] = {}
_executor = ThreadPoolExecutor(max_workers=4)


class ResearchRequest(BaseModel):
    topic: str


@app.post("/api/research")
async def start_research(body: ResearchRequest) -> dict[str, str]:
    topic = (body.topic or "").strip()
    if not topic:
        raise HTTPException(status_code=400, detail="topic is required")

    loop = asyncio.get_event_loop()
    from research_runtime import new_run_id
    run_id = new_run_id()
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    _run_queues[run_id] = queue

    def _run() -> None:
        try:
            for step, state in iter_research_pipeline(topic):
                event: dict[str, Any] = {
                    "step": step,
                    "state": {"topic": topic, **state},
                }
                asyncio.run_coroutine_threadsafe(queue.put(event), loop).result()
        except Exception as exc:
            err_event: dict[str, Any] = {
                "step": "refiner",
                "state": {"run_id": run_id, "topic": topic, "errors": {"pipeline": str(exc)}},
            }
            asyncio.run_coroutine_threadsafe(queue.put(err_event), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

    _executor.submit(_run)
    return {"run_id": run_id}


@app.get("/api/research/{run_id}/stream")
async def stream_research(run_id: str) -> StreamingResponse:
    queue = _run_queues.get(run_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="Run not found or already completed")

    async def _generate():
        try:
            while True:
                event = await asyncio.wait_for(queue.get(), timeout=360)
                if event is None:
                    break
                yield f"data: {json.dumps(event, ensure_ascii=True)}\n\n"
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'step': 'refiner', 'state': {'errors': {'pipeline': 'timeout'}}})}\n\n"
        finally:
            _run_queues.pop(run_id, None)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/runs")
async def list_runs() -> list[dict[str, Any]]:
    return ResearchRuntime().list_runs(limit=30)


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    run = ResearchRuntime().load_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if "was_refined" not in run:
        run["was_refined"] = bool(
            run.get("refined_report") and run.get("refined_report") != run.get("report")
        )
    return run
