"""FastAPI service: sessions, real-time roleplay over WebSocket, structured feedback.

Run locally:  uvicorn --factory coach.app:create_app --reload
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from coach import telemetry as tm
from coach.agents.coach import CoachAgent
from coach.agents.customer import FALLBACK_LINES, CustomerAgent
from coach.config import Settings
from coach.experiments import assign
from coach.judge import LLMJudge
from coach.llm.base import AllModelsFailed, LLMError
from coach.llm.mock import MockProvider
from coach.llm.router import ModelRouter, StreamInterrupted, StreamResult
from coach.llm.scripted import make_responder
from coach.memory import ConversationMemory
from coach.scenarios import SCENARIOS, get_scenario
from coach.schemas import DIMENSIONS, CreateSessionRequest, SessionInfo
from coach.store import SessionStore

logger = logging.getLogger("coach.app")


def build_router(
    settings: Settings, store: SessionStore | None = None, providers: dict | None = None
) -> ModelRouter:
    if providers is None:
        providers = {
            "mock": MockProvider(
                make_responder(settings.mock_bad_json_rate),
                latency_ms=settings.mock_latency_ms,
                token_delay_ms=settings.mock_token_delay_ms,
                failure_rate=settings.mock_failure_rate,
            )
        }
        if settings.provider_mode == "live":
            from coach.llm.litellm_provider import LiteLLMProvider

            providers["*"] = LiteLLMProvider()
    return ModelRouter(
        providers,
        {
            "customer": settings.customer_models,
            "coach": settings.coach_models,
            "judge": settings.judge_models,
            "judge_pairwise": settings.judge_models,
        },
        rate_limit_rpm=settings.rate_limit_rpm,
        prioritise_realtime=settings.prioritise_realtime,
        timeout_s=settings.request_timeout_s,
        max_retries=settings.max_retries,
        backoff_base_s=settings.backoff_base_s,
        breaker_failure_threshold=settings.breaker_failure_threshold,
        breaker_reset_s=settings.breaker_reset_s,
        on_call=store.add_call if store else None,
    )


def create_app(
    settings: Settings | None = None,
    router: ModelRouter | None = None,
    store: SessionStore | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    store = store or SessionStore(settings.db_path)
    router = router or build_router(settings, store)
    if router._on_call is None:
        router._on_call = store.add_call
    coach_agent = CoachAgent(router)
    judge = LLMJudge(router)
    memories: dict[str, ConversationMemory] = {}
    background: set[asyncio.Task] = set()
    rng = random.Random()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        tm.setup_logging()
        tm.setup_tracing()
        tm.log(
            logger,
            logging.INFO,
            "service started",
            mode=settings.provider_mode,
            customer_models=settings.customer_models,
            coach_models=settings.coach_models,
        )
        yield
        for task in list(background):
            task.cancel()

    app = FastAPI(title="Conversation Coach Agent", version="0.1.0", lifespan=lifespan)
    app.state.settings, app.state.store, app.state.router = settings, store, router
    app.state.background = background

    def memory_for(session_id: str) -> ConversationMemory:
        """Per-instance cache; rebuilt from the store if this instance has not seen the session."""
        if session_id not in memories:
            memory = ConversationMemory(settings.memory_token_budget)
            for row in store.turns(session_id):
                memory.add("user" if row["role"] == "trainee" else "assistant", row["content"])
            memories[session_id] = memory
        return memories[session_id]

    # ------------------------------------------------------------------ basics
    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "mode": settings.provider_mode}

    @app.get("/readyz")
    async def readyz():
        store._all("SELECT 1")
        return {"status": "ready"}

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/scenarios")
    async def scenarios():
        return [{"id": s.id, "title": s.title} for s in SCENARIOS.values()]

    @app.exception_handler(AllModelsFailed)
    async def all_failed(_: Request, exc: AllModelsFailed):
        return JSONResponse(
            status_code=503,
            content={"detail": "model providers unavailable, retry shortly"},
            headers={"Retry-After": "10"},
        )

    # ------------------------------------------------------------------ sessions
    @app.post("/sessions", response_model=SessionInfo)
    async def create_session(body: CreateSessionRequest):
        try:
            scenario = get_scenario(body.scenario_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        variant = assign(settings.experiment_name, body.trainee_id)
        session_id = store.create_session(body.trainee_id, scenario.id, variant)
        store.add_turn(session_id, "customer", scenario.opening_line, model="scripted")
        memory_for(session_id)
        tm.log(
            logger,
            logging.INFO,
            "session created",
            session_id=session_id,
            scenario=scenario.id,
            variant=variant,
        )
        return SessionInfo(
            session_id=session_id,
            scenario_id=scenario.id,
            variant=variant,
            opening_line=scenario.opening_line,
        )

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        session = store.get_session(session_id)
        if session is None:
            raise HTTPException(404, "unknown session")
        return {
            "session_id": session_id,
            "scenario_id": session["scenario_id"],
            "variant": session["variant"],
            "status": session["status"],
            "turns": [
                {"role": r["role"], "content": r["content"], "degraded": bool(r["degraded"])}
                for r in store.turns(session_id)
            ],
            "feedback": store.get_feedback(session_id),
            "cost": store.session_cost(session_id),
        }

    # ------------------------------------------------------------------ realtime roleplay
    @app.websocket("/sessions/{session_id}/stream")
    async def stream(websocket: WebSocket, session_id: str):
        await websocket.accept()
        session = store.get_session(session_id)
        if session is None:
            await websocket.send_json({"type": "error", "detail": "unknown session"})
            await websocket.close(code=4404)
            return
        agent = CustomerAgent(
            router, get_scenario(session["scenario_id"]), session["variant"], memory_for(session_id)
        )
        try:
            while True:
                event = await websocket.receive_json()
                if event.get("type") == "end_session":
                    store.set_status(session_id, "ended")
                    await websocket.send_json({"type": "session_ended"})
                    await websocket.close()
                    return
                text = str(event.get("text", "")).strip()
                if event.get("type") != "message" or not text:
                    await websocket.send_json(
                        {"type": "error", "detail": "expected {type: message, text}"}
                    )
                    continue
                if (
                    sum(1 for r in store.turns(session_id) if r["role"] == "trainee")
                    >= settings.max_turns
                ):
                    await websocket.send_json(
                        {"type": "error", "detail": "turn limit reached, request feedback"}
                    )
                    continue
                store.add_turn(session_id, "trainee", text[:2000])
                await _stream_reply(websocket, agent, session_id, text[:2000])
        except WebSocketDisconnect:
            tm.log(logger, logging.INFO, "client disconnected", session_id=session_id)

    async def _stream_reply(
        websocket: WebSocket, agent: CustomerAgent, session_id: str, text: str
    ) -> None:
        started = time.perf_counter()
        ttft_ms: float | None = None
        result = StreamResult()
        parts: list[str] = []
        degraded = interrupted = False
        await websocket.send_json({"type": "start"})
        try:
            async for delta in agent.reply(text, result, session_id):
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - started) * 1000
                    tm.TTFT.observe(ttft_ms / 1000)
                parts.append(delta)
                await websocket.send_json({"type": "delta", "text": delta})
        except StreamInterrupted as exc:
            # Text already reached the user; keep it and say so rather than switching models mid-sentence.
            interrupted = degraded = True
            agent.remember_reply(exc.partial_text)
        except (AllModelsFailed, LLMError):
            degraded = True
            line = FALLBACK_LINES[len(parts) % len(FALLBACK_LINES)]
            parts = [line]
            agent.remember_reply(line)
            ttft_ms = (time.perf_counter() - started) * 1000
            await websocket.send_json({"type": "delta", "text": line})
        if degraded:
            tm.DEGRADED_REPLIES.inc()
        reply = "".join(parts)
        store.add_turn(
            session_id,
            "customer",
            reply,
            model=result.model or None,
            ttft_ms=ttft_ms,
            degraded=degraded,
        )
        await websocket.send_json(
            {
                "type": "end",
                "model": result.model,
                "fallback": result.fallback_used,
                "degraded": degraded,
                "interrupted": interrupted,
                "ttft_ms": round(ttft_ms or 0.0, 1),
            }
        )

    # ------------------------------------------------------------------ feedback
    @app.post("/sessions/{session_id}/feedback")
    async def feedback(session_id: str):
        session = store.get_session(session_id)
        if session is None:
            raise HTTPException(404, "unknown session")
        turns = [(r["role"], r["content"]) for r in store.turns(session_id)]
        if not any(role == "trainee" for role, _ in turns):
            raise HTTPException(409, "no trainee turns yet")
        scenario = get_scenario(session["scenario_id"])
        focus = store.weakest_dimensions(session["trainee_id"])
        try:
            result = await coach_agent.review(
                scenario, turns, focus_areas=focus, conversation_id=session_id
            )
        except LLMError as exc:
            tm.log(logger, logging.ERROR, "feedback failed", session_id=session_id, error=str(exc))
            return JSONResponse(
                status_code=503,
                content={"detail": "feedback unavailable, retry shortly"},
                headers={"Retry-After": "10"},
            )

        fb = result.feedback
        store.save_feedback(session_id, fb.model_dump(), fb.overall, result.model, result.repairs)
        store.save_focus(session["trainee_id"], {d: fb.score_of(d) for d in DIMENSIONS})
        store.set_status(session_id, "reviewed")
        cost = store.session_cost(session_id)
        tm.CONVERSATION_COST.observe(cost["total_usd"])

        if rng.random() < settings.online_eval_sample_rate:
            task = asyncio.create_task(_online_eval(session_id, turns, fb.overall))
            background.add(task)
            task.add_done_callback(background.discard)

        return {
            "session_id": session_id,
            "overall": fb.overall,
            "feedback": fb.model_dump(),
            "previous_focus": focus,
            "model": result.model,
            "repairs": result.repairs,
            "tool_calls": result.tool_calls,
            "cost_usd": cost["total_usd"],
        }

    async def _online_eval(session_id: str, turns, coach_overall: float) -> None:
        """Score a sample of live sessions with an independent judge; alert when they drift apart."""
        try:
            scores = await judge.score(turns)
        except Exception as exc:  # noqa: BLE001 - never let the sampler crash the service
            tm.log(
                logger, logging.WARNING, "online eval failed", session_id=session_id, error=str(exc)
            )
            return
        judge_overall = sum(scores.values()) / len(scores)
        store.save_online_eval(session_id, judge.name, judge_overall, coach_overall)
        tm.ONLINE_JUDGE_GAP.observe(abs(judge_overall - coach_overall))

    # ------------------------------------------------------------------ experiments
    @app.get("/experiments/report")
    async def experiment_report():
        from experiments.analysis import analyse_rows

        rows = store.experiment_rows()
        if not rows:
            raise HTTPException(404, "no finished sessions yet")
        return analyse_rows(rows, control="control", treatment="realism_v2").to_dict()

    return app
