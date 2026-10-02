from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from coach.app import build_router, create_app
from coach.store import SessionStore

LINES = [
    "Thanks for making the time. What does your current setup look like today?",
    "I hear you, and I'm sorry this happened. Is it the price itself, or the timing within your budget year?",
    "Teams your size usually save around 6 hours a week on manual reporting.",
    "Let's book 30 minutes next week, I'll send a calendar invite now.",
]


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def _session(client, trainee="trainee-1", scenario="renewal_price_increase"):
    r = client.post("/sessions", json={"trainee_id": trainee, "scenario_id": scenario})
    assert r.status_code == 200
    return r.json()


def _talk(client, session_id, lines):
    ends = []
    with client.websocket_connect(f"/sessions/{session_id}/stream") as ws:
        for line in lines:
            ws.send_json({"type": "message", "text": line})
            assert ws.receive_json()["type"] == "start"
            deltas = []
            while True:
                event = ws.receive_json()
                if event["type"] == "delta":
                    deltas.append(event["text"])
                else:
                    assert event["type"] == "end"
                    ends.append({**event, "text": "".join(deltas)})
                    break
        ws.send_json({"type": "end_session"})
        assert ws.receive_json()["type"] == "session_ended"
    return ends


def test_health_and_scenarios(client):
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.get("/readyz").status_code == 200
    assert len(client.get("/scenarios").json()) == 3


def test_unknown_scenario_is_404(client):
    assert (
        client.post("/sessions", json={"trainee_id": "t", "scenario_id": "nope"}).status_code == 404
    )


def test_full_roleplay_then_feedback(client):
    info = _session(client)
    assert info["variant"] in ("control", "realism_v2")
    ends = _talk(client, info["session_id"], LINES)
    assert all(e["text"] and not e["degraded"] for e in ends)
    assert all(e["ttft_ms"] > 0 for e in ends)

    r = client.post(f"/sessions/{info['session_id']}/feedback")
    assert r.status_code == 200
    body = r.json()
    assert body["overall"] >= 4
    assert body["tool_calls"] == 2
    assert body["cost_usd"] > 0

    session = client.get(f"/sessions/{info['session_id']}").json()
    assert session["status"] == "reviewed"
    assert len(session["turns"]) == 1 + 2 * len(LINES)
    assert set(session["cost"]["by_task"]) >= {"customer", "coach"}


def test_second_session_receives_previous_focus_areas(client):
    first = _session(client, trainee="t-focus")
    _talk(client, first["session_id"], ["It has dashboards, alerts and an API."])
    client.post(f"/sessions/{first['session_id']}/feedback")
    second = _session(client, trainee="t-focus")
    _talk(client, second["session_id"], LINES[:1])
    body = client.post(f"/sessions/{second['session_id']}/feedback").json()
    assert body["previous_focus"]  # long-term memory from the first session


def test_feedback_before_any_trainee_turn_is_409(client):
    info = _session(client)
    assert client.post(f"/sessions/{info['session_id']}/feedback").status_code == 409


def test_websocket_rejects_bad_messages_and_unknown_sessions(client):
    with client.websocket_connect("/sessions/nope/stream") as ws:
        assert ws.receive_json()["type"] == "error"
    info = _session(client)
    with client.websocket_connect(f"/sessions/{info['session_id']}/stream") as ws:
        ws.send_json({"type": "message", "text": "   "})
        assert ws.receive_json()["type"] == "error"


def test_degrades_gracefully_when_every_customer_model_fails(settings):
    store = SessionStore(settings.db_path)
    failing = settings.__class__(
        **{**settings.__dict__, "mock_failure_rate": {"mock/fast-a": 1.0, "mock/fast-b": 1.0}}
    )
    app = create_app(failing, router=build_router(failing, store), store=store)
    with TestClient(app) as c:
        info = _session(c)
        ends = _talk(c, info["session_id"], ["Hello?"])
        assert ends[0]["degraded"] and ends[0]["text"]  # canned line, no crash
        metrics = c.get("/metrics").text
        assert "coach_degraded_replies_total" in metrics


def test_feedback_returns_503_when_coach_models_are_down(settings):
    store = SessionStore(settings.db_path)
    failing = settings.__class__(
        **{**settings.__dict__, "mock_failure_rate": {"mock/smart-a": 1.0, "mock/smart-b": 1.0}}
    )
    app = create_app(failing, router=build_router(failing, store), store=store)
    with TestClient(app) as c:
        info = _session(c)
        _talk(c, info["session_id"], LINES[:1])
        r = c.post(f"/sessions/{info['session_id']}/feedback")
        assert r.status_code == 503 and r.headers["retry-after"] == "10"


def test_online_eval_and_experiment_report(client):
    for i in range(6):
        info = _session(client, trainee=f"t{i}")
        _talk(client, info["session_id"], LINES[: 1 + i % 4])
        client.post(f"/sessions/{info['session_id']}/feedback")
    time.sleep(0.2)  # sampled online evals run in the background
    report = client.get("/experiments/report").json()
    assert report["n_control"] + report["n_treatment"] == 6
    assert report["decision"] in ("keep_running", "invalid")
    assert "coach_online_judge_gap_count" in client.get("/metrics").text
