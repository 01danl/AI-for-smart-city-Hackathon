"""Streamlit bridge for a persistent browser audio session and server-side POI tools."""

from __future__ import annotations

from pathlib import Path
import time

import streamlit as st
import streamlit.components.v1 as components
from pydantic import ValidationError

from backend.generation import live_voice as service
from backend.generation.recommender import RecommendRequest, RecommendResponse
from backend.retrieval.geo import round_coords

_component = components.declare_component(
    "astana_live_voice", path=str(Path(__file__).with_name("live_voice_component")),
)


def reset_live_voice() -> None:
    """A new iframe key releases microphone/socket when navigating to another conversation."""
    st.session_state.pop("_live_voice", None)
    st.session_state["_live_voice_epoch"] = st.session_state.get("_live_voice_epoch", 0) + 1


def handle_event(event: dict, state: dict, context: RecommendRequest) -> None:
    """Acknowledge every event once; cancelled/unconsumed results never become map cards."""
    if not isinstance(event, dict) or not isinstance(event.get("id"), str):
        return
    if state.get("ack") == event["id"]:
        return
    state["ack"] = event["id"]
    session = event.get("session")
    kind = event.get("type")
    if kind == "connect" and isinstance(session, str) and 0 < len(session) <= 100:
        last_attempt = state.get("last_attempt", 0)
        state.update(session=session, pending={}, tool_results=[], grant=None, last_attempt=time.monotonic())
        lat, lon = round_coords(context.lat, context.lon)
        state["context"] = context.model_copy(update={"lat": lat, "lon": lon}).model_dump_json()
        try:
            if time.monotonic() - last_attempt < 3:
                raise service.LiveVoiceError("Подождите несколько секунд перед новым подключением.")
            state["grant"] = {"session": session, **service.create_grant()}
        except service.LiveVoiceError as exc:
            state["grant"] = {"session": session, "error": str(exc)}
        return
    if kind == "stopped" and event.get("closed_session") == state.get("session"):
        state.update(session=None, grant=None, tool_results=[], pending={})
        return
    if not session or session != state.get("session"):
        return
    if kind == "tool":
        results = []
        calls = event.get("calls")
        if not isinstance(calls, list):
            return
        for call in calls[:3]:
            if not isinstance(call, dict) or not isinstance(call.get("id"), str):
                continue
            call_id = call["id"][:200]
            result = {"id": call_id, "name": call.get("name", ""), "response": {"error": "Unknown tool"}}
            try:
                if call.get("name") != "search_places":
                    results.append(result)
                    continue
                # A repeated ID reuses the prior result, never a second search.
                if call_id in state["pending"]:
                    results.append(state["pending"][call_id]["tool"])
                    continue
                if len(state["pending"]) >= 30:
                    raise ValueError("Session tool limit")
                req = RecommendRequest.model_validate_json(state["context"])
                result["response"], response = service.search_places(call.get("args", {}), req)
                display_req = req.model_copy(update={
                    "query": call["args"]["query"], "auto_anchor": False,
                    "max_distance_m": result["response"]["used_data"]["radius_m"],
                })
                state["pending"][call_id] = {
                    "tool": result, "req_json": display_req.model_dump_json(),
                    "response_json": response.model_dump_json(),
                    "query": str(event.get("question") or call["args"]["query"])[:1000],
                }
            except (ValidationError, ValueError, TypeError, KeyError):
                result["response"] = {"error": "Invalid search arguments. Ask a short clarification and try again."}
            except Exception:
                # Provider/model text must never contain stack traces, credentials or raw GPS.
                result["response"] = {"error": "Local POI search unavailable. Do not invent places. Please try again."}
            results.append(result)
        state["tool_results"] = results
    elif kind == "used":
        ids = event.get("call_ids")
        if isinstance(ids, list):
            for call_id in ids[:3]:
                if isinstance(call_id, str) and call_id in state.get("pending", {}):
                    state["latest"] = state["pending"][call_id]


def render_live_voice(context: RecommendRequest) -> dict | None:
    state = st.session_state.setdefault("_live_voice", {"ack": None, "pending": {}, "tool_results": []})
    event = _component(
        configured=service.configured(), ack=state.get("ack"), session=state.get("session"),
        grant=state.get("grant"), tool_results=state.get("tool_results", []),
        key=f"live_voice_{st.session_state.get('_live_voice_epoch', 0)}", default=None,
    )
    if isinstance(event, dict) and event.get("id") != state.get("ack"):
        handle_event(event, state, context)
        st.rerun()
    latest = state.get("latest")
    if latest:
        return {
            "request": RecommendRequest.model_validate_json(latest["req_json"]),
            "response": RecommendResponse.model_validate_json(latest["response_json"]),
            "query": latest["query"],
        }
    return None
