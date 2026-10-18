"""Streamlit bridge for a persistent browser audio session and server-side POI tools."""

from __future__ import annotations

from pathlib import Path
import time
from collections.abc import Callable

import streamlit as st
import streamlit.components.v1 as components
from pydantic import ValidationError

from backend.generation import live_voice as service
from backend.generation.recommender import RecommendRequest
from backend.retrieval.geo import round_coords

_component = components.declare_component(
    "astana_live_voice", path=str(Path(__file__).with_name("live_voice_component")),
)


def reset_live_voice() -> None:
    """A new iframe key releases microphone/socket when navigating to another conversation."""
    for convo in st.session_state.get("convos", []):
        for entry in convo["entries"]:
            if entry.get("via") == "live_voice" and not entry.get("voice_final"):
                entry["voice_final"] = True
                entry["voice_interrupted"] = True
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
        state.update(session=session, history_session=session, turns={}, pending={}, tool_results=[],
                     grant=None, phase="connecting", last_attempt=time.monotonic())
        state.pop("start_label", None)
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
        for turn in state.get("turns", {}).values():
            turn["final"] = True
        state.update(session=None, grant=None, tool_results=[], pending={}, phase="idle")
        return
    if not session or session != state.get("session"):
        return
    if kind == "speech":
        turn_id = event.get("turn_id")
        if not isinstance(turn_id, str) or not 0 < len(turn_id) <= 100:
            return
        turn = state.setdefault("turns", {}).setdefault(turn_id, {})
        turn.update(question=str(event.get("question") or "")[:4000],
                    answer=str(event.get("answer") or "")[:16000],
                    final=bool(event.get("final")), interrupted=bool(event.get("interrupted")))
    elif kind == "tool":
        state["phase"] = "searching"
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
                    "turn_id": str(event.get("turn_id") or call_id)[:100],
                    "search_details": result["response"]["used_data"],
                }
            except service.VoiceSearchError as exc:
                result["response"] = {"error": str(exc)}
            except (ValidationError, ValueError, TypeError, KeyError):
                result["response"] = {"error": "Invalid search arguments. Ask a short clarification and try again."}
            except Exception:
                # Provider/model text must never contain stack traces, credentials or raw GPS.
                result["response"] = {"error": "Local POI search unavailable. Do not invent places. Please try again."}
            results.append(result)
        state["tool_results"] = results
    elif kind == "used":
        state["phase"] = "answering"
        ids = event.get("call_ids")
        if isinstance(ids, list):
            for call_id in ids[:3]:
                if isinstance(call_id, str) and call_id in state.get("pending", {}):
                    result = state["pending"][call_id]
                    state["latest"] = result
                    turn = state.setdefault("turns", {}).setdefault(result.get("turn_id", call_id), {})
                    turn.setdefault("question", result["query"])
                    turn["result"] = result


def render_live_voice(context: RecommendRequest, on_update: Callable[[dict], None]) -> None:
    state = st.session_state.setdefault("_live_voice", {"ack": None, "pending": {}, "tool_results": []})
    if state.get("session"):
        state.setdefault("history_session", state["session"])
    event = _component(
        configured=service.configured(), ack=state.get("ack"), session=state.get("session"),
        grant=state.get("grant"), tool_results=state.get("tool_results", []),
        key=f"live_voice_chat_{st.session_state.get('_live_voice_epoch', 0)}", default=None,
    )
    if isinstance(event, dict) and event.get("id") != state.get("ack"):
        handle_event(event, state, context)
        on_update(state)
        st.rerun()
