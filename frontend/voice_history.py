"""Merge live speech and verified search results into the regular session chat history."""

from __future__ import annotations

from collections.abc import MutableMapping

from backend.generation.recommender import RecommendRequest


def sync_voice_history(store: MutableMapping, voice: dict, start: str, *, max_history: int = 12) -> None:
    """Upsert by voice session/turn, keeping the iframe mounted and avoiding another LLM call."""
    turns = voice.get("turns", {})
    if not turns:
        return
    convo_id = voice.get("convo_id", store.get("active_convo"))
    convo = next((item for item in store["convos"] if item["id"] == convo_id), None)
    if convo is None:
        convo = {"id": store["next_convo_id"], "title": "Голосовой разговор", "entries": []}
        store["next_convo_id"] += 1
        store["convos"].append(convo)
    voice["convo_id"] = convo["id"]
    store["active_convo"] = convo["id"]
    context = RecommendRequest.model_validate_json(voice["context"])
    for turn_id, turn in turns.items():
        identity = f"{voice['history_session']}:{turn_id}"
        entry = next((item for item in convo["entries"] if item.get("voice_turn") == identity), None)
        if entry is None:
            entry = {
                "voice_turn": identity, "via": "live_voice", "start": start,
                "query": "", "answer_language": context.lang,
                "req_json": context.model_copy(update={"query": "Голосовой разговор"}).model_dump_json(),
            }
            convo["entries"].append(entry)
        if turn.get("question"):
            entry["query"] = turn["question"]
            req = RecommendRequest.model_validate_json(entry["req_json"])
            entry["req_json"] = req.model_copy(update={"query": turn["question"]}).model_dump_json()
        entry["assistant_text"] = turn.get("answer", "")
        entry["voice_final"] = turn.get("final", False)
        entry["voice_interrupted"] = turn.get("interrupted", False)
        result = turn.get("result")
        if result:
            entry["search_query"] = RecommendRequest.model_validate_json(result["req_json"]).query
            entry["voice_search"] = result.get("search_details", {})
        if result and entry.get("response_json") != result["response_json"]:
            entry["response_json"] = result["response_json"]
            entry["presentation"] = {
                "language": context.lang, "requested_language": context.lang, "labels": {},
                "response_json": result["response_json"], "note": "",
            }
            # A new chart key reapplies fitted bounds even after a user pans the previous map.
            store["map_view_revision"] = store.get("map_view_revision", 0) + 1
        if len(convo["entries"]) == 1 and entry["query"]:
            title = " ".join(entry["query"].split())
            convo["title"] = title[:37] + "…" if len(title) > 38 else title
    convo["entries"] = convo["entries"][-max_history:]
    kept = {entry.get("voice_turn") for entry in convo["entries"]}
    voice["turns"] = {key: turn for key, turn in turns.items()
                      if f"{voice['history_session']}:{key}" in kept}


def latest_result_entry(entries: list[dict]) -> dict | None:
    """Small talk and interrupted speech must not clear the last verified map."""
    return next((entry for entry in reversed(entries) if entry.get("response_json")), None)
