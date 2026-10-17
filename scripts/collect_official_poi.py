"""Import reviewed POIs from the official Visit Astana tourism portal.

Download: python scripts/collect_official_poi.py --download
Preview:  python scripts/collect_official_poi.py
Apply:    python scripts/collect_official_poi.py --apply

The manifest is a reviewed allowlist, not generated venue data. Names and map
markers are extracted from the source HTML; absent optional fields stay null.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.retrieval.geo import haversine_m
from backend.retrieval.schema import POI

CACHE = ROOT / "data/raw/visit_astana"
MANIFEST = ROOT / "data/official_poi_manifest.json"


class SourceText(HTMLParser):
    """Read named divs without running any scripts from the source page."""
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.active = []
        self.fields = {}

    def handle_starttag(self, tag, attrs):
        if tag == "div":
            self.depth += 1
            classes = dict(attrs).get("class", "").split()
            for key in ("teatr_ban_title", "teatr_ban_info", "center_text--history"):
                if key in classes:
                    self.active.append((key, self.depth))
                    self.fields[key] = []
        if tag in {"br", "p", "li"}:
            self.handle_data(" ")

    def handle_endtag(self, tag):
        if tag == "div":
            self.active = [(k, d) for k, d in self.active if d != self.depth]
            self.depth -= 1
        self.handle_data(" ")

    def handle_data(self, data):
        for key, _ in self.active:
            self.fields[key].append(data)

    def text(self, key):
        return " ".join("".join(self.fields.get(key, [])).replace("\u200b", "").split())


def parse_page(html: str, entry: dict, retrieved_at: str) -> tuple[dict, dict]:
    parser = SourceText()
    parser.feed(html)
    name = parser.text("teatr_ban_title")
    address = parser.text("teatr_ban_info")
    markers = set(re.findall(
        r"new\s+ymaps\.Placemark\s*\(\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", html
    ))
    if not name or not address or len(markers) != 1:
        raise ValueError(f"Missing or ambiguous name/address/map marker: {entry['url']}")
    lat, lon = map(float, markers.pop())
    if not (51.03 <= lat <= 51.25 and 71.25 <= lon <= 71.65):
        raise ValueError(f"Marker outside Astana collection area: {entry['url']}")
    history = parser.text("center_text--history")
    hours = entry.get("opening_hours")
    if hours and entry["hours_evidence"] not in history:
        raise ValueError(f"Opening-hours evidence changed: {entry['url']}")
    phone = entry.get("phone")
    if phone and phone not in history:
        raise ValueError(f"Phone evidence changed: {entry['url']}")
    note = entry.get("visit_note")
    if note and note not in history:
        raise ValueError(f"Visit-condition evidence changed: {entry['url']}")
    slug = urlparse(entry["url"]).path.strip("/").split("/")[-1]
    poi = POI(
        id=f"visitastana_{slug}", name=name, lat=lat, lon=lon,
        category=entry["category"], tags=[], description=note,
        opening_hours=hours, contacts={"phone": phone, "website": None},
        source=entry["url"], collected_at=retrieved_at[:10],
    ).model_dump()
    evidence = {
        "url": entry["url"], "retrieved_at": retrieved_at,
        "html_sha256": hashlib.sha256(html.encode()).hexdigest(),
        "name": name, "address": address,
        "coordinates": {"lat": lat, "lon": lon, "locator": "ymaps.Placemark"},
        "hours_evidence": entry.get("hours_evidence"), "phone_evidence": phone,
        "visit_note_evidence": note,
        "category_note": "Internal classification by venue type; not an OSM tag.",
    }
    return poi, evidence


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    existing_path = ROOT / "data/poi.json"
    existing = json.loads(existing_path.read_text())
    sources_path = ROOT / "data/poi_sources.json"
    sources = json.loads(sources_path.read_text()) if sources_path.exists() else {}
    additions = []
    CACHE.mkdir(parents=True, exist_ok=True)
    for entry in manifest:
        url = entry["url"]
        if urlparse(url).netloc != "visitastana.kz" or not url.startswith("https://"):
            raise ValueError("Only the reviewed official portal is allowed")
        slug = urlparse(url).path.strip("/").split("/")[-1]
        cache = CACHE / f"{slug}.json"
        if args.download:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            page = {"retrieved_at": datetime.now(timezone.utc).isoformat(), "html": response.text}
            cache.write_text(json.dumps(page, ensure_ascii=False))
        page = json.loads(cache.read_text())
        try:
            poi, evidence = parse_page(page["html"], entry, page["retrieved_at"])
        except ValueError as error:
            if args.apply:
                raise
            print(f"REJECTED: {error}")
            continue
        sources[poi["id"]] = evidence
        additions.append(poi)
        neighbors = sorted(
            (haversine_m(poi["lat"], poi["lon"], p["lat"], p["lon"]), p["name"])
            for p in existing if p["id"] != poi["id"]
        )[:3]
        print(poi["name"], (poi["lat"], poi["lon"]), "near:", [(round(d), n) for d, n in neighbors])
    if args.apply:
        official_path = ROOT / "data/official_poi.json"
        combined = {p["id"]: p for p in existing}
        combined.update({p["id"]: p for p in additions})
        official_path.write_text(json.dumps(additions, ensure_ascii=False, indent=1) + "\n")
        existing_path.write_text(json.dumps(sorted(combined.values(), key=lambda p: p["id"]), ensure_ascii=False, indent=1) + "\n")
        sources_path.write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n")
        print(f"Applied {len(additions)} reviewed records; total {len(combined)}")


if __name__ == "__main__":
    main()
