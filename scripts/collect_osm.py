"""Collect POIs for Astana from OpenStreetMap via the Overpass API.

Usage:
    python scripts/collect_osm.py                 # fetch + normalize -> data/poi.json
    python scripts/collect_osm.py --from-raw      # re-normalize the cached raw dump only

Steps: query Overpass (nodes + ways/relations with center) inside the Astana bbox,
cache the raw response in data/raw/, normalize to the POI contract (CLAUDE.md),
deduplicate, and write data/poi.json sorted by id for stable diffs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.retrieval.geo import haversine_m  # noqa: E402
from backend.retrieval.schema import POI  # noqa: E402

RAW_PATH = ROOT / "data" / "raw" / "overpass_astana.json"
OUT_PATH = ROOT / "data" / "poi.json"

# south, west, north, east: city of Astana incl. Left Bank, Right Bank and Expo area.
ASTANA_BBOX = (51.03, 71.25, 51.25, 71.65)

OVERPASS_URLS = [
    os.getenv("OVERPASS_URL", "https://overpass-api.de/api/interpreter"),
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

# OSM key -> {value: our category}. Only named objects are collected.
CATEGORY_MAP: dict[str, dict[str, str]] = {
    "amenity": {
        "cafe": "cafe", "restaurant": "restaurant", "fast_food": "fast_food",
        "bar": "bar", "pub": "bar", "ice_cream": "cafe", "theatre": "theatre",
        "cinema": "cinema", "library": "library", "arts_centre": "arts_centre",
        "place_of_worship": "place_of_worship", "fountain": "attraction",
        "exhibition_centre": "exhibition", "planetarium": "museum",
    },
    "tourism": {
        "museum": "museum", "gallery": "gallery", "attraction": "attraction",
        "viewpoint": "viewpoint", "zoo": "zoo", "theme_park": "theme_park",
        "artwork": "artwork", "aquarium": "attraction",
    },
    "leisure": {
        "park": "park", "garden": "park", "water_park": "theme_park",
        "ice_rink": "sport", "sports_centre": "sport", "fitness_centre": "sport",
        "stadium": "stadium",
    },
    "shop": {"mall": "mall", "books": "bookstore"},
    "historic": {"monument": "monument", "memorial": "monument"},
}

# Generic tags that are only worth collecting for notable objects (with a `wikidata` link):
# e.g. the Palace of Peace and Reconciliation is tagged just amenity=community_centre.
NOTABLE_ONLY_MAP: dict[str, dict[str, str]] = {
    "amenity": {"community_centre": "attraction"},
}

# Russian names for landmarks that only have a Kazakh `name` in OSM.
NAME_OVERRIDES: dict[str, str] = {
    "way/166197368": "Дворец мира и согласия",
}

FOOD_CATEGORIES = {"cafe", "restaurant", "fast_food", "bar"}

# Mapping errors where the street address was put into `name`.
ADDRESS_NAME_RE = re.compile(
    r"^(проспект|улица|ул\.|пр\.|шоссе|переулок)\s|\s(даңғылы|көшесі)$", re.IGNORECASE
)


def build_query(bbox: tuple[float, float, float, float], timeout: int = 180) -> str:
    s, w, n, e = bbox
    selectors = []
    for key, values in CATEGORY_MAP.items():
        regex = "|".join(sorted(values))
        selectors.append(f'nwr["{key}"~"^({regex})$"]["name"]({s},{w},{n},{e});')
    for key, values in NOTABLE_ONLY_MAP.items():
        regex = "|".join(sorted(values))
        selectors.append(f'nwr["{key}"~"^({regex})$"]["name"]["wikidata"]({s},{w},{n},{e});')
    body = "\n  ".join(selectors)
    return f"[out:json][timeout:{timeout}];\n(\n  {body}\n);\nout center tags;"


def fetch_overpass(query: str, retries: int = 3) -> dict:
    headers = {"User-Agent": "astana-walks-hackathon/0.1 (educational project)"}
    last_err: Exception | None = None
    for url in OVERPASS_URLS:
        for attempt in range(retries):
            try:
                print(f"[overpass] {url} (attempt {attempt + 1})", file=sys.stderr)
                r = requests.post(url, data={"data": query}, headers=headers, timeout=240)
                if r.status_code in (429, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError) as err:
                last_err = err
                time.sleep(10 * 2 ** attempt)  # servers are often at capacity
    raise RuntimeError(f"All Overpass endpoints failed: {last_err}")


def _category(tags: dict[str, str]) -> str | None:
    for key, values in CATEGORY_MAP.items():
        if tags.get(key) in values:
            return values[tags[key]]
    if tags.get("wikidata"):
        for key, values in NOTABLE_ONLY_MAP.items():
            if tags.get(key) in values:
                return values[tags[key]]
    return None


def _name(tags: dict[str, str]) -> str | None:
    for key in ("name:ru", "name", "name:kk", "name:en"):
        if tags.get(key, "").strip():
            return tags[key].strip()
    return None


def _derived_tags(tags: dict[str, str], category: str) -> list[str]:
    """Map raw OSM tags to the short, search-friendly tags used by retrieval."""
    out: set[str] = set()
    for key in ("amenity", "tourism", "leisure", "shop", "historic"):
        if key in tags:
            out.add(tags[key])
    for c in re.split(r"[;,]", tags.get("cuisine", "")):
        if c.strip():
            out.add(f"cuisine_{c.strip().lower()}")
    if category == "cafe" and tags.get("amenity") == "cafe":
        out.add("coffee")
    if tags.get("internet_access") in ("wlan", "yes", "wifi"):
        out.add("wifi")
    if tags.get("outdoor_seating") == "yes":
        out.add("terrace")
    if tags.get("wheelchair") in ("yes", "designated"):
        out.add("wheelchair")
    elif tags.get("wheelchair") == "limited":
        out.add("wheelchair_limited")
    if tags.get("takeaway") in ("yes", "only"):
        out.add("takeaway")
    if tags.get("fee") == "no":
        out.add("free")
    if tags.get("diet:vegetarian") in ("yes", "only"):
        out.add("vegetarian")
    if tags.get("diet:halal") in ("yes", "only"):
        out.add("halal")
    if tags.get("opening_hours") == "24/7":
        out.add("24_7")
    if tags.get("indoor") == "yes" or category in {"museum", "mall", "theatre", "cinema", "library"}:
        out.add("indoor")
    if category in {"park", "viewpoint", "monument", "artwork"}:
        out.add("outdoor")
    if category in FOOD_CATEGORIES:
        out.add("food")
    if category in {"museum", "gallery", "theatre", "arts_centre", "monument", "library"}:
        out.add("culture")
    if tags.get("religion"):
        out.add(tags["religion"])
    if tags.get("wikidata") or tags.get("wikipedia"):
        out.add("landmark")
    if category in {"stadium", "exhibition"}:
        out.add("events")
    return sorted(out)


def normalize(element: dict, collected_at: str) -> POI | None:
    tags = element.get("tags", {})
    name = NAME_OVERRIDES.get(f"{element['type']}/{element['id']}") or _name(tags)
    category = _category(tags)
    if not name or not category or ADDRESS_NAME_RE.search(name):
        return None
    if "lat" in element:
        lat, lon = element["lat"], element["lon"]
    elif "center" in element:
        lat, lon = element["center"]["lat"], element["center"]["lon"]
    else:
        return None
    description = tags.get("description:ru") or tags.get("description") or None
    phone = tags.get("phone") or tags.get("contact:phone")
    website = tags.get("website") or tags.get("contact:website") or tags.get("contact:instagram")
    return POI(
        id=f"osm_{element['type']}_{element['id']}",
        name=name,
        lat=round(lat, 6),
        lon=round(lon, 6),
        description=description,
        category=category,
        tags=_derived_tags(tags, category),
        opening_hours=(tags.get("opening_hours") or "").strip() or None,
        contacts={"phone": phone, "website": website},
        rating=None,
        source="osm",
        collected_at=collected_at,
    )


def _richness(p: POI) -> int:
    return sum([
        p.description is not None, p.opening_hours is not None,
        bool(p.contacts and (p.contacts.phone or p.contacts.website)), len(p.tags),
    ])


def deduplicate(pois: list[POI], radius_m: float = 60) -> list[POI]:
    """Drop near-duplicates: same name (case-insensitive) within `radius_m`.

    OSM often has both a node and a building way for the same venue; keep the richer one.
    """
    by_name: dict[str, list[POI]] = {}
    for p in pois:
        by_name.setdefault(p.name.lower(), []).append(p)
    result: list[POI] = []
    for group in by_name.values():
        group.sort(key=_richness, reverse=True)
        kept: list[POI] = []
        for p in group:
            if all(haversine_m(p.lat, p.lon, k.lat, k.lon) > radius_m for k in kept):
                kept.append(p)
        result.extend(kept)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-raw", action="store_true", help="skip download, use cached raw dump")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    if args.from_raw:
        raw = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    else:
        raw = fetch_overpass(build_query(ASTANA_BBOX))
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        RAW_PATH.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    collected_at = raw.get("osm3s", {}).get("timestamp_osm_base", "")[:10] or date.today().isoformat()
    elements = raw.get("elements", [])
    pois = [p for p in (normalize(e, collected_at) for e in elements) if p]
    pois = sorted(deduplicate(pois), key=lambda p: p.id)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps([p.model_dump() for p in pois], ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )

    counts: dict[str, int] = {}
    for p in pois:
        counts[p.category] = counts.get(p.category, 0) + 1
    with_hours = sum(p.opening_hours is not None for p in pois)
    print(f"elements={len(elements)} pois={len(pois)} with_opening_hours={with_hours}")
    for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<18} {n}")
    print(f"-> {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
