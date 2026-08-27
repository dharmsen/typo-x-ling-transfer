#!/usr/bin/env python3
"""Download per-language Wikipedia article counts into ./data/wikipedia_articles.csv.

For each ATLAS language (data/atlas_languages.csv) fetch the article count of
its Wikipedia edition from the MediaWiki siteinfo API and write a COMMITTED
snapshot. Article counts drift daily, so the snapshot is version-controlled for
reproducibility (re-run to refresh). Mirrors download_wals.py: stdlib only.

atlas_name is already the Wikipedia language-edition code for every ATLAS
language, with one override: fil -> tl (there is no Filipino Wikipedia, so
Filipino maps to the Tagalog edition).
"""
from __future__ import annotations

import csv
import datetime
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"

# atlas_name -> Wikipedia edition code; identity except where ATLAS and
# Wikipedia disagree on the language code.
WIKI_CODE_OVERRIDES = {"fil": "tl"}  # Filipino -> Tagalog Wikipedia (no 'fil' edition)

FIELDS = ["glottocode", "atlas_name", "wiki_code", "articles", "retrieved_at"]

# MediaWiki rate-limits rapid bursts (~38 editions here); be polite and back off
# on HTTP 429 so the committed snapshot reflects real counts, not transients.
# REQUEST_DELAY keeps the whole run under the per-IP request budget; the 429
# backoff recovers the occasional straggler that still gets throttled.
REQUEST_DELAY = 1.0
MAX_RETRIES = 5


def wiki_code(atlas_name):
    """Wikipedia edition code for an ATLAS language code (applies overrides)."""
    return WIKI_CODE_OVERRIDES.get(atlas_name, atlas_name)


def parse_article_count(json_text):
    """Read query.statistics.articles from a siteinfo response, or None."""
    try:
        return int(json.loads(json_text)["query"]["statistics"]["articles"])
    except (KeyError, TypeError, ValueError):
        return None


def fetch_article_count(code, timeout=20.0):
    """Live article count for one Wikipedia edition, or None on failure.

    Retries with exponential backoff on HTTP 429 (rate limit); other network or
    HTTP errors are treated as missing. Sleeps REQUEST_DELAY between requests
    to stay polite across the ~38 editions.
    """
    url = ("https://%s.wikipedia.org/w/api.php"
           "?action=query&meta=siteinfo&siprop=statistics&format=json" % code)
    req = urllib.request.Request(
        url, headers={"User-Agent": "grambank-blt-proxy/1.0 (research)"})
    backoff = 2.0
    for attempt in range(MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return parse_article_count(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2
                continue
            print("  WARNING: fetch failed for %s: %s" % (code, e), file=sys.stderr)
            return None
        except Exception as e:  # network/HTTP errors -> treat as missing
            print("  WARNING: fetch failed for %s: %s" % (code, e), file=sys.stderr)
            return None
        finally:
            time.sleep(REQUEST_DELAY)


def build_rows(mapping_rows, fetch_fn, retrieved_at):
    """Snapshot rows from atlas mapping rows using an INJECTABLE fetcher.

    mapping_rows: iterable of dicts with 'atlas_name' and 'glottocode'.
    fetch_fn: callable(code) -> article count (int) or None. Injected so tests
    never touch the network.
    """
    rows = []
    for r in mapping_rows:
        atlas_name = (r.get("atlas_name") or "").strip()
        if not atlas_name:
            continue
        code = wiki_code(atlas_name)
        count = fetch_fn(code)
        if count is None:
            print("  WARNING: no article count for %s (wiki=%s)" % (atlas_name, code),
                  file=sys.stderr)
            count = 0
        rows.append({"glottocode": (r.get("glottocode") or "").strip(),
                     "atlas_name": atlas_name, "wiki_code": code,
                     "articles": count, "retrieved_at": retrieved_at})
    return rows


def write_snapshot(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_atlas_mapping(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    mapping = read_atlas_mapping(DATA / "atlas_languages.csv")
    retrieved_at = datetime.date.today().isoformat()
    rows = build_rows(mapping, fetch_article_count, retrieved_at)
    out = DATA / "wikipedia_articles.csv"
    write_snapshot(rows, out)
    covered = sum(1 for r in rows if r["glottocode"])
    print("Wrote %d rows (%d with a glottocode) to %s" % (len(rows), covered, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
