#!/usr/bin/env python3
"""Download per-language Wikipedia lead paragraphs into data/wikipedia_text/.

For each ATLAS language (data/atlas_languages.csv) fetch the plaintext lead
section of N random articles from the MediaWiki API and write one gzipped text
file per edition, plus a runtime manifest (data/wikipedia_text/manifest.csv,
inside the gitignored corpora directory) recording counts, sizes, and sha256
hashes. The committed data/wikipedia_text_manifest.csv is the frozen paper-time
snapshot; it is read, never written, by extra_baselines provenance. Intro-only
extracts are a deliberate API-budget choice: full-article extracts are limited
to one page per request, while exintro extracts batch 20 pages per request.
Random sampling makes every run a fresh snapshot; each run's manifest sha256
pins the snapshot that run produced. Mirrors download_wikipedia.py: stdlib
only, injectable fetcher for tests.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import gzip
import hashlib
import http.client
import json
import signal
import sys
import time
from pathlib import Path

from download_wikipedia import wiki_code, read_atlas_mapping

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
CORPUS_DIR = DATA / "wikipedia_text"

FIELDS = ["glottocode", "atlas_name", "wiki_code", "articles_fetched",
          "bytes_uncompressed", "sha256", "retrieved_at"]
BATCH = 20          # extracts per request (exlimit=max caps at 20 for users)
MAX_BATCHES = 400   # hard stop: 20*400 = 8000 extracts per language
MIN_ARTICLES = 500  # below this the corpus is flagged as thin
REQUEST_DELAY = 0.5
MAX_RETRIES = 5


_CONNECTIONS = {}
WATCHDOG_SECONDS = 60  # hard cap per attempt; a batch normally takes seconds


def _alarm(signum, frame):
    raise TimeoutError("fetch watchdog (%ds) - stale keep-alive connection"
                       % WATCHDOG_SECONDS)


def _install_watchdog():
    """SIGALRM watchdog: a keep-alive connection the server lets go stale can
    block indefinitely in poll (observed on this network; the socket timeout
    alone did not fire through the TLS layer). Only valid in the main
    thread; returns False where signals are unavailable."""
    try:
        signal.signal(signal.SIGALRM, _alarm)
        return True
    except ValueError:
        return False


def _connection(code, timeout=20.0):
    """Cached HTTPS connection per edition. http.client connects lazily on
    first request, so constructing (and unit-testing) the cache is offline."""
    conn = _CONNECTIONS.get(code)
    if conn is None:
        conn = http.client.HTTPSConnection("%s.wikipedia.org" % code,
                                           timeout=timeout)
        _CONNECTIONS[code] = conn
    return conn


def _close_connection(code):
    conn = _CONNECTIONS.pop(code, None)
    if conn is not None:
        conn.close()


def fetch_batch(code, n=BATCH, timeout=20.0):
    """One generator=random API call -> raw JSON text, or None on failure.

    Uses a persistent keep-alive connection per edition (_connection): a
    fresh TCP+TLS connection per request costs ~8 s against the MediaWiki
    API from this network, a held-open connection well under a second.
    Each attempt runs under a SIGALRM watchdog so a stale connection raises
    instead of hanging. Retries with exponential backoff on HTTP 429 or a
    dropped connection; other errors are treated as missing. Sleeps
    REQUEST_DELAY after every attempt to stay polite."""
    path = ("/w/api.php?action=query&format=json&generator=random"
            "&grnnamespace=0&grnlimit=%d"
            "&prop=extracts&explaintext=1&exintro=1&exlimit=max" % n)
    _install_watchdog()
    backoff = 2.0
    for attempt in range(MAX_RETRIES + 1):
        try:
            signal.setitimer(signal.ITIMER_REAL, WATCHDOG_SECONDS)
            conn = _connection(code, timeout)
            conn.request("GET", path,
                         headers={"User-Agent": "grambank-blt-proxy/1.0 (research)"})
            resp = conn.getresponse()
            body = resp.read().decode("utf-8")
            if resp.status == 429 and attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.status != 200:
                _close_connection(code)
                print("  WARNING: fetch failed for %s: HTTP %d"
                      % (code, resp.status), file=sys.stderr)
                return None
            return body
        except Exception as e:  # stale keep-alive etc. -> reconnect, retry
            _close_connection(code)
            if attempt >= MAX_RETRIES:
                print("  WARNING: fetch failed for %s: %s" % (code, e), file=sys.stderr)
                return None
            time.sleep(backoff)
            backoff *= 2
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            time.sleep(REQUEST_DELAY)


def parse_batch(json_text):
    """[(title, extract)] from one API response; empty on bad input."""
    try:
        pages = json.loads(json_text)["query"]["pages"].values()
    except (KeyError, TypeError, ValueError):
        return []
    return [(p.get("title", ""), p.get("extract", ""))
            for p in pages if p.get("extract")]


def build_corpus(mapping_rows, fetch_fn, n_articles, out_dir, retrieved_at,
                 resume=False, manifest_path=None):
    """Per-language corpora from an INJECTABLE fetcher (tests never touch the
    network). fetch_fn(code, n) -> raw JSON text or None. Writes
    <code>.txt.gz per language (articles joined by blank lines), returns
    manifest rows in mapping order. Titles are deduplicated across batches;
    a language below MIN_ARTICLES is flagged as thin but still written.
    With manifest_path, each completed language's row is appended to the
    ledger immediately, so an interrupted run leaves a manifest consistent
    with the .txt.gz files on disk. With resume=True, a language whose
    .txt.gz exists AND has a ledger row is skipped and its row reused."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prev = {}
    if resume and manifest_path and Path(manifest_path).exists():
        with open(manifest_path, newline="") as f:
            prev = {r["wiki_code"]: r for r in csv.DictReader(f)}
    elif manifest_path:
        with open(manifest_path, "w", newline="") as f:   # fresh run: header
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    rows = []
    for r in mapping_rows:
        atlas_name = (r.get("atlas_name") or "").strip()
        if not atlas_name:
            continue
        code = wiki_code(atlas_name)
        out_path = out_dir / ("%s.txt.gz" % code)
        if resume and out_path.exists() and code in prev:
            rows.append(dict(prev[code]))
            continue
        seen, texts = set(), []
        for _ in range(MAX_BATCHES):
            if len(texts) >= n_articles:
                break
            raw = fetch_fn(code, min(BATCH, n_articles - len(texts)))
            if raw is None:
                break
            fresh = [(title, text) for title, text in parse_batch(raw)
                     if title and title not in seen and text]
            seen.update(title for title, _ in fresh)
            texts.extend(text for _, text in fresh)
        blob = "\n\n".join(texts)
        with gzip.open(out_dir / ("%s.txt.gz" % code), "wt", encoding="utf-8",
                       newline="\n") as f:  # pinned so sha256 is platform-proof
            f.write(blob)
        if len(texts) < MIN_ARTICLES:
            print("  WARNING: thin corpus for %s (%d articles)" % (code, len(texts)),
                  file=sys.stderr)
        rows.append({"glottocode": (r.get("glottocode") or "").strip(),
                     "atlas_name": atlas_name, "wiki_code": code,
                     "articles_fetched": len(texts),
                     "bytes_uncompressed": len(blob.encode("utf-8")),
                     "sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
                     "retrieved_at": retrieved_at})
        if manifest_path:
            with open(manifest_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerow(rows[-1])
    return rows


def write_manifest(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Snapshot Wikipedia lead-paragraph corpora")
    ap.add_argument("--articles", type=int, default=2000,
                    help="articles per language edition (default 2000)")
    ap.add_argument("--resume", action="store_true",
                    help="skip languages whose .txt.gz already exists with a "
                         "manifest row, reusing that row")
    ap.add_argument("--modeling-only", action="store_true",
                    help="fetch only the 24 modeling languages (atlas_name in "
                         "data/iso6393.csv); the other editions are never "
                         "consumed by the extra baselines")
    args = ap.parse_args()
    if args.articles > MAX_BATCHES * BATCH:
        print("  WARNING: --articles capped at %d (MAX_BATCHES * BATCH)"
              % (MAX_BATCHES * BATCH), file=sys.stderr)
    mapping = read_atlas_mapping(DATA / "atlas_languages.csv")
    if args.modeling_only:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from experiments import extra_baselines
        keep = set(extra_baselines.iso_codes())          # the 24 modeling langs
        mapping = [r for r in mapping if (r.get("atlas_name") or "").strip() in keep]
    retrieved_at = datetime.date.today().isoformat()
    runtime_manifest = CORPUS_DIR / "manifest.csv"
    rows = build_corpus(mapping, fetch_batch, args.articles, CORPUS_DIR,
                        retrieved_at, resume=args.resume,
                        manifest_path=runtime_manifest)
    write_manifest(rows, runtime_manifest)
    covered = sum(1 for r in rows if r["glottocode"])
    print("Wrote %d corpora (%d with a glottocode) + manifest" % (len(rows), covered))
    return 0


if __name__ == "__main__":
    sys.exit(main())
