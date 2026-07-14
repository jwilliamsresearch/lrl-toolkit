"""Raw Common Crawl connector.

Queries the Common Crawl CDX index for URL patterns, then fetches individual
WARC records by HTTP byte-range from ``data.commoncrawl.org`` and extracts main
text. This is the "process the raw crawl yourself" path; for most languages the
curated derivatives (OSCAR/CulturaX/MADLAD/Glot500) are easier, but raw CC lets
you target specific domains a derivative may have dropped.

The public CDX index is frequently slow (503/504); queries retry with backoff.
Because raw CC is not language-filtered, pair this connector with the clean
stage's language-ID filter.
"""

from __future__ import annotations

import gzip
import io
import json
import time
from collections.abc import Iterator

from ...corpus import Document
from ...registry import SourceHint
from ..base import BaseConnector, require

_UA = {"User-Agent": "lrl-toolkit/0.1 (+https://github.com/lrl-toolkit/lrl-toolkit)"}
_COLLINFO = "https://index.commoncrawl.org/collinfo.json"
_DATA = "https://data.commoncrawl.org/"


class CommonCrawlConnector(BaseConnector):
    name = "commoncrawl"
    kind = "mono"

    def _newest_crawl(self, requests) -> str:
        info = requests.get(_COLLINFO, headers=_UA, timeout=30).json()
        return info[0]["id"]

    def _cdx_query(self, requests, crawl: str, url_pattern: str, limit: int) -> list[dict]:
        api = f"https://index.commoncrawl.org/{crawl}-index"
        params = {
            "url": url_pattern,
            "output": "json",
            "limit": limit,
            "filter": ["=status:200", "=mime:text/html"],
        }
        last = None
        for attempt in range(5):
            r = requests.get(api, params=params, headers=_UA, timeout=90)
            if r.status_code == 200 and r.text.strip().startswith("{"):
                return [json.loads(line) for line in r.text.splitlines() if line.strip()]
            last = r.status_code
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(
            f"Common Crawl CDX index unavailable for {crawl} (last status {last}). "
            "The public index is often overloaded; try again later."
        )

    def _fetch_text(self, requests, extractor, rec: dict) -> str | None:
        off, ln = int(rec["offset"]), int(rec["length"])
        headers = dict(_UA)
        headers["Range"] = f"bytes={off}-{off + ln - 1}"
        resp = requests.get(_DATA + rec["filename"], headers=headers, timeout=180)
        if resp.status_code not in (200, 206):
            return None
        warcio = require("warcio")
        stream = io.BytesIO(resp.content)
        try:
            record = next(warcio.ArchiveIterator(stream))
        except (StopIteration, gzip.BadGzipFile):
            return None
        html = record.content_stream().read()
        if extractor is not None:
            return extractor.extract(html) or None
        return html.decode("utf-8", "replace")

    def iter_documents(
        self, hint: SourceHint, *, max_docs: int | None = None, token: str | None = None
    ) -> Iterator[Document]:
        requests = require("requests")
        try:
            extractor = require("trafilatura")
        except Exception:
            extractor = None  # fall back to raw HTML if trafilatura absent

        patterns = hint.params.get("url_patterns") or (
            [hint.params["url_pattern"]] if hint.params.get("url_pattern") else []
        )
        if not patterns:
            raise ValueError(
                "commoncrawl source needs params.url_pattern or params.url_patterns "
                "(e.g. 'cy.wikipedia.org/*')."
            )
        crawl = hint.params.get("crawl") or self._newest_crawl(requests)
        per_pattern = max_docs or hint.params.get("limit", 100)

        n = 0
        for pattern in patterns:
            for rec in self._cdx_query(requests, crawl, pattern, per_pattern):
                text = self._fetch_text(requests, extractor, rec)
                if not text:
                    continue
                yield Document(
                    text=text,
                    source=self.name,
                    meta={"url": rec.get("url"), "crawl": crawl},
                )
                n += 1
                if max_docs is not None and n >= max_docs:
                    return
