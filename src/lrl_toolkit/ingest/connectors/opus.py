"""OPUS connector — parallel corpora via the OPUS API.

OPUS aggregates dozens of parallel corpora (OpenSubtitles, Tatoeba, bible-uedin,
CCAligned, wikimedia, and NLLB itself). The API lists downloadable Moses-format
bitexts for a language pair; we download and stream the sentence pairs.

Emits :class:`Document`s whose ``text`` is the target-language side (so the data
can also feed monolingual target text) with the aligned pair in
``meta['translation']``. Set ``params.corpus='NLLB'`` to pull NLLB specifically.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator

from ...corpus import Document
from ...registry import SourceHint
from ..base import BaseConnector, require

_API = "https://opus.nlpl.eu/opusapi/"
_UA = {"User-Agent": "lrl-toolkit/0.1 (+https://github.com/lrl-toolkit/lrl-toolkit)"}


class OpusConnector(BaseConnector):
    name = "opus"
    kind = "parallel"

    def _list_corpora(self, requests, src: str, tgt: str) -> list[dict]:
        params = {"source": src, "target": tgt, "preprocessing": "moses", "version": "latest"}
        r = requests.get(_API, params=params, headers=_UA, timeout=60)
        r.raise_for_status()
        return r.json().get("corpora", [])

    def iter_documents(
        self, hint: SourceHint, *, max_docs: int | None = None, token: str | None = None
    ) -> Iterator[Document]:
        requests = require("requests")
        src = hint.params.get("src")
        tgt = hint.params.get("tgt", "en")
        if not src:
            raise ValueError("opus source needs params.src (e.g. 'cy'); tgt defaults to 'en'.")
        wanted = set(hint.params.get("corpora", [])) or None
        one = hint.params.get("corpus")
        if one:
            wanted = {one}

        entries = self._list_corpora(requests, src, tgt)
        # OPUS may order the pair either way; the API normalizes, but the moses
        # zip contains files named <corpus>.<pair>.<lang>.
        n = 0
        for entry in entries:
            corpus = entry.get("corpus")
            if wanted is not None and corpus not in wanted:
                continue
            url = entry.get("url")
            if not url:
                continue
            for doc in self._stream_moses_zip(requests, url, corpus, src, tgt):
                yield doc
                n += 1
                if max_docs is not None and n >= max_docs:
                    return

    def _stream_moses_zip(
        self, requests, url: str, corpus: str, src: str, tgt: str
    ) -> Iterator[Document]:
        resp = requests.get(url, headers=_UA, timeout=300)
        if resp.status_code != 200:
            return
        try:
            zf = zipfile.ZipFile(io.BytesIO(resp.content))
        except zipfile.BadZipFile:
            return
        names = zf.namelist()
        src_name = next((n for n in names if n.endswith(f".{src}")), None)
        tgt_name = next((n for n in names if n.endswith(f".{tgt}")), None)
        if not (src_name and tgt_name):
            return
        with zf.open(src_name) as sf, zf.open(tgt_name) as tf:
            for s_line, t_line in zip(
                io.TextIOWrapper(sf, encoding="utf-8"),
                io.TextIOWrapper(tf, encoding="utf-8"),
                strict=False,
            ):
                s, t = s_line.strip(), t_line.strip()
                if not s or not t:
                    continue
                yield Document(
                    text=s,  # source == target language (the LRL)
                    source=self.name,
                    meta={"corpus": corpus, "translation": {src: s, tgt: t}},
                )
