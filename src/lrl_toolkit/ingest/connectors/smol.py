"""SMOL / GATITOS connector (google/smol).

Google's SMOL provides professionally translated data for 100+ heavily
under-resourced languages: GATITOS (token/term translations), SmolSent
(sentences), and SmolDoc (documents). Configs are named
``<variant>__<src>_<tgt>`` (e.g. ``gatitos__en_dje``). Not every language is
present; the connector discovers the right config for the requested language.
"""

from __future__ import annotations

from collections.abc import Iterator

from ...corpus import Document
from ...registry import SourceHint
from ..base import BaseConnector, require


class SmolConnector(BaseConnector):
    name = "smol"
    kind = "parallel"

    def _resolve_config(self, datasets, variant: str, src: str, tgt: str) -> str:
        exact = f"{variant}__{src}_{tgt}"
        configs = datasets.get_dataset_config_names("google/smol")
        if exact in configs:
            return exact
        # Fall back to any config for this variant whose target matches the lang
        # (handles script suffixes like 'iu-Latn').
        for c in configs:
            if c.startswith(f"{variant}__{src}_") and c.split("_")[-1].startswith(tgt):
                return c
        raise ValueError(
            f"SMOL has no '{variant}' config for {src}->{tgt}. "
            f"Example available configs: {[c for c in configs if c.startswith(variant)][:5]}"
        )

    def iter_documents(
        self, hint: SourceHint, *, max_docs: int | None = None, token: str | None = None
    ) -> Iterator[Document]:
        datasets = require("datasets")
        variant = hint.params.get("variant", "gatitos")  # gatitos | smolsent | smoldoc
        src = hint.params.get("src", "en")
        tgt = hint.params.get("tgt")
        if not tgt:
            raise ValueError("smol source needs params.tgt (the target language, e.g. 'dje').")
        config = self._resolve_config(datasets, variant, src, tgt)

        ds = datasets.load_dataset(
            "google/smol", config, split="train", streaming=True, token=token
        )
        n = 0
        for row in ds:
            target = self._target_text(row)
            if not target:
                continue
            yield Document(
                text=target,
                source=self.name,
                meta={"variant": variant, "sl": row.get("sl"), "tl": row.get("tl"),
                      "translation": {src: row.get("src"), tgt: target}},
            )
            n += 1
            if max_docs is not None and n >= max_docs:
                return

    @staticmethod
    def _target_text(row: dict) -> str:
        if row.get("trg"):
            return row["trg"]
        trgs = row.get("trgs")
        if isinstance(trgs, list) and trgs:
            return trgs[0]
        if isinstance(trgs, str) and trgs:
            return trgs
        return ""
