"""FLORES+ chrF: English->target translation quality (openlanguagedata/flores_plus).

A *generative* probe: the model is few-shot prompted to translate English FLORES
sentences into the target language, and the output is scored with chrF (sacrebleu)
against the human reference. Uses the `dev` split for few-shot exemplars and
`devtest` for scoring so the two are disjoint.

FLORES+ is **gated** — coverage reports `gated=True` when no HF_TOKEN is present,
rather than silently skipping. It is also the benchmark whose data the toolkit
deliberately keeps out of *training* (see the FLORES-in-training safeguards), so it
is a clean held-out test here.

This probes translation, which is *not* what these models are primarily built for,
so scores will be modest; the base-vs-adapted delta is the meaningful part.
"""

from __future__ import annotations

from ...registry import LanguageProfile
from .base import Benchmark, Coverage, MetricDirection, ModelBundle, Score

_REPO = "openlanguagedata/flores_plus"


class FloresChrfBenchmark(Benchmark):
    name = "flores"
    direction = MetricDirection.higher_better
    caveats = (
        "Probes translation, which is not these models' primary task; expect modest "
        "absolute chrF. Read the base-vs-adapted delta.",
    )

    def _code(self, lang: LanguageProfile) -> str:
        return lang.nllb_code or lang.lang_script_code()

    def coverage(self, lang: LanguageProfile, *, has_token: bool) -> Coverage:
        code = self._code(lang)
        if not has_token:
            return Coverage(False, "FLORES+ is gated; set HF_TOKEN and accept its terms.",
                            code, gated=True)
        from .base import dataset_configs

        configs = dataset_configs(_REPO, token=None)
        if configs is None:
            return Coverage(False, "Could not list FLORES+ configs (offline or no access).",
                            code, gated=True)
        if code in configs and "eng_Latn" in configs:
            return Coverage(True, f"FLORES+ covers {code}.", code)
        return Coverage(False, f"{lang.display_name} ({code}) is not in FLORES+.", code)

    def score(self, bundle: ModelBundle, lang: LanguageProfile, *, limit: int) -> Score:
        import os

        import torch
        from datasets import load_dataset

        try:
            import sacrebleu
        except ImportError:
            return Score(value=None, note="sacrebleu missing; pip install 'lrl-toolkit[eval]'")

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        code = self._code(lang)
        model, tok = bundle.model, bundle.tokenizer
        lang_name = lang.display_name

        # Few-shot exemplars from dev; align by 'id'.
        eng_dev = {r["id"]: r["text"] for r in load_dataset(_REPO, "eng_Latn", split="dev",
                                                            streaming=True, token=token)}
        tgt_dev = {r["id"]: r["text"] for r in load_dataset(_REPO, code, split="dev",
                                                            streaming=True, token=token)}
        shot_ids = [i for i in eng_dev if i in tgt_dev][:3]
        shots = "".join(
            f"English: {eng_dev[i]}\n{lang_name}: {tgt_dev[i]}\n\n" for i in shot_ids
        )

        eng_test = {r["id"]: r["text"] for r in load_dataset(_REPO, "eng_Latn", split="devtest",
                                                             streaming=True, token=token)}
        tgt_test = {r["id"]: r["text"] for r in load_dataset(_REPO, code, split="devtest",
                                                             streaming=True, token=token)}
        pair_ids = [i for i in eng_test if i in tgt_test][:limit]
        if not pair_ids:
            return Score(value=None, note="no aligned FLORES pairs")

        hyps, refs = [], []
        device = next(model.parameters()).device
        for i in pair_ids:
            prompt = f"{shots}English: {eng_test[i]}\n{lang_name}:"
            ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=128, do_sample=False)
            gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            hyps.append(gen.split("\n")[0].strip())
            refs.append(tgt_test[i])

        chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
        return Score(value=chrf, n=len(hyps), note="chrF (few-shot En->target translation)")
