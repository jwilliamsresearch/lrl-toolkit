"""Teacher-LLM abstraction for generating and translating conversational data.

**Only local / open-weight teachers are supported by design.** Proprietary hosted
APIs (Claude, OpenAI, Gemini, …) prohibit using their outputs to train other
models in their terms of service, so they are unsuitable as teachers for a
model-*building* toolkit. Providers here run open models you control:

* ``mock``   — deterministic, offline; used in tests and dry runs.
* ``ollama`` — a local Ollama server (recommended; free, no API key). Pull an
  open multilingual model first, e.g. ``ollama pull qwen2.5:7b``.
* ``local``  — a local Hugging Face text-generation model via ``transformers``.

The convdata stage talks only to :class:`BaseTeacher`, never to a specific
backend, so new open providers can be added without touching the stage.
"""

from __future__ import annotations

import abc
import json
import os
import re

from ..utils import get_logger

log = get_logger("lrl.convdata.teacher")

_DEFAULT_MODELS = {
    "ollama": "qwen2.5:7b",
    "local": "HuggingFaceTB/SmolLM2-360M-Instruct",
}
_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# A single completion can only hold so many full instruction-response pairs
# before its output token budget truncates the JSON array mid-object. Large
# requests (n in the thousands) are split into batches this size instead of
# one all-at-once request that would silently come back mostly empty.
_GEN_BATCH_SIZE = 15
_GEN_MAX_TOKENS = 4096


class BaseTeacher(abc.ABC):
    provider: str

    @abc.abstractmethod
    def translate(self, text: str, target_language: str, source_language: str = "English") -> str:
        ...

    @abc.abstractmethod
    def generate_pairs(
        self, n: int, target_language: str, contexts: list[str] | None = None
    ) -> list[dict]:
        """Return a list of {'instruction': ..., 'response': ...} dicts."""
        ...


def _extract_json_array(text: str) -> list:
    """Best-effort extraction of the first JSON array from an LLM response."""
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []


# --------------------------------------------------------------------------- #
# Mock (offline, deterministic)
# --------------------------------------------------------------------------- #
class MockTeacher(BaseTeacher):
    provider = "mock"

    def translate(self, text: str, target_language: str, source_language: str = "English") -> str:
        return f"[{target_language}] " + re.sub(r"\s+", " ", text).strip()

    def generate_pairs(
        self, n: int, target_language: str, contexts: list[str] | None = None
    ) -> list[dict]:
        pairs = []
        for i in range(n):
            topic = ""
            if contexts:
                snippet = contexts[i % len(contexts)]
                topic = " (" + " ".join(snippet.split()[:4]) + ")"
            pairs.append(
                {
                    "instruction": f"[{target_language}] Question {i + 1}{topic}?",
                    "response": f"[{target_language}] Answer {i + 1}.",
                }
            )
        return pairs


# --------------------------------------------------------------------------- #
# Open / local providers
# --------------------------------------------------------------------------- #
class _PromptTeacher(BaseTeacher):
    """Shared prompt building + parsing for chat-completion providers."""

    def __init__(self, model: str | None):
        self.model = model or _DEFAULT_MODELS[self.provider]

    @abc.abstractmethod
    def _chat(self, system: str, user: str, max_tokens: int = 2048) -> str:
        ...

    def translate(self, text: str, target_language: str, source_language: str = "English") -> str:
        system = (
            f"You are a professional translator into {target_language}. "
            "Translate faithfully and return ONLY the translation, no notes."
        )
        return self._chat(system, text, max_tokens=1024).strip()

    def _generate_batch(
        self, batch_n: int, target_language: str, contexts: list[str] | None
    ) -> list[dict]:
        ctx = ""
        if contexts:
            joined = "\n---\n".join(c[:500] for c in contexts)
            ctx = f"\nGround the topics in this sample text:\n{joined}\n"
        system = (
            f"You generate high-quality instruction-tuning data in {target_language}. "
            "Write natural, diverse instructions a real user might ask, with correct, "
            "helpful responses, entirely in the target language."
        )
        user = (
            f"Generate {batch_n} instruction-response pairs.{ctx}\n"
            'Return ONLY a JSON array of objects with keys "instruction" and "response".'
        )
        raw = self._chat(system, user, max_tokens=_GEN_MAX_TOKENS)
        return [
            {"instruction": p["instruction"], "response": p["response"]}
            for p in _extract_json_array(raw)
            if isinstance(p, dict) and p.get("instruction") and p.get("response")
        ]

    def generate_pairs(
        self, n: int, target_language: str, contexts: list[str] | None = None
    ) -> list[dict]:
        """Generate n pairs, batching requests since one completion can only hold
        so many full pairs before its output token budget truncates the JSON
        array mid-object (a single request for a large n silently returns very
        few pairs otherwise)."""
        collected: list[dict] = []
        max_attempts = max(1, -(-n // _GEN_BATCH_SIZE)) * 3  # allow retries for empty batches
        attempt = 0
        call_idx = 0
        while len(collected) < n and attempt < max_attempts:
            attempt += 1
            batch_n = min(_GEN_BATCH_SIZE, n - len(collected))
            batch_ctx = None
            if contexts:
                # Rotate the context window per call so batches ground on
                # different corpus snippets instead of the same first few.
                offset = (call_idx * 5) % len(contexts)
                batch_ctx = (contexts[offset:] + contexts[:offset])[:5]
            call_idx += 1
            batch = self._generate_batch(batch_n, target_language, batch_ctx)
            if not batch:
                log.warning("[teacher] batch %d produced 0 parseable pairs; retrying", attempt)
                continue
            collected.extend(batch)
        if len(collected) < n:
            log.warning(
                "[teacher] only generated %d/%d requested pairs after %d attempts",
                len(collected),
                n,
                attempt,
            )
        return collected[:n]


class OllamaTeacher(_PromptTeacher):
    provider = "ollama"

    def __init__(self, model: str | None):
        super().__init__(model)
        self._host = _OLLAMA_HOST.rstrip("/")

    def _chat(self, system: str, user: str, max_tokens: int = 2048) -> str:
        import requests

        try:
            resp = requests.post(
                f"{self._host}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0.8},
                },
                timeout=600,
            )
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self._host}. Is it running? "
                "Install from https://ollama.com and `ollama serve`."
            ) from exc
        if resp.status_code == 404:
            raise RuntimeError(
                f"Ollama model '{self.model}' not found. Pull it: `ollama pull {self.model}`."
            )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")


class LocalTeacher(_PromptTeacher):
    provider = "local"

    def __init__(self, model: str | None):
        super().__init__(model)
        from transformers import pipeline

        self._pipe = pipeline("text-generation", model=self.model)

    def _chat(self, system: str, user: str, max_tokens: int = 2048) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        out = self._pipe(messages, max_new_tokens=max_tokens, do_sample=True, temperature=0.8)
        generated = out[0]["generated_text"]
        # transformers returns the full message list for chat inputs.
        if isinstance(generated, list):
            return generated[-1]["content"]
        return str(generated)


def get_teacher(provider: str, model: str | None = None) -> BaseTeacher:
    provider = provider.lower()
    if provider == "mock":
        return MockTeacher()
    if provider == "ollama":
        return OllamaTeacher(model)
    if provider == "local":
        return LocalTeacher(model)
    if provider in ("claude", "anthropic", "openai", "gpt", "gemini"):
        raise ValueError(
            f"Provider '{provider}' is a proprietary hosted API whose terms prohibit using "
            "outputs to train models. Use a local/open teacher: 'ollama' or 'local'."
        )
    raise ValueError(f"Unknown teacher provider '{provider}'. Use mock/ollama/local.")
