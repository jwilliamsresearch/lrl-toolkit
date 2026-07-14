# Contributing to lrl-toolkit

Thanks for helping build language technology for under-served languages. Contributions of all kinds are welcome — code, language profiles, source connectors, evaluation sets, documentation, and (crucially) **native-speaker review**.

## High-value contributions

- **New language profiles** (`configs/languages/*.yaml`) — the single most useful thing you can add. See existing profiles for the shape.
- **Source connectors** (`src/lrl_toolkit/ingest/`) — implement the `BaseConnector` interface for a new corpus source.
- **Native-speaker review & evaluation** — help judge generated conversational data and build small native eval sets. You do not need to be a programmer to help here.
- **Cleaning rules** — script/orthography normalization for a specific language or script.

## Development setup

```bash
git clone https://github.com/lrl-toolkit/lrl-toolkit
cd lrl-toolkit
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ".[all]"
pytest
```

## Standards

- **Formatting & linting:** `ruff format` and `ruff check`. Type hints where practical; `mypy` on `src/`.
- **Tests:** add tests under `tests/`. New connectors should mock network access. The `examples/smoke.yaml` end-to-end run must keep passing.
- **Config-first:** prefer adding capability via config + a small module over hard-coding. Every stage reads the validated `ProjectConfig`.
- **Provenance:** any new data source must produce a `ProvenanceRecord` with a license. See [DATA_ETHICS.md](DATA_ETHICS.md).

## Adding a language profile

1. Copy an existing `configs/languages/<lang>.yaml`.
2. Fill in ISO 639-3 code, script(s), text direction, dialects/orthographies, and the source catalog.
3. Add a note on any script normalization the language needs.
4. Open a PR. If you're a speaker of the language, please say so — it helps reviewers.

## Pull requests

- Keep PRs focused. One connector, one language, or one fix per PR where possible.
- Describe *why*, not just *what*. Link related issues.
- By contributing you agree your work is licensed under [Apache-2.0](LICENSE).

## Code of Conduct

Participation is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
