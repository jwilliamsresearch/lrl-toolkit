"""Test fixtures, incl. an offline config root so the pipeline needs no network."""

import sys
from pathlib import Path

import pytest
import yaml

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Long, clean sample docs (pass the quality filter: >200 chars, >20 words).
_GOOD_DOCS = [
    (
        "Mae'r gath ddu yn eistedd ar y wal yn yr heulwen gynnes tra bod yr adar "
        "bach yn canu yn y coed uwchben. Roedd y plant yn chwarae yn yr ardd drwy'r "
        "prynhawn cyn i'r glaw ddechrau disgyn yn drwm o'r cymylau llwyd uwchben y mynydd."
    ),
    (
        "Aeth y ffermwr i'r farchnad yn gynnar y bore i werthu ei ddefaid a'i wartheg "
        "cyn dychwelyd adref gyda bwyd a nwyddau ar gyfer yr wythnos nesaf i'r teulu "
        "cyfan a oedd yn aros amdano wrth y drws yn llawen ac yn barod i helpu."
    ),
    (
        "Mae hanes hir a chyfoethog gan yr iaith Gymraeg sy'n cael ei siarad gan "
        "gannoedd o filoedd o bobl ledled Cymru a thu hwnt, ac mae ymdrechion mawr "
        "yn cael eu gwneud heddiw i sicrhau ei bod yn ffynnu ymysg y genhedlaeth nesaf."
    ),
]
_JUNK_DOC = "### >>> !!! @@@ ??? *** ||| ~~~ <<< +++ === --- ::: ;;; ,,, ... 999 000 111"


@pytest.fixture
def offline_configs(tmp_path, monkeypatch):
    """Create a temp config root with local-corpus language profiles and export it
    via LRL_CONFIG_PATH so tests run fully offline (models/compute come from the
    bundled configs)."""
    root = tmp_path / "cfg"
    langs = root / "languages"
    langs.mkdir(parents=True)

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for i, text in enumerate(_GOOD_DOCS):
        (corpus / f"doc{i}.txt").write_text(text, encoding="utf-8")
    # An exact duplicate (should be deduped) and a junk doc (should be dropped).
    (corpus / "dup.txt").write_text(_GOOD_DOCS[0], encoding="utf-8")
    (corpus / "junk.txt").write_text(_JUNK_DOC, encoding="utf-8")

    def _profile(license_value: str) -> dict:
        return {
            "display_name": "Test Language",
            "iso639_3": "tst",
            "scripts": ["Latin"],
            "sources": [
                {
                    "connector": "local",
                    "params": {"path": str(corpus), "license": license_value},
                }
            ],
        }

    (langs / "testlang.yaml").write_text(yaml.safe_dump(_profile("CC0-1.0")), encoding="utf-8")
    (langs / "testlang_unlicensed.yaml").write_text(
        yaml.safe_dump(_profile("unknown")), encoding="utf-8"
    )

    monkeypatch.setenv("LRL_CONFIG_PATH", str(root))
    return {"root": root, "corpus": corpus, "n_good": len(_GOOD_DOCS)}


def _build_tiny_model(dest: Path) -> None:
    """Create a minimal random Llama causal LM + fast BPE tokenizer on disk, so
    the tokenizer/pretrain stages are testable offline and in seconds."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    corpus = _GOOD_DOCS + ["abcdefghijklmnopqrstuvwxyz " * 20]
    tk = Tokenizer(models.BPE(unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    trainer = trainers.BpeTrainer(
        vocab_size=500, special_tokens=["[UNK]", "[PAD]", "<|endoftext|>"]
    )
    tk.train_from_iterator(corpus, trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<|endoftext|>",
        bos_token="<|endoftext|>",
    )

    cfg = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(cfg)
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest)
    tokenizer.save_pretrained(dest)


@pytest.fixture
def tiny_model(offline_configs, tmp_path):
    """Register a tiny local model profile ('tinytest') alongside offline_configs."""
    root = offline_configs["root"]
    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = tmp_path / "tinymodel"
    _build_tiny_model(model_path)
    (models_dir / "tinytest.yaml").write_text(
        yaml.safe_dump(
            {
                "hf_id": str(model_path),
                "family": "llama",
                "arch": "decoder",
                "tokenizer_type": "bpe",
                "context_length": 128,
            }
        ),
        encoding="utf-8",
    )
    return {**offline_configs, "model": "tinytest", "model_path": model_path}
