"""Unit tests for normalization, quality filters, dedup, PII, and corpus IO."""

from lrl_toolkit.clean import filters, pii
from lrl_toolkit.clean.dedup import Deduper
from lrl_toolkit.clean.normalize import normalize_text
from lrl_toolkit.corpus import Document, ShardWriter, iter_documents
from lrl_toolkit.registry import NormalizationRules


def test_normalize_persian_yeh_kaf_and_digits():
    rules = NormalizationRules(normalize_digits=True, rules=["persian_yeh_kaf"])
    # Arabic yeh (U+064A) -> Persian yeh (U+06CC); Arabic kaf -> Persian kaf.
    src = "يك ٠١٢"
    out = normalize_text(src, rules)
    assert "ی" in out and "ک" in out
    assert "012" in out  # digits folded to ASCII


def test_normalize_strips_controls_and_nfc():
    rules = NormalizationRules(rules=["kurdish_latin"])
    out = normalize_text("a\x00b\x07c", rules)
    assert out == "abc"


def test_quality_filter_drops_short_and_symbolic():
    assert not filters.assess("too short").keep
    assert filters.assess("!!! ??? ### " * 30, min_quality=0.5).keep is False
    good = (
        "This is a reasonably long and clean paragraph of ordinary prose text that "
        "should comfortably pass the quality heuristics because it has enough words, "
        "enough characters, few symbols, and no repeated lines whatsoever here."
    )
    assert filters.assess(good).keep


def test_exact_deduper():
    d = Deduper("exact")
    assert d.is_duplicate("hello world") is False
    assert d.is_duplicate("hello   world") is True  # canonicalized whitespace
    assert d.is_duplicate("something else") is False


def test_pii_scrub():
    text = "Email me at a.b@example.com or visit https://x.com and call 0123 456 789."
    out, n = pii.scrub(text)
    assert "[EMAIL]" in out and "[URL]" in out and "[NUM]" in out
    assert n >= 3


def test_corpus_roundtrip(tmp_path):
    writer = ShardWriter(tmp_path / "c", prefix="p", docs_per_shard=2)
    docs = [Document(text=f"doc {i}", source="unit", meta={"i": i}) for i in range(5)]
    for d in docs:
        writer.write(d)
    writer.close()
    assert len(writer.shards) == 3  # 5 docs, 2 per shard

    read = list(iter_documents(tmp_path / "c"))
    assert [d.text for d in read] == [d.text for d in docs]
    assert read[0].meta["i"] == 0
