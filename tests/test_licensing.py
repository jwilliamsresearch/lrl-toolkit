"""License resolution for export: what license should the released model carry
when trained on differently-licensed sources."""

import pytest

from lrl_toolkit.licensing import LicenseConflictError, resolve_release_license
from lrl_toolkit.manifest import ProvenanceRecord


def _rec(source: str, license_: str, url: str | None = None) -> ProvenanceRecord:
    return ProvenanceRecord(source=source, license=license_, url=url, n_docs=1)


def test_share_alike_dominates_permissive_sources():
    # The Kurmanji scenario: Wikipedia (CC-BY-SA) + CulturaX/FineWeb-2 (ODC-BY) +
    # MADLAD-400 (CC-BY). SA is the strictest and satisfies everyone via attribution.
    records = [
        _rec("wikipedia", "CC-BY-SA-4.0", "https://ku.wikipedia.org"),
        _rec("culturax", "ODC-BY-1.0"),
        _rec("madlad400", "CC-BY-4.0"),
        _rec("fineweb2", "ODC-BY-1.0"),
    ]
    result = resolve_release_license(records)
    assert result.license == "CC-BY-SA-4.0"
    assert "share-alike" in result.rationale.lower()
    assert len(result.attributions) == 4
    assert any("wikipedia" in a.lower() for a in result.attributions)


def test_all_permissive_sources_get_multi_license_statement():
    records = [_rec("culturax", "ODC-BY-1.0"), _rec("madlad400", "CC-BY-4.0")]
    result = resolve_release_license(records)
    assert "ODC-BY-1.0" in result.license
    assert "CC-BY-4.0" in result.license
    assert "CC-BY-SA" not in result.license


def test_all_cc0_resolves_to_cc0():
    records = [_rec("a", "CC0-1.0"), _rec("b", "CC0-1.0")]
    result = resolve_release_license(records)
    assert result.license == "CC0-1.0"


def test_conflicting_share_alike_licenses_raise():
    records = [_rec("a", "CC-BY-SA-4.0"), _rec("b", "CC-BY-SA-3.0")]
    with pytest.raises(LicenseConflictError):
        resolve_release_license(records)


def test_non_derivative_source_blocks_export():
    records = [_rec("wikipedia", "CC-BY-SA-4.0"), _rec("restricted", "CC-BY-NC-4.0")]
    with pytest.raises(LicenseConflictError, match="non-derivative|non-commercial"):
        resolve_release_license(records)


def test_unclassified_license_raises_rather_than_guessing():
    records = [_rec("mystery", "Some-Weird-License-1.0")]
    with pytest.raises(LicenseConflictError, match="[Uu]nclassified"):
        resolve_release_license(records)
