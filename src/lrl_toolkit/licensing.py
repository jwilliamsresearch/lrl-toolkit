"""Resolve a single release license for a model trained on multiple data sources.

Combining differently-licensed corpora into one training run is normal and legal;
the question the export stage actually has to answer is what license to put on the
*resulting model*. This module answers it with one rule, applied consistently:

- **Share-alike (copyleft) sources dominate.** If any source is share-alike (e.g.
  CC-BY-SA-4.0), the model is released under that license — it's the only choice
  that simultaneously satisfies the share-alike source (same license) and every
  attribution-only source (their sole requirement, attribution, is preserved by
  crediting them in the model card). This is the standard, legally conservative
  answer used across the field for models trained partly on ShareAlike data.
- **Two different share-alike licenses can't both dominate** — that's a genuine,
  unresolvable conflict, not something to paper over. Export refuses.
- **No-derivatives / non-commercial sources are an outright block.** A trained
  model is, at minimum, plausibly a derivative use; a source that forbids that
  can't be laundered by mixing in other data. Export refuses.
- **All-permissive sources** get a plain multi-license attribution statement
  rather than a fabricated single license name.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .manifest import ProvenanceRecord


class LicenseConflictError(RuntimeError):
    """Raised when source licenses cannot be reconciled into one release license."""


# Known licenses, classified by the constraint that matters for combination.
# "share_alike": derivative works must carry the same (or a declared-compatible)
#   license.
# "permissive": attribution required, no restriction on the derivative's license.
# "public_domain": no restriction at all.
# "blocked": forbids derivative works or commercial use outright — can't be
#   reconciled by choosing a stricter umbrella license.
_SHARE_ALIKE = {"cc-by-sa-4.0", "cc-by-sa-3.0"}
_PERMISSIVE = {"cc-by-4.0", "cc-by-3.0", "odc-by-1.0", "apache-2.0", "mit"}
_PUBLIC_DOMAIN = {"cc0-1.0", "public-domain", "pd"}
_BLOCKED = {"cc-by-nc-4.0", "cc-by-nd-4.0", "cc-by-nc-nd-4.0", "cc-by-nc-sa-4.0"}


def _norm(license_id: str) -> str:
    return license_id.strip().lower()


@dataclass
class ResolvedLicense:
    license: str
    """The license to release the model under."""
    rationale: str
    """Human-readable explanation, for the model card."""
    attributions: list[str] = field(default_factory=list)
    """Per-source 'Source (License) — URL' lines, required for BY/SA compliance."""


def _attribution_line(rec: ProvenanceRecord) -> str:
    parts = [rec.source, f"({rec.license})" if rec.license else "(license unrecorded)"]
    if rec.url:
        parts.append(f"— {rec.url}")
    return " ".join(parts)


def resolve_release_license(provenance: list[ProvenanceRecord]) -> ResolvedLicense:
    """Determine the license to release a model under, given its source provenance.

    Raises :class:`LicenseConflictError` if the sources cannot be reconciled —
    this is a hard export-time gate, not a warning.
    """
    licensed = [r for r in provenance if r.license]
    attributions = [_attribution_line(r) for r in licensed]

    distinct = {_norm(r.license) for r in licensed if r.license}  # type: ignore[arg-type]

    blocked = distinct & _BLOCKED
    if blocked:
        names = ", ".join(sorted(r.source for r in licensed if _norm(r.license or "") in blocked))
        raise LicenseConflictError(
            f"Source(s) under a non-derivative/non-commercial license ({', '.join(sorted(blocked))}) "
            f"cannot be reconciled into a released model: {names}. Remove these sources or obtain "
            "explicit permission before export."
        )

    share_alike = distinct & _SHARE_ALIKE
    if len(share_alike) > 1:
        raise LicenseConflictError(
            f"Multiple distinct share-alike licenses present ({', '.join(sorted(share_alike))}); "
            "these cannot be simultaneously satisfied by one release license. Resolve manually."
        )

    unknown = distinct - _SHARE_ALIKE - _PERMISSIVE - _PUBLIC_DOMAIN - _BLOCKED
    if unknown:
        raise LicenseConflictError(
            f"Unclassified license(s) {sorted(unknown)} — add them to licensing.py's classification "
            "tables (as share-alike/permissive/public-domain/blocked) before export can proceed."
        )

    if share_alike:
        sa = next(iter(share_alike))
        canonical = next(r.license for r in licensed if _norm(r.license or "") == sa)  # type: ignore[arg-type]
        return ResolvedLicense(
            license=canonical,
            rationale=(
                f"{canonical} is the strictest (share-alike) license among this model's sources. "
                "Share-alike requires derivative works to carry the same license; every other source "
                "here is attribution-only and is satisfied by attribution, so the whole model is "
                f"released under {canonical} to honor all sources at once."
            ),
            attributions=attributions,
        )

    if distinct and distinct <= _PUBLIC_DOMAIN:
        return ResolvedLicense(
            license="CC0-1.0",
            rationale="All sources are public-domain / CC0; no restrictions apply.",
            attributions=attributions,
        )

    names = ", ".join(sorted({r.license for r in licensed if r.license}))  # type: ignore[misc]
    return ResolvedLicense(
        license=f"Multiple permissive licenses ({names})",
        rationale=(
            "All sources are attribution-only permissive licenses with no share-alike clause; the "
            "model is released with attribution to each source rather than under a single reused "
            "license name (none of them applies to the whole work by itself)."
        ),
        attributions=attributions,
    )
