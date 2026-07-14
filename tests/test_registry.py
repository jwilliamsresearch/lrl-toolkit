"""The bundled seed profiles must all load and validate."""

import pytest

from lrl_toolkit.registry import (
    list_compute,
    list_languages,
    list_models,
    load_compute,
    load_language,
    load_model,
)

SEED_LANGUAGES = {"welsh", "kurmanji", "sorani", "cornish", "farsi"}


def test_seed_languages_present():
    assert SEED_LANGUAGES.issubset(set(list_languages()))


def test_all_bundled_languages_validate():
    slugs = list_languages()
    assert len(slugs) >= 30  # seeds + the expanded set
    for slug in slugs:
        prof = load_language(slug)
        assert prof.iso639_3 and prof.scripts
        assert prof.resolved_script() in prof.scripts
        # If an NLLB code is set it must be script-tagged (e.g. cym_Latn).
        if prof.nllb_code:
            assert "_" in prof.nllb_code


@pytest.mark.parametrize("slug", sorted(SEED_LANGUAGES))
def test_language_profiles_validate(slug):
    prof = load_language(slug)
    assert prof.iso639_3
    assert prof.scripts
    assert prof.resolved_script() in prof.scripts


def test_all_models_validate():
    slugs = list_models()
    assert slugs
    for slug in slugs:
        m = load_model(slug)
        assert m.hf_id


def test_all_compute_validate():
    slugs = list_compute()
    assert {"consumer_gpu", "a100", "cluster"}.issubset(set(slugs))
    consumer = load_compute("consumer_gpu")
    assert consumer.quantization.value == "4bit"


def test_missing_profile_raises():
    from lrl_toolkit.registry import ProfileNotFoundError

    with pytest.raises(ProfileNotFoundError):
        load_language("this-language-does-not-exist")
