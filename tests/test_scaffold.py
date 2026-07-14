"""The wizard/CLI scaffold must always produce a valid project config."""

import yaml

from lrl_toolkit.config import ProjectConfig
from lrl_toolkit.scaffold import scaffold_yaml


def test_scaffold_yaml_parses_to_valid_config():
    text = scaffold_yaml("proj", "welsh", "qwen2.5-1.5b", "consumer_gpu")
    data = yaml.safe_load(text)
    cfg = ProjectConfig.model_validate(data)
    assert cfg.name == "proj"
    assert cfg.language == "welsh"
    assert cfg.convdata.translate_backend == "nllb"


def test_scaffold_yaml_honors_options():
    text = scaffold_yaml(
        "p", "cornish", "gemma2-2b", "a100",
        translate_backend="teacher", synth_provider="mock", review=False, synth_n=10,
    )
    cfg = ProjectConfig.model_validate(yaml.safe_load(text))
    assert cfg.base_model == "gemma2-2b"
    assert cfg.compute == "a100"
    assert cfg.convdata.translate_backend == "teacher"
    assert cfg.convdata.synth.provider == "mock"
    assert cfg.convdata.review is False
