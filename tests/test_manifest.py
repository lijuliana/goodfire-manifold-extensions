from manifolds.manifest import load_manifest

CORE = {"colors", "temperature", "age", "geography", "days", "years", "formality",
        "sent_length", "political_bias", "political_bias_aug"}


def test_manifest_loads_and_has_all_tiers():
    m = load_manifest()
    assert len(m.manifolds) >= 30
    assert {x.name for x in m.by_tier("core")} == CORE
    assert len(m.by_tier("extended")) >= 9
    assert len(m.by_tier("cyclic")) >= 4
    assert len(m.by_tier("vision")) >= 5
    assert len(set(m.names)) == len(m.names)


def test_core_entries_have_templates_and_geometry_keys():
    m = load_manifest()
    keys = {x.name: x.geometry_key for x in m.by_tier("core")}
    assert keys == {
        "colors": "paraboloid", "temperature": "line", "age": "line", "geography": "tree",
        "days": "circle", "years": "helix", "formality": "line", "sent_length": "line",
        "political_bias": "ordinal", "political_bias_aug": "ordinal",
    }
    for x in m.by_tier("core"):
        assert x.prompt_template, x.name
        assert x.model == "meta-llama/Llama-3.1-8B"
        assert x.layer == 19
        assert x.status == "pending"
