"""Extended + cyclic builders that need no network."""

import numpy as np

from manifolds.analysis.shapes import check_circumplex, check_fourier_probe
from manifolds.prompts.cyclic import (
    build_addition_sum_fourier,
    build_hours_circle,
    build_months_circle,
    build_weekdays_circle,
)
from manifolds.prompts.extended import (
    build_adverbs_pace,
    build_bible_verses,
    build_chemical_bonds,
    build_classical_composers,
    build_numbers_1_100,
    build_ordinals,
    build_scotus_cases,
    build_slant_rhymes_ore,
    build_statistical_uncertainty,
    ordinal_word,
)


def test_ordinal_words():
    assert [ordinal_word(n) for n in (1, 2, 3, 11, 12, 20, 21, 33, 40, 99, 100)] == [
        "first", "second", "third", "eleventh", "twelfth", "twentieth", "twenty-first",
        "thirty-third", "fortieth", "ninety-ninth", "one hundredth"]
    ps = build_ordinals()
    assert len(ps) == 100 and ps.prompts[0] == "She finished the race in first place"


def test_extended_builders_shapes():
    assert len(build_classical_composers()) >= 80
    assert len(build_scotus_cases()) >= 50
    assert len(build_chemical_bonds()) >= 40
    ps = build_adverbs_pace()
    assert len(ps) == 5 * 18 and ps.labels[0]["speed_rank"] == 0.0
    ps = build_statistical_uncertainty()
    vals = np.array([lab["ln_std_over_mean"] for lab in ps.labels])
    assert len(ps) == 300 and vals.min() > -5.2 and vals.max() < 0.6
    ps = build_bible_verses()
    assert ps.labels[0]["book"] == "Genesis" and ps.labels[-1]["book"] == "Revelation"
    ps = build_slant_rhymes_ore()
    assert {lab["rhyme_grade"] for lab in ps.labels} == {0.0, 1.0, 2.0}
    assert len(build_numbers_1_100()) == 100


def test_cyclic_spans_point_at_concept():
    for ps, key in ((build_weekdays_circle(), "day"), (build_months_circle(), "month")):
        assert ps.spans is not None and len(ps.spans) == len(ps)
        for p, lab, (a, b) in zip(ps.prompts, ps.labels, ps.spans, strict=True):
            assert p[a:b] == lab[key]
    ps = build_hours_circle()
    assert len(ps) == 24 * 48 and ps.spans is not None
    a, b = ps.spans[0]
    assert ps.prompts[0][a:b] == "00"
    ps = build_addition_sum_fourier()
    assert len(ps) == 1200 and ps.spans is None
    assert all(lab["sum"] == lab["a"] + lab["b"] for lab in ps.labels)


def test_circumplex_and_fourier_checks(rng):
    emos = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
    va = {"happiness": (0.8, 0.5), "surprise": (0.2, 0.8), "fear": (-0.6, 0.8),
          "anger": (-0.7, 0.7), "disgust": (-0.7, 0.3), "sadness": (-0.7, -0.5)}
    group = np.repeat(np.array(emos), 30)
    local = np.array([[*va[e], 0.0] for e in group]) * 3 + 0.15 * rng.normal(size=(180, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    coords = local @ q.T
    v = np.array([va[e][0] for e in group])
    a = np.array([va[e][1] for e in group])
    rep = check_circumplex(coords, group, v, a)
    assert rep.status == "matches", rep.reason
    assert check_circumplex(coords, rng.permutation(group), v, a).status != "matches"
    # fourier: activations carry cos/sin of sum/10 and sum/5 through random directions
    s = rng.integers(2, 200, size=400).astype(float)
    feats = np.c_[np.cos(2 * np.pi * s / 10), np.sin(2 * np.pi * s / 10),
                  np.cos(2 * np.pi * s / 5), np.sin(2 * np.pi * s / 5), s / 200]
    acts = feats @ rng.normal(size=(5, 64)) + 0.05 * rng.normal(size=(400, 64))
    coords = acts[:, :3]
    rep = check_fourier_probe(acts, coords, s)
    assert rep.status in ("matches", "partial"), rep.reason
    assert rep.extras["probe_r2"][10] > 0.8 and rep.extras["control_r2"] < 0.5
