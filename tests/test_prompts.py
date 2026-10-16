from manifolds.prompts.core import (
    DAYS,
    TIMES,
    build_age,
    build_days,
    build_temperature,
    build_years,
)


def test_temperature_matches_released_template():
    ps = build_temperature()
    assert len(ps) == 150
    assert ps.prompts[0] == "Today it's -30 degrees Fahrenheit outside"
    assert ps.prompts[-1] == "Today it's 119 degrees Fahrenheit outside"
    assert ps.labels[32] == {"fahrenheit": 2.0, "celsius": (2 - 32) * 5 / 9}


def test_age_matches_released_template():
    ps = build_age()
    assert len(ps) == 99
    assert ps.prompts[0] == "They are 1 years old. "
    assert ps.labels[-1]["age"] == 99.0


def test_days_grid():
    ps = build_days()
    assert len(ps) == 56 == len(DAYS) * len(TIMES)
    assert ps.prompts[0] == "It's the crack of dawn on Monday"
    assert ps.labels[9] == {"day": "Tuesday", "day_idx": 1, "time": "early morning", "time_idx": 1}
    assert ps.deviations  # records the 420 vs 56 discrepancy


def test_years_helix_labels():
    ps = build_years()
    assert len(ps) == 199
    assert ps.prompts[0] == "The date is 1800"
    assert ps.prompts[-1] == "The date is 1998"
    assert ps.labels[7] == {"year": 1807.0, "decade_digit": 7.0, "decade": 0.0}
