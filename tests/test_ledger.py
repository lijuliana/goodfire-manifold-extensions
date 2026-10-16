from manifolds import ledger


def test_ledger_roundtrip(tmp_path):
    p = tmp_path / "ledger.jsonl"
    e = ledger.LedgerEntry("azure", "vm", "Tesla T4", "2026-10-15T20:00:00+00:00",
                           "2026-10-15T20:30:00+00:00", 0.5, "extract colors")
    ledger.append(e, p)
    ledger.append(ledger.LedgerEntry("aws", "i", "L40S", e.start, e.end, 0.25, "x"), p)
    assert ledger.read(p) == [e, ledger.read(p)[1]]
    assert ledger.summarize(p) == {"azure": 0.5, "aws": 0.25, "all": 0.75}
    assert ledger.hours_between(e.start, e.end) == 0.5
    q = tmp_path / "other.jsonl"
    ledger.append(e, q)
    ledger.append(ledger.LedgerEntry("gcp", "g", "A100", e.start, e.end, 1.0, "y"), q)
    assert ledger.merge(q, p) == 1 and len(ledger.read(p)) == 3
