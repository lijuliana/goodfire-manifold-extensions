"""Print markdown tables of the debate extensions for every non-archived page (from results/)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from manifolds.manifest import load_manifest  # noqa: E402
from manifolds.site.build import FEATURED_ORDER  # noqa: E402

R = Path("results")
names = [n for n in FEATURED_ORDER if not load_manifest().get(n).archived]


def x(v):
    return "–" if v is None else f"{v:g}"


def f2(v):
    return "–" if v is None else f"{v:.2f}"


print("| manifold | read at | task (clean) | shape halves at X | k-NN < ½ at X | task halves at X | shape at X = 5% |")
print("|---|---|---|---|---|---|---|")
for n in names:
    a = json.loads((R / n / "ablation.json").read_text())
    bp, c = a["breaking_points"], a["conditions"]
    ps = json.loads((R / n / "prompts.json").read_text())
    pos = "concept word" if ps.get("spans") else "last token"
    task = f"{a['task']} ({a['clean_task_accuracy']:.0%})" if a.get("task") else "–"
    print(f"| {n} | {pos} | {task} | {x(bp.get('shape_half'))} | {x(bp.get('knn_half'))} | "
          f"{x(bp.get('task_half')) if a.get('task') else '–'} | {c['weak_5']['shape_metric']:.2f} of {c['clean']['shape_metric']:.2f} |")
print()
print("| manifold | top writers' tuning η² (random) | top-256 writers zeroed: shape (random 256) | k-NN kept (random) | task (random) |")
print("|---|---|---|---|---|")
for n in names:
    r = json.loads((R / n / "neurons.json").read_text())
    c = r["causal"]
    t, rr = c["top_256"], c["random_256"]
    eta = sum(w["eta2"] for w in r["writers"]) / len(r["writers"])
    task = (f"{t['task_accuracy']:.0%} ({rr['task_accuracy']:.0%})" if t.get("task_accuracy") is not None else "–")
    print(f"| {n} | {eta:.2f} ({r['writers_random_eta2_mean']:.2f}) | {t['shape_metric']:.2f} of {r['clean']['shape_metric']:.2f} ({rr['shape_metric']:.2f}) | "
          f"{t['knn_overlap']:.2f} ({rr['knn_overlap']:.2f}) | {task} |")
print()
print("| manifold | neutral prefix: k-NN kept | most disruptive domain | its k-NN kept | max move into a prefix domain's subspace (random) |")
print("|---|---|---|---|---|")
for n in names:
    c = json.loads((R / n / "crosstalk.json").read_text())["conditions"]
    nu = c["neutral"]
    others = {k: v for k, v in c.items() if k != "neutral"}
    w = min(others, key=lambda k: others[k]["knn_overlap"])
    into = [(v["move_fraction_in_prefix_domain_subspace"], k) for k, v in others.items()
            if v["move_fraction_in_prefix_domain_subspace"] is not None]
    nul = max(v["move_fraction_in_null_subspace"] for v in others.values())
    mx = max(into) if into else None
    print(f"| {n} | {nu['knn_overlap']:.2f} | {w} | {others[w]['knn_overlap']:.2f} | "
          f"{'–' if mx is None else f'{mx[0]:.1%} ({mx[1]})'} ({nul:.1%}) |")
print()
print("| manifold | display position | 0 | 1B | 21B | 105B | 399B | 1.2T | 4T | final | first passes | task (final) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
want = ["0B", "1B", "21B", "105B", "399B", "1196B", "4001B", "final"]
for n in names:
    p = R / n / "checkpoints.json"
    if not p.exists():
        continue
    c = json.loads(p.read_text())
    by = {k["label"]: k for k in c["checkpoints"]}
    cells = [("–" if by[w]["shape_metric"] < 0.1 else f"{by[w]['shape_metric']:.2f}") if w in by else "?" for w in want]
    first = next((k["label"] for k in c["checkpoints"] if k["status"] == "matches"), "never")
    fin = c["checkpoints"][-1]
    task = f"{fin['task_accuracy']:.0%}" if fin.get("task_accuracy") is not None else "–"
    print(f"| {n} | {c['display_layer']} of {c['n_layers']} | " + " | ".join(cells) + f" | {first} | {task} |")
