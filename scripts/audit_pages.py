"""Audit every visible page for consistency: same sections, same pipeline version, same
parameters, same presentation, except for differences that are intended (probe view on
Fourier-probe manifolds, steering only where a task is defined, SAE layer = read layer)."""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")
from manifolds.analysis.shapes import ROLES  # noqa: E402
from manifolds.manifest import load_manifest  # noqa: E402
from manifolds.steer import STEER_TASKS  # noqa: E402

R, S = Path("results"), Path("site/manifolds")
man = load_manifest()
names = [m.name for m in man.manifolds if not m.archived and (R / m.name / "report.json").exists()]
SECTIONS = ["<h2>What the axes mean</h2>", "<h2>Across the layers</h2>",
            "<h2>Weak activations: does", "<h2>Which neurons write", "<h2>Interference with other",
            "<h2>Forming during pre-training", "<summary><b>Steering along the manifold</b>",
            "<summary><b>SAE tiling</b>", "<summary><b>What the source reports</b>",
            "<summary><b>Replication status</b>", "<summary><b>Prompts</b>",
            "<summary><b>Extraction</b>"]


def j(n, f):
    p = R / n / f
    return json.loads(p.read_text()) if p.exists() else None


def frames_have_captions(fig):
    fr = fig.get("frames", [])
    return bool(fr) and all(f.get("layout", {}).get("annotations") for f in fr)


def marker_mode(fig):
    for t in fig.get("data", []):
        if t.get("type") == "scatter3d" and t.get("name") not in ("shadow",):
            c = t.get("marker", {}).get("color")
            if isinstance(c, list) and c and isinstance(c[0], str) and c[0].startswith("#"):
                return "categorical/hex"
            if isinstance(c, (list, dict)):
                return "scale"
    return "?"


rows = {}
for n in names:
    r = {}
    html = (S / f"{n}.html").read_text() if (S / f"{n}.html").exists() else ""
    pos = [html.find(h) for h in SECTIONS]
    r["sections"] = "".join("Y" if p >= 0 else "-" for p in pos)
    present = [p for p in pos if p >= 0]
    r["order_ok"] = present == sorted(present)
    probe = ROLES[n]["kind"] == "fourier_probe"
    r["kind"] = "probe" if probe else "pca"
    rep = j(n, "report.json")
    r["legend_order"] = rep["color"].get("category_order", [])[:2] if rep["color"].get("category_order") else "-"
    d = j(n, "depth.json") or {}
    r["depth_frames_view"] = d.get("frames_view", "(not stamped)")
    df = j(n, "depth_figure.json") or {}
    r["depth_n_frames"] = len(df.get("frames", []))
    r["depth_colour"] = marker_mode(df)
    r["axis_labels"] = (R / n / "axis_labels.json").exists()
    sae = j(n, "sae.json")
    r["sae_layer"] = re.search(r"L(\d+)R", sae["sae"]["repo"]).group(1) if sae else "-"
    st = (d.get("steering") or {})
    r["steer"] = ("expected" if n in STEER_TASKS else "none") + "/" + ("present" if st else "absent")
    r["steer_code"] = "new" if st and "steered_position" in st.get("extras", {}) else ("old" if st else "-")
    a = j(n, "ablation.json")
    af = j(n, "ablation_figure.json") or {}
    r["abl"] = (f"{len(a['conditions'])}c kl{a['n_kl']} pts{a['n_plotted']} view={'Y' if 'view' in a else '-'} cap={'Y' if frames_have_captions(af) else '-'} {marker_mode(af)}") if a else "-"
    nu = j(n, "neurons.json")
    nf = j(n, "neurons_figure.json") or {}
    r["neu"] = (f"ks{sorted({v['k'] for v in nu['causal'].values()})} prompts{nu['n_prompts']} view={'Y' if 'view' in nu else '-'} cap={'Y' if frames_have_captions(nf) else '-'} {marker_mode(nf)}") if nu else "-"
    ct = j(n, "crosstalk.json")
    r["xt"] = (f"{len(ct['conditions'])}dom centred={'Y' if 'preprocessing' in ct else '-'} ex={'Y' if all('examples' in v for v in ct['conditions'].values()) else '-'}") if ct else "-"
    ctp = R / n / "crosstalk.json"
    r["xt_time"] = ctp.stat().st_mtime if ctp.exists() else 0
    w = j(n, "woven_figure.json") or {}
    r["woven_partner_grey"] = any("partner" in str(t.get("name", "")) for t in w.get("data", []))
    ck = j(n, "checkpoints.json")
    cf = j(n, "checkpoints_figure.json") or {}
    if ck:
        lay = [tuple(c.get("layers_checked", [])) for c in ck["checkpoints"]]
        nonfinal = sorted({len(x) for x in lay[:-1]})
        r["ckpt"] = f"view={'Y' if 'view' in ck else '-'} rule={'Y' if 'display_layer_rule' in ck else '-'} cap={'Y' if frames_have_captions(cf) else '-'} layers/ckpt={nonfinal}+{len(lay[-1])} {marker_mode(cf)}"
    else:
        r["ckpt"] = "-"
    def prov(obj, key="provenance"):
        pv = (obj or {}).get(key) or {}
        return (f"{pv.get('pipeline_version', '-')} {pv.get('code_commit', '-')} {pv.get('gpu', '-')} "
                f"{pv.get('dtype', '')} bs{pv.get('batch_size', '')}").strip() if pv else "(none)"
    r["prov_sae"] = prov(sae)
    r["prov_steer"] = prov(st) if st else "-"
    r["prov_layers"] = prov(d, "frames_provenance")
    r["prov_ablation"] = prov(a)
    r["prov_neurons"] = prov(nu)
    r["prov_crosstalk"] = prov(ct) + (f" null{(ct or {}).get('provenance', {}).get('null_bases_sha')}" if ct else "")
    r["prov_checkpoints"] = prov(ck)
    r["report_rubric"] = rep.get("rubric_version")
    rows[n] = r

keys = [k for k in next(iter(rows.values())) if k not in ("xt_time",)]
print(f"{len(names)} pages; section key: " + ", ".join(f"{i}={re.sub('<[^>]+>', '', h)}" for i, h in enumerate(SECTIONS)))
for k in keys:
    groups = defaultdict(list)
    for n, r in rows.items():
        groups[str(r[k])].append(n)
    if len(groups) == 1:
        print(f"[same] {k}: {next(iter(groups))}")
    else:
        print(f"[DIFF] {k}:")
        for v, ns in sorted(groups.items(), key=lambda t: -len(t[1])):
            print(f"         {v}  <- {', '.join(ns)}")

