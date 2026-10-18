"""Command-line entry point: prompts | extract | analyze | site | ledger | status."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from manifolds import RESULTS_DIR, SITE_DIR
from manifolds.config import DEFAULT_CONFIG_PATH, load_config
from manifolds.manifest import load_manifest


def _select(args: argparse.Namespace) -> list[str]:
    from manifolds.prompts import BUILDERS

    manifest = load_manifest()
    if getattr(args, "manifold", None):
        names = list(args.manifold)
    elif getattr(args, "tier", None):
        names = [m.name for m in manifest.by_tier(args.tier) if not m.archived]
    else:
        names = [m.name for m in manifest.manifolds if m.name in BUILDERS and not m.archived]
    missing = [n for n in names if n not in BUILDERS]
    if missing:
        print(f"skipping (no prompt builder yet): {missing}", file=sys.stderr)
    return [n for n in names if n in BUILDERS]


def cmd_prompts(args: argparse.Namespace) -> int:
    from manifolds.prompts import build

    for name in _select(args):
        ps = build(name)
        ps.save(RESULTS_DIR / name / "prompts.json")
        print(f"{name}: {len(ps)} prompts  template={ps.template!r}")
        for d in ps.deviations:
            print(f"   deviation: {d}")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    from manifolds.extract import (
        ActivationExtractor,
        config_from_run,
        extract_and_cache,
        precision_check,
    )
    from manifolds.prompts import build

    cfg = load_config(args.config)
    manifest = load_manifest()
    ext = ActivationExtractor(config_from_run(cfg, device=args.device))
    print(json.dumps(ext.metadata(), indent=1))
    names = _select(args)
    checked = False
    for name in names:
        out_dir = RESULTS_DIR / name
        if (out_dir / "activations.npy").exists() and not args.force:
            print(f"{name}: cached, skipping (use --force to redo)")
            continue
        ps_path = out_dir / "prompts.json"
        from manifolds.prompts.types import PromptSet

        ps = PromptSet.load(ps_path) if ps_path.exists() else build(name)
        spec = manifest.get(name)
        layer = spec.layer if isinstance(spec.layer, int) else cfg.model.layer
        pos = "concept token" if ps.spans else "last token"
        print(f"{name}: extracting {len(ps)} prompts (block {layer}, {pos})")
        extract_and_cache(ps, ext, out_dir, layer=layer, provider=args.provider,
                          instance=args.instance)
        if args.precision_check and not checked and ext.cfg.dtype != "float32":
            pc = precision_check(ext, ps.prompts, cfg.precision_check.n_prompts)
            pc["manifold"] = name
            pc["passed"] = pc["cosine_min"] >= cfg.precision_check.min_cosine
            (out_dir / "precision_check.json").write_text(json.dumps(pc, indent=1))
            print(f"precision check on {name}: {pc}")
            checked = True
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from manifolds.analysis.report import analyze_manifold

    cfg = load_config(args.config)
    manifest = load_manifest()
    statuses: dict[str, dict[str, str]] = {}
    status_path = RESULTS_DIR / "status.json"
    if status_path.exists():
        statuses = json.loads(status_path.read_text(encoding="utf-8"))
    for name in _select(args):
        if not (RESULTS_DIR / name / "activations.npy").exists():
            print(f"{name}: no activations, skipping", file=sys.stderr)
            continue
        rep = analyze_manifold(manifest.get(name), cfg)
        statuses[name] = {"status": rep["status"], "reason": rep["shape"]["reason"]}
        evr = ", ".join(f"{v:.2f}" for v in rep["pca"]["explained_variance_ratio"])
        print(f"{name:16s} {rep['status']:8s} evr=[{evr}]  {rep['shape']['reason']}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(statuses, indent=1), encoding="utf-8")
    return 0


def cmd_sae(args: argparse.Namespace) -> int:
    import torch

    from manifolds.sae.llama_scope import LlamaScopeSAE
    from manifolds.sae.report import analyze_sae

    cfg = load_config(args.config)
    manifest = load_manifest()
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    saes: dict[int, LlamaScopeSAE] = {}
    for name in _select(args):
        if not (RESULTS_DIR / name / "activations.npy").exists():
            print(f"{name}: no activations, skipping", file=sys.stderr)
            continue
        spec = manifest.get(name)
        layer = spec.layer if isinstance(spec.layer, int) else cfg.model.layer
        if layer not in saes:   # Llama Scope has one residual SAE per layer (L{layer}R-8x)
            saes[layer] = LlamaScopeSAE.from_hf(subfolder=f"Llama3_1-8B-Base-L{layer}R-8x",
                                                device=device)
            print(json.dumps(saes[layer].info, indent=1))
        sae = saes[layer]
        rep = analyze_sae(spec, cfg, sae, max_k=args.max_k,
                          max_points=args.max_points, provider=args.provider,
                          instance=args.instance)
        st, sm = rep["stats"], rep["summary"]
        print(f"{name:16s} n={rep['n_points']} L0={st['l0_mean']:.1f} "
              f"VE_total={st['ve_total']:.3f} VE_within={st['ve_within']:.2f} "
              f"geo/pca@amb={sm['geometric_over_pca_at_ambient_dim']} "
              f"k_match={sm['k_geometric_to_match_pca_at_ambient_dim']} "
              f"geo/random@64={sm['geometric_over_random_at_64']} "
              f"dilution={sm['consistent_with_dilution_claim']} ({rep['elapsed_s']}s)")
    return 0


def cmd_depth(args: argparse.Namespace) -> int:
    import torch

    from manifolds.depth import DepthConfig, FullModel
    from manifolds.depth_report import run_depth
    from manifolds.lens import TunedLens

    cfg = load_config(args.config)
    manifest = load_manifest()
    fm = FullModel(DepthConfig(model_id=cfg.model.id, fallback_model_id=cfg.model.fallback_id,
                               dtype=args.dtype, device=args.device,
                               batch_size=args.batch_size, gpu_memory_gib=args.gpu_gib,
                               seed=cfg.seed))
    print(f"full model {fm.model_id} ({fm.resolution}), {fm.n_layers} blocks")
    tuned = (None if (args.no_tuned_lens or args.figure_only)
             else TunedLens.from_hf(device="cpu", dtype=torch.float32))
    jlens = None
    if not args.no_jlens and not args.figure_only:
        from manifolds.jlens import JacobianLensFull

        jlens = JacobianLensFull.from_hf()
        print(f"jacobian lens: {jlens.info.get('repo')} ({jlens.n_prompts} prompts)")
    for name in _select(args):
        if not (RESULTS_DIR / name / "activations.npy").exists():
            print(f"{name}: no activations, skipping", file=sys.stderr)
            continue
        if args.figure_only:
            from manifolds.depth_report import rebuild_depth_figure

            r = rebuild_depth_figure(manifest.get(name), cfg, fm, max_points=args.max_points,
                                     provider=args.provider, instance=args.instance,
                                     recapture=args.recapture)
            print(f"{name:18s} depth figure rebuilt ({r['frames_view']}, {r['elapsed_s']}s)")
            continue
        rep = run_depth(manifest.get(name), cfg, fm, tuned, jlens=jlens,
                        max_points=args.max_points, do_steer=not args.no_steer,
                        provider=args.provider, instance=args.instance)
        c = rep["consistency_block19_vs_truncated"]
        s19 = rep["scores"][str(cfg.model.layer + 1)]
        print(f"{name:18s} n={rep['n_points']} cos19={c['cosine_min']:.5f} "
              f"status@19={s19['status']} steer={'yes' if rep['steering'] else 'no'} "
              f"({rep['elapsed_s']}s)")
    return 0


def _full_model(args: argparse.Namespace, cfg: Any) -> Any:
    from manifolds.depth import DepthConfig, FullModel

    fm = FullModel(DepthConfig(model_id=cfg.model.id, fallback_model_id=cfg.model.fallback_id,
                               dtype=args.dtype, device=args.device, batch_size=args.batch_size,
                               gpu_memory_gib=args.gpu_gib, seed=cfg.seed))
    print(f"full model {fm.model_id} ({fm.resolution}), {fm.n_layers} blocks, dtype {args.dtype}")
    return fm


def cmd_ablate(args: argparse.Namespace) -> int:
    from manifolds.ablate import Ablator, conditions, run_global, run_manifold

    cfg = load_config(args.config)
    manifest = load_manifest()
    fm = _full_model(args, cfg)
    ablator = Ablator(fm.model, seed=cfg.seed)
    conds = conditions()
    glob = run_global(fm, ablator, conds, n_heldout=args.n_heldout, n_popqa=args.n_popqa,
                      lm_batch=args.lm_batch, provider=args.provider, instance=args.instance)
    for name in _select(args):
        if not (RESULTS_DIR / name / "activations.npy").exists():
            print(f"{name}: no activations, skipping", file=sys.stderr)
            continue
        if (RESULTS_DIR / name / "ablation.json").exists() and not args.force:
            print(f"{name}: ablation.json exists, skipping (use --force)")
            continue
        rep = run_manifold(manifest.get(name), cfg, fm, ablator, conds, glob,
                           max_points=args.max_points, provider=args.provider,
                           instance=args.instance, workers=args.workers)
        print(f"{name:20s} breaking points {rep['breaking_points']} ({rep['elapsed_s']}s)")
    ablator.remove()
    return 0


def cmd_neurons(args: argparse.Namespace) -> int:
    from manifolds.neurons import run_neurons

    cfg = load_config(args.config)
    manifest = load_manifest()
    fm = _full_model(args, cfg)
    for name in _select(args):
        if not (RESULTS_DIR / name / "activations.npy").exists():
            print(f"{name}: no activations, skipping", file=sys.stderr)
            continue
        if (RESULTS_DIR / name / "neurons.json").exists() and not args.force:
            print(f"{name}: neurons.json exists, skipping (use --force)")
            continue
        rep = run_neurons(manifest.get(name), cfg, fm, max_prompts=args.max_prompts,
                          max_points=args.max_points, provider=args.provider,
                          instance=args.instance)
        print(f"{name:20s} half the subspace variance in {rep['n_neurons_for_half_share']} "
              f"neurons; top writer share {rep['writers'][0]['share']:.3f} "
              f"({rep['elapsed_s']}s)")
    return 0


def cmd_interference(args: argparse.Namespace) -> int:
    from manifolds.interference import run_geometry, woven_figure

    cfg = load_config(args.config)
    manifest = load_manifest()
    names = [m.name for m in manifest.manifolds
             if (RESULTS_DIR / m.name / "activations.npy").exists() and not m.archived]
    res = run_geometry(manifest, cfg, names, k=args.k)
    print(f"{len(res['names'])} manifolds; null (pooled) {res['null_pooled_mean']:.4f}, "
          f"isotropic {res['null_isotropic_mean']:.4f}")
    for r in res["pairs_sorted"][:10]:
        print(f"  {r['a']:22s} {r['b']:22s} overlap={r['overlap']:.3f} "
              f"weighted={r['weighted']:.3f}")
    for name in _select(args):
        if name not in res["names"]:
            continue
        i = res["names"].index(name)
        row = [(res["names"][j], res["overlap"][i][j]) for j in range(len(res["names"]))
               if j != i and res["overlap"][i][j] is not None]
        row = [r for r in row if r[1] == r[1]]
        if not row:
            continue
        partner = max(row, key=lambda r: r[1])[0]
        fig = woven_figure(RESULTS_DIR, cfg, name, partner)
        (RESULTS_DIR / name / "woven_figure.json").write_text(fig.to_json(), encoding="utf-8")
        (RESULTS_DIR / name / "woven.json").write_text(
            json.dumps({"partner": partner, "overlap": max(row, key=lambda r: r[1])[1]}),
            encoding="utf-8")
        print(f"  {name}: woven view with {partner}")
    return 0


def cmd_crosstalk(args: argparse.Namespace) -> int:
    from manifolds.interference import GLOBAL_DIR, run_crosstalk

    cfg = load_config(args.config)
    manifest = load_manifest()
    geo_path = GLOBAL_DIR / "geometry.json"
    if not geo_path.exists():
        print("run `manifolds interference` first", file=sys.stderr)
        return 1
    geometry = json.loads(geo_path.read_text(encoding="utf-8"))
    fm = _full_model(args, cfg)
    for name in _select(args):
        if not (RESULTS_DIR / name / "activations.npy").exists():
            continue
        if (RESULTS_DIR / name / "crosstalk.json").exists() and not args.force:
            print(f"{name}: crosstalk.json exists, skipping (use --force)")
            continue
        run_crosstalk(manifest.get(name), cfg, fm, geometry, max_prompts=args.max_prompts,
                      provider=args.provider, instance=args.instance)
    return 0


def cmd_checkpoints(args: argparse.Namespace) -> int:
    from manifolds.checkpoints import run_checkpoints

    cfg = load_config(args.config)
    manifest = load_manifest()
    names = _select(args)
    summary = run_checkpoints(manifest, cfg, names, device=args.device, batch_size=args.batch_size,
                              max_prompts=args.max_prompts, max_points=args.max_points,
                              provider=args.provider, instance=args.instance,
                              workers=args.workers, recompute=args.recompute_metrics)
    print(json.dumps(summary["display_layers"]))
    return 0


def cmd_steer(args: argparse.Namespace) -> int:
    import time

    import numpy as np
    import torch

    from manifolds import ledger
    from manifolds.analysis.pca import filter_norm_outliers
    from manifolds.depth_report import steer_figure
    from manifolds.prompts.types import PromptSet
    from manifolds.steer import STEER_TASKS, run_steering

    cfg = load_config(args.config)
    manifest = load_manifest()
    fm = _full_model(args, cfg)
    for name in _select(args):
        if name not in STEER_TASKS:
            print(f"{name}: no steering task, skipping")
            continue
        d = RESULTS_DIR / name
        ps = PromptSet.load(d / "prompts.json")
        acts = np.load(d / "activations.npy")
        idx = np.flatnonzero(filter_norm_outliers(acts, cfg.analysis.outlier_n_std))
        labels = [ps.labels[i] for i in idx]
        prompts = [ps.prompts[i] for i in idx]
        spans = [ps.spans[i] for i in idx] if ps.spans is not None else None
        spec = manifest.get(name)
        layer = spec.layer if isinstance(spec.layer, int) else cfg.model.layer
        start, t0 = ledger.now_iso(), time.time()
        res = run_steering(fm, name, acts[idx], labels, prompts, layer=layer, spans=spans).to_dict()
        from manifolds.provenance import stamp

        res["provenance"] = stamp(fm)
        (d / "steer_figure.json").write_text(steer_figure(res).to_json(), encoding="utf-8")
        dj_path = d / "depth.json"
        if dj_path.exists():
            dj = json.loads(dj_path.read_text(encoding="utf-8"))
            dj["steering"] = res
            dj_path.write_text(json.dumps(dj), encoding="utf-8")
        if torch.cuda.is_available():
            ledger.append(ledger.LedgerEntry(args.provider, args.instance,
                                             torch.cuda.get_device_name(0), start,
                                             ledger.now_iso(), round((time.time() - t0) / 3600, 4),
                                             f"steering {name}"))
        print(f"{name:18s} steered at {res['extras']['steered_position']}: "
              f"{res['base_prompt']!r}")
    return 0


def cmd_site(args: argparse.Namespace) -> int:
    from manifolds.site.build import build_site

    n = build_site(RESULTS_DIR, Path(args.out))
    print(f"site built with {n} manifold pages -> {args.out}")
    return 0


def cmd_ledger(args: argparse.Namespace) -> int:
    from manifolds import ledger

    if args.ledger_cmd == "summary":
        for k, v in ledger.summarize().items():
            print(f"{k:8s} {v:8.3f} GPU-h")
        return 0
    if args.ledger_cmd == "merge":
        n = ledger.merge(Path(args.src))
        print(f"merged {n} new ledger entries")
        return 0
    if args.ledger_cmd == "add":
        hours = args.hours if args.hours is not None else ledger.hours_between(args.start, args.end)
        ledger.append(ledger.LedgerEntry(args.provider, args.instance, args.gpu, args.start,
                                         args.end, hours, args.purpose, args.notes))
        print(f"logged {hours:.3f} GPU-h")
        return 0
    return 1


def cmd_status(_args: argparse.Namespace) -> int:
    manifest = load_manifest()
    status_path = RESULTS_DIR / "status.json"
    statuses = json.loads(status_path.read_text()) if status_path.exists() else {}
    print(f"{'manifold':24s} {'tier':9s} {'geometry':28s} {'status':8s} reason")
    for m in manifest.manifolds:
        s = statuses.get(m.name, {"status": "pending", "reason": ""})
        print(f"{m.name:24s} {m.tier:9s} {m.expected_geometry[:28]:28s} "
              f"{s['status']:8s} {s['reason'][:60]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="manifolds")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_select(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--manifold", nargs="*", help="manifold names (default: all with builders)")
        sp.add_argument("--tier", choices=["core", "extended", "cyclic", "vision"])
        sp.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))

    sp = sub.add_parser("prompts", help="build and save prompt sets")
    add_select(sp)
    sp.set_defaults(fn=cmd_prompts)
    sp = sub.add_parser("extract", help="extract layer activations")
    add_select(sp)
    sp.add_argument("--device", default="auto")
    sp.add_argument("--provider", default="unknown")
    sp.add_argument("--instance", default="unknown")
    sp.add_argument("--precision-check", action="store_true")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(fn=cmd_extract)
    sp = sub.add_parser("analyze", help="PCA + shape checks + figures")
    add_select(sp)
    sp.set_defaults(fn=cmd_analyze)
    sp = sub.add_parser("sae", help="SAE tiling analysis with Llama Scope L19R-8x")
    add_select(sp)
    sp.add_argument("--device", default="auto")
    sp.add_argument("--provider", default="unknown")
    sp.add_argument("--instance", default="unknown")
    sp.add_argument("--max-k", type=int, default=64)
    sp.add_argument("--max-points", type=int, default=2000)
    sp.set_defaults(fn=cmd_sae)
    sp = sub.add_parser("depth", help="all-layer sweep + logit/tuned lens + steering (full model)")
    add_select(sp)
    sp.add_argument("--device", default="auto")
    sp.add_argument("--provider", default="unknown")
    sp.add_argument("--instance", default="unknown")
    sp.add_argument("--batch-size", type=int, default=8)
    sp.add_argument("--gpu-gib", type=float, default=12.0)
    sp.add_argument("--max-points", type=int, default=800)
    sp.add_argument("--no-tuned-lens", action="store_true")
    sp.add_argument("--no-jlens", action="store_true")
    sp.add_argument("--no-steer", action="store_true")
    sp.add_argument("--figure-only", action="store_true",
                    help="rebuild only the layer-slider figure (probe view for probe manifolds)")
    sp.add_argument("--dtype", default="float16")
    sp.add_argument("--recapture", action="store_true",
                    help="with --figure-only: capture the residual states again")
    sp.set_defaults(fn=cmd_depth)

    def add_gpu(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--device", default="auto")
        sp.add_argument("--dtype", default="bfloat16")
        sp.add_argument("--provider", default="unknown")
        sp.add_argument("--instance", default="unknown")
        sp.add_argument("--batch-size", type=int, default=16)
        sp.add_argument("--gpu-gib", type=float, default=44.0)
        sp.add_argument("--force", action="store_true")
        sp.add_argument("--workers", type=int, default=1,
                        help="processes for the CPU-bound shape checks")

    sp = sub.add_parser("ablate",
                        help="weak-activation ablation sweep (all MLP blocks) vs geometry")
    add_select(sp)
    add_gpu(sp)
    sp.add_argument("--max-points", type=int, default=800)
    sp.add_argument("--n-heldout", type=int, default=64)
    sp.add_argument("--n-popqa", type=int, default=300)
    sp.add_argument("--lm-batch", type=int, default=4)
    sp.set_defaults(fn=cmd_ablate)
    sp = sub.add_parser("neurons",
                        help="MLP neurons that write / read the manifold + causal ablation")
    add_select(sp)
    add_gpu(sp)
    sp.add_argument("--max-prompts", type=int, default=1000)
    sp.add_argument("--max-points", type=int, default=800)
    sp.set_defaults(fn=cmd_neurons)
    sp = sub.add_parser("interference", help="subspace overlap between manifolds (no GPU)")
    add_select(sp)
    sp.add_argument("--k", type=int, default=3)
    sp.set_defaults(fn=cmd_interference)
    sp = sub.add_parser("crosstalk", help="prefix sentences from other domains and re-read")
    add_select(sp)
    add_gpu(sp)
    sp.add_argument("--max-prompts", type=int, default=800)
    sp.set_defaults(fn=cmd_crosstalk)
    sp = sub.add_parser("checkpoints", help="OLMo-2 1B training-checkpoint sweep")
    add_select(sp)
    add_gpu(sp)
    sp.add_argument("--max-prompts", type=int, default=600)
    sp.add_argument("--max-points", type=int, default=600)
    sp.add_argument("--recompute-metrics", action="store_true",
                    help="recompute cached per-checkpoint shape metrics")
    sp.set_defaults(fn=cmd_checkpoints)
    sp = sub.add_parser("steer", help="manifold steering only (writes steer_figure.json)")
    add_select(sp)
    add_gpu(sp)
    sp.set_defaults(fn=cmd_steer)
    sp = sub.add_parser("site", help="build the static site")
    sp.add_argument("--out", default=str(SITE_DIR))
    sp.set_defaults(fn=cmd_site)
    sp = sub.add_parser("ledger", help="GPU-hour ledger")
    lsub = sp.add_subparsers(dest="ledger_cmd", required=True)
    lsub.add_parser("summary")
    lm = lsub.add_parser("merge")
    lm.add_argument("--from", dest="src", required=True)
    la = lsub.add_parser("add")
    for a in ("provider", "instance", "gpu", "start", "end", "purpose"):
        la.add_argument(f"--{a}", required=True)
    la.add_argument("--hours", type=float, default=None)
    la.add_argument("--notes", default="")
    sp.set_defaults(fn=cmd_ledger)
    sp = sub.add_parser("status", help="print replication status table")
    sp.set_defaults(fn=cmd_status)

    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
