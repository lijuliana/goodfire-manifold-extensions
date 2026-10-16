"""Command-line entry point: prompts | extract | analyze | site | ledger | status."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from manifolds import RESULTS_DIR, SITE_DIR
from manifolds.config import DEFAULT_CONFIG_PATH, load_config
from manifolds.manifest import load_manifest


def _select(args: argparse.Namespace) -> list[str]:
    from manifolds.prompts import BUILDERS

    manifest = load_manifest()
    if getattr(args, "manifold", None):
        names = list(args.manifold)
    elif getattr(args, "tier", None):
        names = [m.name for m in manifest.by_tier(args.tier)]
    else:
        names = [m.name for m in manifest.manifolds if m.name in BUILDERS]
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
        print(f"{name}: extracting {len(ps)} prompts")
        extract_and_cache(ps, ext, out_dir, provider=args.provider, instance=args.instance)
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
    sae = LlamaScopeSAE.from_hf(device=device)
    print(json.dumps(sae.info, indent=1))
    for name in _select(args):
        if not (RESULTS_DIR / name / "activations.npy").exists():
            print(f"{name}: no activations, skipping", file=sys.stderr)
            continue
        rep = analyze_sae(manifest.get(name), cfg, sae, max_k=args.max_k,
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
                               dtype=cfg.model.dtype, device=args.device,
                               batch_size=args.batch_size, gpu_memory_gib=args.gpu_gib,
                               seed=cfg.seed))
    print(f"full model {fm.model_id} ({fm.resolution}), {fm.n_layers} blocks")
    tuned = None if args.no_tuned_lens else TunedLens.from_hf(device="cpu", dtype=torch.float32)
    jlens = None
    if not args.no_jlens:
        from manifolds.jlens import JacobianLensFull

        jlens = JacobianLensFull.from_hf()
        print(f"jacobian lens: {jlens.info.get('repo')} ({jlens.n_prompts} prompts)")
    for name in _select(args):
        if not (RESULTS_DIR / name / "activations.npy").exists():
            print(f"{name}: no activations, skipping", file=sys.stderr)
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
    sp.set_defaults(fn=cmd_depth)
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
