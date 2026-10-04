"""Contexte émergent + séquences symboliques : flux continu sans étiquette."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from core.symbolic import FAMILIES, PRIMITIVES, ambiguity_rate
from core.symbolic_lifelong import (
    few_shot_curve, label_ablation, pairs_to_disambiguate, routing_scores, run_ablations, run_stream,
    sensitivity, summarize, token_scores, zero_shot_composition,
)


def main(quick=False):
    print("=== WCE EMERGENT CONTEXT + SYMBOLIC SEQUENCES ===")
    print(f"\nAlphabet: {''.join(sorted(set('abcdefgh')))} | length 5 | {len(FAMILIES)} families | {len(PRIMITIVES)} WCE primitives")
    amb = ambiguity_rate(FAMILIES)
    print(f"Overlap: input alone ambiguous for {amb['input_only']:.0%} of samples (same input distribution for all families)")
    print(f"         one (x, y) pair shared by >= 2 families: {amb['single_pair']:.1%}; "
          f"pairs needed to disambiguate: {pairs_to_disambiguate()}")

    run = run_stream()
    s = run.system
    summary = summarize(run)
    print(f"\nUnlabeled stream: {len(s.online)} experiences, {len(run.segments)} regimes (no task label given)")
    print("\nEmergent contexts (families shown for evaluation only):")
    for cid, ctx in s.contexts.items():
        fam = ctx.families.most_common(1)[0][0] if ctx.families else "?"
        how = "zero-shot composition" if ctx.skill.kind in ("composition", "hybrid") and not ctx.skill.trained else \
              {"primitive": "structural induction", "neural": "new neural expert", "refinement": "few-shot refinement",
               "fast": "fast weights"}.get(ctx.skill.kind, ctx.skill.kind)
        print(f"  {cid:<6} skill={ctx.skill.name:<16} {how:<22} examples={ctx.skill.examples:<4} ~ {fam}")
    print("\nStream events:")
    for e in s.events:
        if e["event"] == "known_context":
            print(f"  step {e['step']:>4}: return to known context {e['context']} ({e['skill']}) after {e['examples']} pairs")
    cp = summary["change_points"]
    print(f"\nChange-point detection: precision {cp['precision']} recall {cp['recall']} F1 {cp['f1']} "
          f"mean delay {cp['mean_delay']} experience(s)")
    print(f"Novelty (new vs known regime): {summary['novelty_accuracy']}")

    print("\nZero-shot composition (Reverse, RotateLeft learned separately):")
    z = zero_shot_composition()
    w = z["wce_composition"]
    print(f"  WCE: skill {w['skill']} from {w['examples']} examples, gradient used: {w['trained']}, accuracy {w['accuracy']:.3f}")
    print(f"  network alone, 3 examples: {z['network_alone_3_examples']:.3f}; 128 examples: {z['network_alone_128_examples']:.3f}")

    print("\nFew-shot new family map_A2 (map_A with two images swapped):")
    print("  shots  WCE-refinement  fast-weights  new-expert")
    for r in few_shot_curve():
        print(f"  {r['shots']:>5}  {r['wce_refinement']:>14.3f}  {r['fast_weights']:>12.3f}  {r['new_expert']:>10.3f}")
    ctx = next(c for c in s.contexts.values() if c.skill.kind == "refinement")
    print(f"  in the stream: MetaController chose {ctx.skill.name} after {ctx.skill.examples} examples")

    print("\nFinal accuracy without labels (3 support pairs):")
    print("  " + ", ".join(f"{f}={a:.2f}" for f, a in run.final_accuracy.items()))
    print(f"  mean {summary['final_accuracy']}, max forgetting {summary['max_forgetting']}")

    print("\nHybrid router (identification from ONE pair, no label):")
    r = routing_scores(s, sorted({seg.family for seg in run.segments}))
    print(f"  neural only {r['neural']} | structural only {r['structural']} ({r['structural_evaluations']} checks)"
          f" | hybrid {r['hybrid']} ({r['hybrid_evaluations']} checks)")

    print("\nCommands (progressive removal of explicit tokens):")
    for token, t in token_scores(s).items():
        print(f"  {token:<5} -> {t['family']:<9} token only {t['token_only']:.2f} | no token, no support {t['no_token_no_support']:.2f}")

    print("\nCapacity curve (families learned vs parameters):")
    print("  contexts  experts  total_params  active_params")
    for c in s.capacity_curve:
        print(f"  {c['contexts']:>8}  {c['experts']:>7}  {c['total_parameters']:>12}  {c['active_parameters']:>13}")
    print(f"  tasks/experts {summary['tasks_per_expert']} | skills/primitives {summary['skills_per_primitive']} | "
          f"zero-shot compositions {summary['zero_shot_compositions']}")
    print(f"\nMemory used by successes: {summary['memory_usage']}")
    print(f"Snapshots: incremental {summary['snapshot_bytes_incremental']} bytes vs full {summary['snapshot_bytes_full']} bytes")
    print(f"Context embedding stability: {summary['embedding_stability']} | cluster purity (eval): {summary['cluster_purity']}")

    if quick: return
    print("\nTask-label ablation (A labels / B boundaries only / C nothing):")
    for mode, res in label_ablation().items(): print(f"  {mode:<10} {res}")
    print("\nArchitecture ablations (same stream):")
    for name, res in run_ablations().items():
        print(f"  {name:<30} final {res['final_accuracy']:.3f} online {res['online_accuracy']:.3f} "
              f"params {res['total_parameters']} experts {res['experts']}")
    print("\nThreshold sensitivity (reduced stream, one parameter at a time):")
    for param, rows in sensitivity().items():
        print(f"  {param}: " + " | ".join(f"{r['value']}: acc {r['final_accuracy']:.2f} ctx {r['contexts']} "
                                         f"exp {r['experts']} F1 {r['f1']}{' *' if r['stable'] else ''}" for r in rows))
    print("  (* = stable region: within 0.02 of best final accuracy)")


if __name__ == "__main__":
    torch.manual_seed(0)
    main(quick="--quick" in sys.argv)
