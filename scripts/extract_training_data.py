"""
Extract SFT and DPO training pairs from Meta-Harness iteration history.

Source data:
  - tools/meta-harness/registry.json        (version scores, gates, attribution)
  - tools/meta-harness/versions/iteration_log.json  (full reasoning, diagnoses, learnings)
  - tools/meta-harness/versions/*/reasoning.md       (per-version reasoning)
  - tools/meta-harness/versions/*/config.yaml        (per-version model config)

Output:
  - model_training/hpc/llm/data/sft_train.jsonl      (instruction-completion pairs)
  - model_training/hpc/llm/data/dpo_train.jsonl       (chosen-rejected pairs)
  - model_training/hpc/llm/data/stats.json             (dataset statistics)

Usage:
  python extract_training_data.py --mmm-root ../../marketing-mix
  # On HPC:
  python extract_training_data.py --mmm-root /shared/project/tdr-mmm-hpc/marketing-mix
"""

import argparse
import json
import os
import yaml
from pathlib import Path


# --- Constants ---

SYSTEM_PROMPT = (
    "You are an expert Bayesian Marketing Mix Modelling (MMM) optimization agent. "
    "You help optimize hierarchical PyMC models for pharmaceutical channel attribution. "
    "Given diagnostic context (prior results, gate failures, constraints), you propose "
    "evidence-grounded model configuration changes that improve fit and pass evaluation gates.\n\n"
    "Evaluation gates:\n"
    "- Email attribution < 30%\n"
    "- F2F plausibility: green <=75%, yellow <=85%, red >85%\n"
    "- R-squared > 0.85\n"
    "- Divergences = 0\n"
    "- Trust score > 50\n"
    "- ESS bulk min > 100\n\n"
    "Locked parameters: email_metric=opens, geographic_grouping=sector (13 GERS), "
    "date_window_start=2023-01-01, channels_excluded=[printed_materials].\n\n"
    "pymc-marketing MUST be pinned to 0.18.0 (0.19.x breaks F2F via PR #2293)."
)

GATE_THRESHOLDS = {
    "email_attribution_pct": 30.0,
    "f2f_plausibility_pct": 75.0,
    "r_squared": 0.85,
    "divergences": 0,
    "trust_score": 50,
    "ess_bulk_min": 100,
}


def load_registry(mmm_root: Path) -> list:
    """Load registry.json — all version records."""
    path = mmm_root / "tools" / "meta-harness" / "registry.json"
    with open(path) as f:
        return json.load(f)


def load_iteration_log(mmm_root: Path) -> dict:
    """Load iteration_log.json — detailed iteration history."""
    path = mmm_root / "tools" / "meta-harness" / "versions" / "iteration_log.json"
    with open(path) as f:
        return json.load(f)


def load_version_reasoning(mmm_root: Path, version_id: str) -> str | None:
    """Load reasoning.md for a specific version."""
    path = mmm_root / "tools" / "meta-harness" / "versions" / version_id / "reasoning.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def load_version_config(mmm_root: Path, version_id: str) -> dict | None:
    """Load config.yaml for a specific version."""
    path = mmm_root / "tools" / "meta-harness" / "versions" / version_id / "config.yaml"
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return None


def version_passes_all_gates(version: dict) -> bool:
    """Check if a registry version passes all 6 hard gates."""
    scores = version.get("scores", {})
    gates = version.get("gates", {})

    checks = [
        gates.get("email_attribution_pct") is not None and gates["email_attribution_pct"] < 30,
        gates.get("f2f_plausibility_pct") is not None and gates["f2f_plausibility_pct"] <= 75,
        scores.get("r2") is not None and scores["r2"] > 0.85,
        scores.get("divergences") is not None and scores["divergences"] == 0,
        scores.get("trust_score") is not None and scores["trust_score"] > 50,
        scores.get("ess_bulk_min") is not None and scores["ess_bulk_min"] > 100,
    ]
    return all(checks)


def format_version_summary(version: dict) -> str:
    """Format a version's key metrics into a readable summary."""
    vid = version.get("id", "?")
    scores = version.get("scores", {})
    gates = version.get("gates", {})
    attr = version.get("attribution", {})

    lines = [f"Version {vid}:"]
    if attr:
        lines.append(f"  Attribution: {json.dumps(attr, indent=None)}")
    if scores.get("r2"):
        lines.append(f"  R²: {scores['r2']}")
    if scores.get("trust_score"):
        lines.append(f"  Trust: {scores['trust_score']} ({scores.get('trust_grade', '?')})")
    if scores.get("ess_bulk_min"):
        lines.append(f"  ESS bulk: {scores['ess_bulk_min']}")
    if scores.get("divergences") is not None:
        lines.append(f"  Divergences: {scores['divergences']}")
    if gates.get("email_attribution_pct") is not None:
        lines.append(f"  Email attr: {gates['email_attribution_pct']}%")
    if gates.get("f2f_plausibility_pct") is not None:
        lines.append(f"  F2F plausibility: {gates['f2f_plausibility_pct']}%")

    gate_pass = version.get("gate_pass")
    if gate_pass is not None:
        lines.append(f"  All gates pass: {gate_pass}")
    failures = version.get("gate_failures", [])
    if failures:
        lines.append(f"  Gate failures: {', '.join(failures)}")

    return "\n".join(lines)


def build_sft_pairs(registry: list, iteration_log: dict, mmm_root: Path) -> list:
    """
    Build SFT training pairs from the iteration log.

    Each pair:
      - system: domain expert system prompt
      - user: diagnostic context (prior results, what failed, constraints)
      - assistant: the reasoning + proposed config changes + evidence
    """
    pairs = []
    iterations = iteration_log.get("iterations", [])

    # Build version lookup from registry
    version_map = {v["id"]: v for v in registry.get("versions", registry) if isinstance(v, dict) and "id" in v}

    for i, iteration in enumerate(iterations):
        iter_type = iteration.get("type", "model_run")

        # Skip pure diagnostic entries — they don't produce proposals
        if iter_type in ("diagnosis", "diagnosis_correction"):
            continue

        # --- Build diagnostic context (what the proposer sees) ---
        context_parts = []

        # Prior iteration results
        if i > 0:
            prev = iterations[i - 1]
            if "results" in prev:
                context_parts.append(f"Previous iteration ({prev.get('approach', '?')}):")
                for k, v in prev["results"].items():
                    context_parts.append(f"  {k}: {json.dumps(v)}")
            if "diagnosis" in prev:
                context_parts.append(f"Diagnosis: {prev['diagnosis']}")
            if "learning" in prev:
                learning = prev["learning"]
                if isinstance(learning, list):
                    context_parts.append("Learnings:\n" + "\n".join(f"  - {l}" for l in learning))
                else:
                    context_parts.append(f"Learning: {learning}")

        # Gate status from registry for parent versions
        versions = iteration.get("versions", [])
        for vid in versions:
            if vid in version_map:
                context_parts.append(format_version_summary(version_map[vid]))

        # Constraints reminder
        context_parts.append(
            "Constraints: email_metric=opens, 13 GERS sectors, "
            "pymc-marketing 0.18.0, date_start=2023-01-01"
        )

        user_msg = (
            "Based on the diagnostic context below, propose the next model configuration "
            "changes to improve the MMM. Provide evidence-grounded reasoning.\n\n"
            + "\n".join(context_parts)
        )

        # --- Build the assistant response (what we're training on) ---
        response_parts = []

        approach = iteration.get("approach", "")
        if approach:
            response_parts.append(f"## Approach\n{approach}\n")

        reasoning = iteration.get("proposer_reasoning", "")
        if reasoning:
            response_parts.append(f"## Reasoning\n{reasoning}\n")

        config_changes = iteration.get("config_changes", {})
        if config_changes:
            response_parts.append("## Config Changes")
            for vid, changes in config_changes.items():
                response_parts.append(f"\n### {vid}")
                if isinstance(changes, list):
                    for c in changes:
                        param = c.get("param", "?")
                        fr = c.get("from", "default")
                        to = c.get("to", c.get("value", "?"))
                        reason = c.get("reason", "")
                        line = f"- `{param}`: {fr} → {to}"
                        if reason:
                            line += f" ({reason})"
                        response_parts.append(line)
            response_parts.append("")

        # Include version-level reasoning if available
        for vid in versions:
            ver_reasoning = load_version_reasoning(mmm_root, vid)
            if ver_reasoning:
                response_parts.append(f"## {vid.upper()} Detailed Reasoning\n{ver_reasoning[:2000]}\n")

        # Results and diagnosis (the proposer learns from these)
        results = iteration.get("results", {})
        if results:
            response_parts.append("## Results")
            for k, v in results.items():
                response_parts.append(f"- {k}: {json.dumps(v)}")
            response_parts.append("")

        diagnosis = iteration.get("diagnosis", "")
        if diagnosis:
            response_parts.append(f"## Diagnosis\n{diagnosis}\n")

        learning = iteration.get("learning", "")
        if learning:
            if isinstance(learning, list):
                response_parts.append("## Key Learnings\n" + "\n".join(f"- {l}" for l in learning))
            else:
                response_parts.append(f"## Key Learning\n{learning}")

        assistant_msg = "\n".join(response_parts).strip()

        if user_msg and assistant_msg:
            pairs.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ],
                "metadata": {
                    "iteration": str(iteration.get("iteration", "")),
                    "approach": approach,
                    "versions": versions,
                    "type": "sft",
                },
            })

    return pairs


def build_dpo_pairs(registry: list, iteration_log: dict, mmm_root: Path) -> list:
    """
    Build DPO (preference) training pairs.

    Chosen: versions/approaches that passed gates or improved metrics
    Rejected: versions/approaches that failed gates or regressed

    Each pair shares the same prompt (diagnostic context) but has
    different responses (chosen = good outcome, rejected = bad outcome).
    """
    pairs = []
    iterations = iteration_log.get("iterations", [])

    # Identify good and bad iterations
    good_iterations = []  # Passed gates, improved metrics
    bad_iterations = []   # Failed gates, regressed

    for iteration in iterations:
        if iteration.get("type") in ("diagnosis", "diagnosis_correction", "data_investigation"):
            continue

        results = iteration.get("results", {})
        gates = iteration.get("gates", {})

        # Check if any version in this iteration passes all gates
        all_pass = False
        any_fail = False

        for vid, gate_data in gates.items():
            if isinstance(gate_data, dict):
                passes = all(
                    g.get("pass", False)
                    for g in gate_data.values()
                    if isinstance(g, dict) and "pass" in g
                )
                if passes:
                    all_pass = True
                else:
                    any_fail = True

        # Also check result-level indicators
        for rk, rv in results.items():
            if isinstance(rv, dict):
                trust = rv.get("trust", 100)
                ess = rv.get("ess", 1000)
                if trust < 50 or ess < 100:
                    any_fail = True

        if all_pass:
            good_iterations.append(iteration)
        elif any_fail:
            bad_iterations.append(iteration)

    # Build pairs: each good iteration paired with each bad iteration
    # that came from a similar diagnostic context
    for good in good_iterations:
        for bad in bad_iterations:
            # Build shared prompt (generic diagnostic context)
            prompt = (
                "You are optimizing a hierarchical Bayesian MMM for France Mavenclad. "
                "The model uses pymc-marketing 0.18.0 with 6 media channels across 13 GERS sectors. "
                "Propose configuration changes to pass all 6 evaluation gates: "
                "email <30%, F2F <=75%, R²>0.85, 0 divergences, trust>50, ESS>100.\n\n"
                "Current state: the model has been through multiple iterations of prior tuning. "
                "Key learnings: pymc-marketing 0.18.0 is essential (0.19.x breaks F2F), "
                "literature-grounded channel priors help, adequate sampling (tune>=500) is critical."
            )

            # Chosen response (from good iteration)
            chosen_parts = []
            chosen_parts.append(f"## Approach\n{good.get('approach', '?')}\n")
            chosen_parts.append(f"## Reasoning\n{good.get('proposer_reasoning', '?')}\n")
            config_changes = good.get("config_changes", {})
            if config_changes:
                chosen_parts.append("## Config Changes")
                for vid, changes in config_changes.items():
                    if isinstance(changes, list):
                        for c in changes:
                            chosen_parts.append(f"- `{c.get('param')}`: {c.get('from', '?')} → {c.get('to', c.get('value', '?'))}")
            chosen = "\n".join(chosen_parts)

            # Rejected response (from bad iteration)
            rejected_parts = []
            rejected_parts.append(f"## Approach\n{bad.get('approach', '?')}\n")
            rejected_parts.append(f"## Reasoning\n{bad.get('proposer_reasoning', '?')}\n")
            config_changes = bad.get("config_changes", {})
            if config_changes:
                rejected_parts.append("## Config Changes")
                for vid, changes in config_changes.items():
                    if isinstance(changes, list):
                        for c in changes:
                            rejected_parts.append(f"- `{c.get('param')}`: {c.get('from', '?')} → {c.get('to', c.get('value', '?'))}")
            rejected = "\n".join(rejected_parts)

            pairs.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
                "metadata": {
                    "chosen_iteration": good.get("iteration"),
                    "rejected_iteration": bad.get("iteration"),
                    "chosen_approach": good.get("approach"),
                    "rejected_approach": bad.get("approach"),
                    "type": "dpo",
                },
            })

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Extract MMM training data for LLM fine-tuning")
    parser.add_argument(
        "--mmm-root",
        type=str,
        default="../../marketing-mix",
        help="Path to marketing-mix repo root",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: model_training/hpc/llm/data/)",
    )
    args = parser.parse_args()

    mmm_root = Path(args.mmm_root).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"MMM root: {mmm_root}")
    print(f"Output dir: {output_dir}")

    # Load source data
    print("Loading registry...")
    registry = load_registry(mmm_root)

    print("Loading iteration log...")
    iteration_log = load_iteration_log(mmm_root)
    n_iterations = len(iteration_log.get("iterations", []))
    print(f"  {n_iterations} iterations found")

    # Build SFT pairs
    print("Building SFT pairs...")
    sft_pairs = build_sft_pairs(registry, iteration_log, mmm_root)
    print(f"  {len(sft_pairs)} SFT pairs generated")

    # Build DPO pairs
    print("Building DPO pairs...")
    dpo_pairs = build_dpo_pairs(registry, iteration_log, mmm_root)
    print(f"  {len(dpo_pairs)} DPO pairs generated")

    # Write outputs
    sft_path = output_dir / "sft_train.jsonl"
    with open(sft_path, "w", encoding="utf-8") as f:
        for pair in sft_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Wrote {sft_path}")

    dpo_path = output_dir / "dpo_train.jsonl"
    with open(dpo_path, "w", encoding="utf-8") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Wrote {dpo_path}")

    # Stats
    stats = {
        "source": str(mmm_root),
        "iterations_total": n_iterations,
        "sft_pairs": len(sft_pairs),
        "dpo_pairs": len(dpo_pairs),
        "sft_avg_tokens_est": sum(
            len(p["messages"][1]["content"].split()) + len(p["messages"][2]["content"].split())
            for p in sft_pairs
        ) // max(len(sft_pairs), 1) * 1.3,  # rough word->token estimate
        "dpo_avg_tokens_est": sum(
            len(p["chosen"][0]["content"].split()) + len(p["rejected"][0]["content"].split())
            for p in dpo_pairs
        ) // max(len(dpo_pairs), 1) * 1.3,
        "system_prompt_tokens_est": len(SYSTEM_PROMPT.split()) * 1.3,
    }
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {stats_path}")

    print("\nDone.")
    print(f"  SFT: {len(sft_pairs)} pairs -> {sft_path}")
    print(f"  DPO: {len(dpo_pairs)} pairs -> {dpo_path}")
    print(f"  Stats: {stats_path}")


if __name__ == "__main__":
    main()
