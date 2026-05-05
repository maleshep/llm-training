"""
Augment SFT training data from 12 examples to 50+ via deterministic transformations.

Strategies:
  1. Paraphrase user prompts (vary wording, order of diagnostics, emphasis)
  2. Context window variations (include different amounts of history)
  3. Ablated versions (remove some metrics to teach robustness)

Input:  data/sft_train.jsonl (12 examples)
Output: data/sft_train_augmented.jsonl (50+ examples)

Usage:
  python scripts/augment_training_data.py
  python scripts/augment_training_data.py --input data/sft_train.jsonl --output data/sft_train_augmented.jsonl
"""

import argparse
import copy
import json
import random
import re
from pathlib import Path


# --- Paraphrase templates for user prompts ---

PROMPT_PREFIXES = [
    "Based on the diagnostic context below, propose the next model configuration changes to improve the MMM. Provide evidence-grounded reasoning.",
    "Given the current model state and diagnostics below, recommend configuration adjustments to pass all evaluation gates. Justify with evidence.",
    "Review the diagnostic information below and suggest parameter changes for the next MMM iteration. Support your reasoning with quantitative evidence.",
    "Analyze the diagnostics below and propose concrete configuration changes to improve model fit and pass gates. Use evidence-based reasoning.",
    "The MMM diagnostics are shown below. What configuration changes would you propose to improve performance? Provide evidence for each change.",
]

CONSTRAINT_PHRASINGS = [
    "Constraints: email_metric=opens, 13 GERS sectors, pymc-marketing 0.18.0, date_start=2023-01-01",
    "Remember: email uses opens metric, 13 GERS sectors, pymc-marketing pinned to 0.18.0, data starts 2023-01-01.",
    "Locked: opens as email metric, geographic_grouping=sector (13), pymc-marketing==0.18.0, window from 2023-01-01.",
    "Fixed constraints: email_metric=opens | 13 GERS sectors | pymc-marketing 0.18.0 | start=2023-01-01",
    "Do not change: email_metric (opens), sectors (13 GERS), pymc-marketing version (0.18.0), date window (2023-01-01+).",
]

DIAGNOSTIC_INTROS = [
    "Previous iteration ({approach}):",
    "Last run ({approach}) results:",
    "Prior attempt — {approach}:",
    "Results from {approach}:",
    "{approach} produced:",
]


def paraphrase_user_prompt(original_user: str, seed: int) -> str:
    """Create a paraphrased version of the user prompt."""
    rng = random.Random(seed)

    # Replace the standard prefix
    new_prefix = rng.choice(PROMPT_PREFIXES)

    # Replace constraint phrasing
    new_constraints = rng.choice(CONSTRAINT_PHRASINGS)

    result = original_user

    # Replace the opening instruction line
    first_line_pattern = r"^Based on the diagnostic context below.*?reasoning\."
    if re.search(first_line_pattern, result, re.MULTILINE):
        result = re.sub(first_line_pattern, new_prefix, result, count=1)

    # Replace constraints line
    constraint_pattern = r"Constraints:.*?date_start=2023-01-01"
    if re.search(constraint_pattern, result):
        result = re.sub(constraint_pattern, new_constraints, result)

    return result


def reorder_diagnostics(user_content: str, seed: int) -> str:
    """Reorder diagnostic sections in the user prompt."""
    rng = random.Random(seed)

    # Split into lines, identify diagnostic blocks, shuffle them
    lines = user_content.split("\n")
    prefix_lines = []
    diag_blocks = []
    current_block = []
    suffix_lines = []
    in_diag = False
    past_diag = False

    for line in lines:
        if not in_diag and not past_diag:
            if line.strip().startswith(("Previous iteration", "Version ", "Last run", "Prior attempt", "Results from")):
                in_diag = True
                current_block.append(line)
            else:
                prefix_lines.append(line)
        elif in_diag:
            if line.strip().startswith(("Constraints:", "Remember:", "Locked:", "Fixed constraints:", "Do not change:")):
                if current_block:
                    diag_blocks.append(current_block)
                    current_block = []
                in_diag = False
                past_diag = True
                suffix_lines.append(line)
            elif line.strip() == "" and current_block:
                diag_blocks.append(current_block)
                current_block = []
            else:
                current_block.append(line)
        else:
            suffix_lines.append(line)

    if current_block:
        diag_blocks.append(current_block)

    # Shuffle diagnostic blocks
    rng.shuffle(diag_blocks)

    result_lines = prefix_lines
    for block in diag_blocks:
        result_lines.extend(block)
        result_lines.append("")
    result_lines.extend(suffix_lines)

    return "\n".join(result_lines)


def ablate_metrics(user_content: str, seed: int) -> str:
    """Remove some metrics from the diagnostic context to teach robustness."""
    rng = random.Random(seed)

    # Metrics that can be selectively removed
    removable_patterns = [
        r"  Trust:.*?\n",
        r"  ESS bulk:.*?\n",
        r"  Divergences:.*?\n",
        r"  Email attr:.*?\n",
        r"  F2F plausibility:.*?\n",
        r"  R²:.*?\n",
        r"  All gates pass:.*?\n",
        r"  Gate failures:.*?\n",
    ]

    result = user_content
    # Remove 1-3 metrics randomly
    n_remove = rng.randint(1, min(3, len(removable_patterns)))
    to_remove = rng.sample(removable_patterns, n_remove)
    for pattern in to_remove:
        result = re.sub(pattern, "", result)

    return result


def add_context_history(user_content: str, seed: int) -> str:
    """Add synthetic context history snippets."""
    rng = random.Random(seed)

    history_snippets = [
        "\nHistory: 3 prior iterations tested adstock decay, saturation, and prior calibration.",
        "\nNote: Previous runs showed uniform betas until pymc-marketing 0.18.0 was pinned.",
        "\nContext: Model has been iterating for 2 weeks. Key breakthrough was switching to 0.18.0.",
        "\nBackground: F2F attribution was stuck at <5% for first 4 iterations due to degenerate Gamma priors.",
        "\nPrior attempts: Shared hyperpriors failed (uniform betas). Per-channel priors with sigma=ratio*mu fixed it.",
        "\nIteration history: Started with 0.19.x (broken F2F), moved to 0.18.0, then tuned channel priors.",
    ]

    snippet = rng.choice(history_snippets)

    # Insert before constraints line
    constraint_markers = ["Constraints:", "Remember:", "Locked:", "Fixed constraints:", "Do not change:"]
    for marker in constraint_markers:
        if marker in user_content:
            return user_content.replace(marker, snippet + "\n" + marker)

    return user_content + snippet


def vary_system_prompt(system_content: str, seed: int) -> str:
    """Slightly vary the system prompt for robustness."""
    rng = random.Random(seed)

    # Minor variations that preserve semantics
    variations = [
        # Add emphasis on different aspects
        (r"You are an expert", "You are a specialist"),
        (r"evidence-grounded model configuration changes", "evidence-based parameter adjustments"),
        (r"improve fit and pass evaluation gates", "improve model fit and satisfy all evaluation criteria"),
    ]

    result = system_content
    # Apply 0-1 variations
    if rng.random() < 0.4:
        old, new = rng.choice(variations)
        result = re.sub(old, new, result, count=1)

    return result


def augment_examples(examples: list[dict], target_count: int = 55) -> list[dict]:
    """Augment training examples to reach target_count."""
    augmented = []

    # Keep all originals
    augmented.extend(copy.deepcopy(examples))

    # Generate augmentations round-robin across strategies
    strategies = [
        ("paraphrase", lambda ex, s: _apply_paraphrase(ex, s)),
        ("reorder", lambda ex, s: _apply_reorder(ex, s)),
        ("ablate", lambda ex, s: _apply_ablate(ex, s)),
        ("context", lambda ex, s: _apply_context(ex, s)),
        ("combined", lambda ex, s: _apply_combined(ex, s)),
    ]

    seed = 42
    strategy_idx = 0

    while len(augmented) < target_count:
        for ex in examples:
            if len(augmented) >= target_count:
                break

            strategy_name, strategy_fn = strategies[strategy_idx % len(strategies)]
            new_ex = strategy_fn(copy.deepcopy(ex), seed)
            if new_ex:
                # Tag augmentation in metadata
                meta = new_ex.get("metadata", {})
                meta["augmentation"] = strategy_name
                meta["aug_seed"] = seed
                new_ex["metadata"] = meta
                augmented.append(new_ex)

            seed += 1
            strategy_idx += 1

    return augmented


def _apply_paraphrase(ex: dict, seed: int) -> dict:
    """Apply prompt paraphrasing."""
    messages = ex["messages"]
    messages[0]["content"] = vary_system_prompt(messages[0]["content"], seed)
    messages[1]["content"] = paraphrase_user_prompt(messages[1]["content"], seed)
    return ex


def _apply_reorder(ex: dict, seed: int) -> dict:
    """Apply diagnostic reordering."""
    messages = ex["messages"]
    messages[1]["content"] = reorder_diagnostics(messages[1]["content"], seed)
    return ex


def _apply_ablate(ex: dict, seed: int) -> dict:
    """Apply metric ablation."""
    messages = ex["messages"]
    messages[1]["content"] = ablate_metrics(messages[1]["content"], seed)
    return ex


def _apply_context(ex: dict, seed: int) -> dict:
    """Apply context history addition."""
    messages = ex["messages"]
    messages[1]["content"] = add_context_history(messages[1]["content"], seed)
    return ex


def _apply_combined(ex: dict, seed: int) -> dict:
    """Apply multiple augmentations together."""
    messages = ex["messages"]
    messages[0]["content"] = vary_system_prompt(messages[0]["content"], seed)
    messages[1]["content"] = paraphrase_user_prompt(messages[1]["content"], seed)
    messages[1]["content"] = ablate_metrics(messages[1]["content"], seed + 1)
    messages[1]["content"] = add_context_history(messages[1]["content"], seed + 2)
    return ex


def main():
    parser = argparse.ArgumentParser(description="Augment SFT training data")
    parser.add_argument("--input", type=str, default="data/sft_train.jsonl",
                        help="Input JSONL file")
    parser.add_argument("--output", type=str, default="data/sft_train_augmented.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--target", type=int, default=55,
                        help="Target number of augmented examples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    input_path = project_root / args.input
    output_path = project_root / args.output

    # Load original data
    print(f"Loading from: {input_path}")
    examples = []
    with open(input_path) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"  Original examples: {len(examples)}")

    # Augment
    random.seed(args.seed)
    augmented = augment_examples(examples, target_count=args.target)
    print(f"  Augmented examples: {len(augmented)}")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in augmented:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  Written to: {output_path}")

    # Stats
    strategy_counts = {}
    for ex in augmented:
        aug = ex.get("metadata", {}).get("augmentation", "original")
        strategy_counts[aug] = strategy_counts.get(aug, 0) + 1
    print("\n  Augmentation breakdown:")
    for strategy, count in sorted(strategy_counts.items()):
        print(f"    {strategy}: {count}")


if __name__ == "__main__":
    main()
