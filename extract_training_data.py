"""
Extract training pairs from iteration history for SFT fine-tuning.

This script reads your project's iteration results (configs, diagnostics,
decisions) and produces structured SFT training pairs in chat format.

Each pair follows the pattern:
  System: You are a domain optimization expert...
  User: Here's the previous config and results. What should we change?
  Assistant: ## Approach ... ## Reasoning ... ## Config Changes ...

Usage:
  python extract_training_data.py \
    --iterations-dir ./iterations \
    --output ./data/sft_train.jsonl

The iterations directory should contain JSON files with:
  - config: the configuration used
  - results: evaluation metrics
  - diagnosis: what went wrong
  - changes: what was changed for the next iteration
"""

import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a domain optimization expert. When given a configuration and "
    "evaluation results, analyze what went wrong and propose specific parameter "
    "changes to improve performance. Structure your response with: "
    "## Approach, ## Reasoning, ## Config Changes."
)


def format_config(config: dict) -> str:
    """Format a config dict into a readable string."""
    lines = []
    for key, value in sorted(config.items()):
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def format_results(results: dict) -> str:
    """Format evaluation results into a readable string."""
    lines = []
    for metric, info in results.items():
        if isinstance(info, dict):
            value = info.get("value", "N/A")
            threshold = info.get("threshold", "")
            status = info.get("status", "")
            line = f"- {metric}: {value}"
            if threshold:
                line += f" (threshold: {threshold}"
                if status:
                    line += f", {status}"
                line += ")"
            lines.append(line)
        else:
            lines.append(f"- {metric}: {info}")
    return "\n".join(lines)


def build_user_message(iteration: dict) -> str:
    """Build the user message from an iteration record."""
    parts = []

    if "config" in iteration:
        parts.append("Current config:")
        parts.append(format_config(iteration["config"]))

    if "results" in iteration:
        parts.append("\nEvaluation results:")
        parts.append(format_results(iteration["results"]))

    if "diagnosis" in iteration:
        parts.append(f"\nDiagnosis: {iteration['diagnosis']}")

    parts.append("\nWhat changes would improve this?")
    return "\n".join(parts)


def build_assistant_message(iteration: dict) -> str:
    """Build the assistant response from an iteration record."""
    parts = []

    if "approach" in iteration:
        parts.append(f"## Approach\n\n{iteration['approach']}")

    if "reasoning" in iteration:
        parts.append(f"## Reasoning\n\n{iteration['reasoning']}")

    if "changes" in iteration:
        parts.append("## Config Changes\n")
        for change in iteration["changes"]:
            param = change.get("param", "unknown")
            old_val = change.get("from", "?")
            new_val = change.get("to", "?")
            reason = change.get("reason", "")
            line = f"- `{param}`: {old_val} -> **{new_val}**"
            if reason:
                line += f" ({reason})"
            parts.append(line)

    return "\n\n".join(parts)


def process_iteration(iteration: dict) -> dict | None:
    """Convert a single iteration record into an SFT training example."""
    user_msg = build_user_message(iteration)
    assistant_msg = build_assistant_message(iteration)

    if not assistant_msg.strip():
        return None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract SFT training pairs from iteration history"
    )
    parser.add_argument(
        "--iterations-dir", type=str, required=True,
        help="Directory containing iteration JSON files"
    )
    parser.add_argument(
        "--output", type=str, default="./data/sft_train.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--min-changes", type=int, default=1,
        help="Minimum number of config changes to include an iteration"
    )
    args = parser.parse_args()

    iterations_dir = Path(args.iterations_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find iteration files
    iter_files = sorted(iterations_dir.glob("*.json"))
    if not iter_files:
        print(f"No JSON files found in {iterations_dir}")
        return

    print(f"Found {len(iter_files)} iteration files")

    # Process each iteration
    examples = []
    for iter_file in iter_files:
        with open(iter_file) as f:
            iteration = json.load(f)

        # Skip iterations without enough changes
        changes = iteration.get("changes", [])
        if len(changes) < args.min_changes:
            continue

        example = process_iteration(iteration)
        if example:
            examples.append(example)

    # Write output
    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"Wrote {len(examples)} training examples to {output_path}")
    print(f"Skipped {len(iter_files) - len(examples)} iterations (insufficient data)")


if __name__ == "__main__":
    main()
