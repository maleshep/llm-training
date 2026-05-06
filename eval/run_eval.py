"""
Evaluation harness: Fine-tuned Qwen vs Sonnet 4.6, judged by Opus 4.6.

Runs each golden test prompt through both models, then uses Opus 4.6 as a judge
to score responses on: accuracy, completeness, evidence quality, and domain expertise.

Prerequisites:
  - SSH tunnel to HPC: ssh -L 8100:NODE:8100 -N M316235@onehpc.merckgroup.com
  - API key for Uptimize Bedrock (Sonnet 4.6 + Opus 4.6)

Usage:
  python eval/run_eval.py
  python eval/run_eval.py --test-ids diag-01 config-02
  python eval/run_eval.py --skip-qwen  # only run Sonnet (no tunnel needed)
"""

import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime

import httpx

# --- API Configuration ---

UPTIMIZE_DEV_KEY = "03ed1ad4-9e15-4d86-bb07-5f3cddb52907"
UPTIMIZE_DEV_KEY_2 = "ddc1039d-a2d8-4d0c-8988-3d5894333cad"
UPTIMIZE_PROD_KEY = "7fbd6c14-32dd-4420-a14f-84366e67f4a1"

SONNET_MODEL = "eu.anthropic.claude-sonnet-4-6"
OPUS_MODEL = "eu.anthropic.claude-opus-4-6-v1"

UPTIMIZE_DEV_ENDPOINT = "https://api.nlp.dev.uptimize.merckgroup.com/model/{model}/invoke"
UPTIMIZE_PROD_ENDPOINT = "https://api.nlp.p.uptimize.merckgroup.com/model/{model}/invoke"

QWEN_ENDPOINT = "http://localhost:8100/v1/chat/completions"
QWEN_MODEL_NAME = "qwen3.6-35b-a3b"

HEADERS_BEDROCK = {
    "Content-Type": "application/json",
    "openai-standard": "True",
    "anthropic-version": "bedrock-2023-05-31",
}

# --- System Prompts ---

MMM_SYSTEM_PROMPT = (
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

JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator of Bayesian Marketing Mix Model (MMM) optimization advice. "
    "You will be given a question and two responses from different models. "
    "Score each response on four dimensions (1-10 scale):\n\n"
    "1. **Accuracy** — Are the facts correct? Are thresholds, formulas, and causal claims valid?\n"
    "2. **Completeness** — Does it address all aspects of the question? Does it cover edge cases?\n"
    "3. **Evidence Quality** — Are claims supported by quantitative data, literature references, or prior results?\n"
    "4. **Domain Expertise** — Does it demonstrate deep understanding of Bayesian MMM, pymc-marketing, "
    "pharma channel dynamics, and the specific project constraints?\n\n"
    "Output your evaluation as JSON with this exact structure:\n"
    "```json\n"
    "{\n"
    '  "model_a": {"accuracy": N, "completeness": N, "evidence": N, "expertise": N, "total": N, "notes": "..."},\n'
    '  "model_b": {"accuracy": N, "completeness": N, "evidence": N, "expertise": N, "total": N, "notes": "..."},\n'
    '  "winner": "model_a" | "model_b" | "tie",\n'
    '  "reasoning": "Brief explanation of why one is better"\n'
    "}\n"
    "```\n"
    "Total = sum of 4 scores (max 40). Be rigorous — a generic answer that sounds right but lacks "
    "specificity should score lower than a precise, evidence-grounded answer."
)


def call_bedrock(model: str, messages: list[dict], api_key: str = None, max_tokens: int = 1500) -> str:
    """Call Uptimize Bedrock endpoint (Sonnet or Opus)."""
    key = api_key or UPTIMIZE_DEV_KEY
    url = UPTIMIZE_DEV_ENDPOINT.format(model=model)
    headers = {**HEADERS_BEDROCK, "api-key": key}

    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }

    with httpx.Client(timeout=120) as client:
        resp = client.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            # Try fallback key
            headers["api-key"] = UPTIMIZE_DEV_KEY_2
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                # Try prod
                url = UPTIMIZE_PROD_ENDPOINT.format(model=model)
                headers["api-key"] = UPTIMIZE_PROD_KEY
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()

        data = resp.json()
        # OpenAI-compatible format
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        # Anthropic native format
        if "content" in data:
            return data["content"][0]["text"]
        raise ValueError(f"Unexpected response format: {list(data.keys())}")


def call_qwen(messages: list[dict], max_tokens: int = 1500) -> str:
    """Call fine-tuned Qwen via local SGLang endpoint."""
    payload = {
        "model": QWEN_MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }

    with httpx.Client(timeout=120) as client:
        resp = client.post(QWEN_ENDPOINT, json=payload, headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def load_golden_tests(path: Path, test_ids: list[str] = None) -> list[dict]:
    """Load golden test prompts."""
    tests = []
    with open(path) as f:
        for line in f:
            test = json.loads(line)
            if test_ids is None or test["id"] in test_ids:
                tests.append(test)
    return tests


def run_evaluation(tests: list[dict], skip_qwen: bool = False, skip_sonnet: bool = False) -> list[dict]:
    """Run all tests through both models and judge."""
    results = []
    total = len(tests)

    for i, test in enumerate(tests, 1):
        print(f"\n[{i}/{total}] {test['id']} ({test['category']}) — {test['difficulty']}")
        print(f"  Q: {test['prompt'][:80]}...")

        messages = [
            {"role": "system", "content": MMM_SYSTEM_PROMPT},
            {"role": "user", "content": test["prompt"]},
        ]

        result = {
            "id": test["id"],
            "category": test["category"],
            "difficulty": test["difficulty"],
            "prompt": test["prompt"],
            "reference_notes": test["reference_notes"],
        }

        # --- Model A: Sonnet 4.6 ---
        if not skip_sonnet:
            print("  -> Calling Sonnet 4.6...", end=" ", flush=True)
            t0 = time.time()
            try:
                sonnet_response = call_bedrock(SONNET_MODEL, messages)
                sonnet_time = time.time() - t0
                result["sonnet_response"] = sonnet_response
                result["sonnet_time_s"] = round(sonnet_time, 1)
                print(f"OK ({sonnet_time:.1f}s, {len(sonnet_response)} chars)")
            except Exception as e:
                print(f"FAILED: {e}")
                result["sonnet_response"] = f"ERROR: {e}"
                result["sonnet_time_s"] = None
        else:
            result["sonnet_response"] = "(skipped)"
            result["sonnet_time_s"] = None

        # --- Model B: Fine-tuned Qwen ---
        if not skip_qwen:
            print("  -> Calling Qwen MMM...", end=" ", flush=True)
            t0 = time.time()
            try:
                qwen_response = call_qwen(messages)
                qwen_time = time.time() - t0
                result["qwen_response"] = qwen_response
                result["qwen_time_s"] = round(qwen_time, 1)
                print(f"OK ({qwen_time:.1f}s, {len(qwen_response)} chars)")
            except Exception as e:
                print(f"FAILED: {e}")
                result["qwen_response"] = f"ERROR: {e}"
                result["qwen_time_s"] = None
        else:
            result["qwen_response"] = "(skipped)"
            result["qwen_time_s"] = None

        # --- Judge: Opus 4.6 ---
        if not skip_sonnet and not skip_qwen and "ERROR" not in result.get("sonnet_response", "") and "ERROR" not in result.get("qwen_response", ""):
            print("  -> Judging with Opus 4.6...", end=" ", flush=True)
            t0 = time.time()

            # Randomize order to avoid position bias
            import random
            random.seed(hash(test["id"]))
            if random.random() < 0.5:
                model_a_label, model_b_label = "Sonnet 4.6", "Qwen MMM (fine-tuned)"
                model_a_resp, model_b_resp = result["sonnet_response"], result["qwen_response"]
                order = "sonnet_first"
            else:
                model_a_label, model_b_label = "Qwen MMM (fine-tuned)", "Sonnet 4.6"
                model_a_resp, model_b_resp = result["qwen_response"], result["sonnet_response"]
                order = "qwen_first"

            judge_prompt = (
                f"## Question\n{test['prompt']}\n\n"
                f"## Reference Answer Notes\n{test['reference_notes']}\n\n"
                f"## Model A Response\n{model_a_resp}\n\n"
                f"## Model B Response\n{model_b_resp}\n\n"
                f"Score both responses. Model A is presented first, Model B second. "
                f"Evaluate purely on quality — ignore which is longer."
            )

            judge_messages = [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": judge_prompt},
            ]

            try:
                judge_response = call_bedrock(OPUS_MODEL, judge_messages, max_tokens=1000)
                judge_time = time.time() - t0
                print(f"OK ({judge_time:.1f}s)")

                # Parse judge JSON
                try:
                    # Extract JSON from response (handle markdown code blocks)
                    json_str = judge_response
                    if "```json" in json_str:
                        json_str = json_str.split("```json")[1].split("```")[0]
                    elif "```" in json_str:
                        json_str = json_str.split("```")[1].split("```")[0]
                    scores = json.loads(json_str.strip())

                    # Map back to actual model names
                    if order == "sonnet_first":
                        result["judge_sonnet"] = scores.get("model_a", {})
                        result["judge_qwen"] = scores.get("model_b", {})
                    else:
                        result["judge_sonnet"] = scores.get("model_b", {})
                        result["judge_qwen"] = scores.get("model_a", {})

                    result["judge_winner_raw"] = scores.get("winner", "")
                    # Translate winner to actual model name
                    if result["judge_winner_raw"] == "model_a":
                        result["winner"] = model_a_label.split(" ")[0].lower()  # "sonnet" or "qwen"
                    elif result["judge_winner_raw"] == "model_b":
                        result["winner"] = model_b_label.split(" ")[0].lower()
                    else:
                        result["winner"] = "tie"

                    result["judge_reasoning"] = scores.get("reasoning", "")
                    result["judge_order"] = order

                    s_total = result["judge_sonnet"].get("total", 0)
                    q_total = result["judge_qwen"].get("total", 0)
                    print(f"  ->Sonnet: {s_total}/40 | Qwen: {q_total}/40 | Winner: {result['winner']}")

                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    print(f"  ->Judge parse error: {e}")
                    result["judge_raw"] = judge_response
                    result["judge_parse_error"] = str(e)

            except Exception as e:
                print(f"FAILED: {e}")
                result["judge_error"] = str(e)

        results.append(result)
        time.sleep(0.5)  # Rate limiting

    return results


def generate_report(results: list[dict], output_dir: Path) -> str:
    """Generate markdown comparison report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    sonnet_wins = sum(1 for r in results if r.get("winner") == "sonnet")
    qwen_wins = sum(1 for r in results if r.get("winner") == "qwen")
    ties = sum(1 for r in results if r.get("winner") == "tie")
    errors = sum(1 for r in results if "judge_error" in r or "judge_parse_error" in r)

    # Category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"sonnet": 0, "qwen": 0, "tie": 0, "sonnet_avg": [], "qwen_avg": []}
        if r.get("winner") == "sonnet":
            categories[cat]["sonnet"] += 1
        elif r.get("winner") == "qwen":
            categories[cat]["qwen"] += 1
        elif r.get("winner") == "tie":
            categories[cat]["tie"] += 1
        if "judge_sonnet" in r and "total" in r.get("judge_sonnet", {}):
            categories[cat]["sonnet_avg"].append(r["judge_sonnet"]["total"])
        if "judge_qwen" in r and "total" in r.get("judge_qwen", {}):
            categories[cat]["qwen_avg"].append(r["judge_qwen"]["total"])

    # Dimension averages
    dims = ["accuracy", "completeness", "evidence", "expertise"]
    sonnet_dims = {d: [] for d in dims}
    qwen_dims = {d: [] for d in dims}
    for r in results:
        for d in dims:
            if "judge_sonnet" in r and d in r.get("judge_sonnet", {}):
                sonnet_dims[d].append(r["judge_sonnet"][d])
            if "judge_qwen" in r and d in r.get("judge_qwen", {}):
                qwen_dims[d].append(r["judge_qwen"][d])

    def avg(lst):
        return round(sum(lst) / len(lst), 1) if lst else 0

    report = f"""# MMM Model Evaluation Report

**Date**: {timestamp}
**Judge**: Claude Opus 4.6
**Models**: Sonnet 4.6 (general) vs Qwen3.6-35B-A3B-MMM (fine-tuned)
**Test cases**: {len(results)}

## Summary

| Metric | Sonnet 4.6 | Qwen MMM | Tie |
|--------|-----------|----------|-----|
| **Wins** | {sonnet_wins} | {qwen_wins} | {ties} |
| **Win rate** | {sonnet_wins/max(len(results)-errors,1)*100:.0f}% | {qwen_wins/max(len(results)-errors,1)*100:.0f}% | {ties/max(len(results)-errors,1)*100:.0f}% |

## Dimension Scores (average /10)

| Dimension | Sonnet 4.6 | Qwen MMM |
|-----------|-----------|----------|
| Accuracy | {avg(sonnet_dims['accuracy'])} | {avg(qwen_dims['accuracy'])} |
| Completeness | {avg(sonnet_dims['completeness'])} | {avg(qwen_dims['completeness'])} |
| Evidence Quality | {avg(sonnet_dims['evidence'])} | {avg(qwen_dims['evidence'])} |
| Domain Expertise | {avg(sonnet_dims['expertise'])} | {avg(qwen_dims['expertise'])} |
| **Overall (/40)** | **{avg([sum(sonnet_dims[d][i] for d in dims) for i in range(min(len(v) for v in sonnet_dims.values())) ] if all(sonnet_dims.values()) else [0])}** | **{avg([sum(qwen_dims[d][i] for d in dims) for i in range(min(len(v) for v in qwen_dims.values()))] if all(qwen_dims.values()) else [0])}** |

## Category Breakdown

| Category | Sonnet Wins | Qwen Wins | Ties | Sonnet Avg | Qwen Avg |
|----------|------------|-----------|------|-----------|----------|
"""
    for cat, data in sorted(categories.items()):
        s_avg = avg(data["sonnet_avg"])
        q_avg = avg(data["qwen_avg"])
        report += f"| {cat} | {data['sonnet']} | {data['qwen']} | {data['tie']} | {s_avg} | {q_avg} |\n"

    report += "\n## Per-Test Results\n\n"
    for r in results:
        s_score = r.get("judge_sonnet", {}).get("total", "?")
        q_score = r.get("judge_qwen", {}).get("total", "?")
        winner = r.get("winner", "error")
        report += f"- **{r['id']}** ({r['category']}/{r['difficulty']}): Sonnet={s_score} Qwen={q_score} → **{winner}**\n"
        if r.get("judge_reasoning"):
            report += f"  - _{r['judge_reasoning']}_\n"

    report += f"\n## Errors\n\n{errors} test(s) had evaluation errors.\n"

    return report


def main():
    parser = argparse.ArgumentParser(description="Run MMM model evaluation")
    parser.add_argument("--test-ids", nargs="*", help="Specific test IDs to run")
    parser.add_argument("--skip-qwen", action="store_true", help="Skip Qwen (no tunnel needed)")
    parser.add_argument("--skip-sonnet", action="store_true", help="Skip Sonnet")
    parser.add_argument("--golden-path", type=str, default=None, help="Path to golden test JSONL")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    golden_path = Path(args.golden_path) if args.golden_path else project_root / "eval" / "golden_test.jsonl"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "eval" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== MMM Model Evaluation ===")
    print(f"Golden tests: {golden_path}")
    print(f"Output: {output_dir}")
    print()

    # Load tests
    tests = load_golden_tests(golden_path, args.test_ids)
    print(f"Loaded {len(tests)} test cases")

    if not args.skip_qwen:
        # Quick health check on Qwen
        print("Checking Qwen endpoint...", end=" ")
        try:
            with httpx.Client(timeout=5) as c:
                r = c.get("http://localhost:8100/health")
                if r.status_code == 200:
                    print("OK")
                else:
                    print(f"WARNING: status {r.status_code}")
        except Exception as e:
            print(f"UNAVAILABLE ({e})")
            print("  ->Start SSH tunnel: ssh -L 8100:NODE:8100 -N M316235@onehpc.merckgroup.com")
            if not args.skip_sonnet:
                print("  ->Running Sonnet-only mode")
                args.skip_qwen = True
            else:
                print("  ->Both models unavailable, exiting")
                sys.exit(1)

    # Run evaluation
    results = run_evaluation(tests, skip_qwen=args.skip_qwen, skip_sonnet=args.skip_sonnet)

    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"eval_{timestamp}.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nRaw results: {results_path}")

    # Generate report
    if not args.skip_qwen and not args.skip_sonnet:
        report = generate_report(results, output_dir)
        report_path = output_dir / f"report_{timestamp}.md"
        report_path.write_text(report, encoding="utf-8")
        print(f"Report: {report_path}")
        print("\n" + "=" * 60)
        print(report[:2000])
    else:
        print("\nSkipped report generation (need both models for comparison)")


if __name__ == "__main__":
    main()
