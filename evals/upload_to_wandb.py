"""
Re-upload locally saved lm_eval results to wandb.

Usage:
    python -m evals.upload_to_wandb \
        --results_dir evals/BASE_FINAL \
        --project apertus-moe-final-evals \
        --run_id BASE_FINAL
"""

import argparse
import copy
import json
import re
from pathlib import Path

import wandb


def remove_none_pattern(input_string: str) -> tuple[str, bool]:
    pattern = re.compile(r",none$")
    result = re.sub(pattern, "", input_string)
    return result, result != input_string


def latest_results(step_dir: Path) -> dict | None:
    jsons = sorted(step_dir.glob("**/results_*.json"))
    if not jsons:
        return None
    return json.loads(jsons[-1].read_text())


def sanitize_results(results: dict) -> tuple[dict, dict]:
    """Mirrors WandbLogger._sanitize_results_dict from lm-evaluation-harness.

    Returns (wandb_summary, wandb_metrics) where wandb_summary holds string-valued
    metrics and wandb_metrics holds all numeric metrics keyed as 'task/metric'.
    """
    _results = copy.deepcopy(results.get("results", {}))
    task_names = list(_results.keys())

    # Strip ,none suffix from all metric names
    for task_name in task_names:
        task_result = copy.deepcopy(_results[task_name])
        for metric_name, metric_value in task_result.items():
            clean_name, removed = remove_none_pattern(metric_name)
            if removed:
                _results[task_name][clean_name] = metric_value
                _results[task_name].pop(metric_name)

    # Separate string values into wandb_summary
    wandb_summary = {}
    for task in task_names:
        task_result = _results.get(task, {})
        for metric_name, metric_value in task_result.items():
            if isinstance(metric_value, str):
                wandb_summary[f"{task}/{metric_name}"] = metric_value

    for summary_key in wandb_summary:
        _task, _metric = summary_key.split("/", 1)
        _results[_task].pop(_metric)

    # Flatten task/metric -> value
    wandb_metrics = {}
    for task_name, task_results in _results.items():
        for metric_name, metric_value in task_results.items():
            wandb_metrics[f"{task_name}/{metric_name}"] = metric_value

    return wandb_summary, wandb_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run_id", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    step_dirs = sorted(
        results_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1])
    )

    run = wandb.init(project=args.project, id=args.run_id, resume="allow")
    run.define_metric("OptimizerStep")
    run.define_metric("*", step_metric="OptimizerStep")

    for step_dir in step_dirs:
        step = int(step_dir.name.split("_")[1])
        data = latest_results(step_dir)
        if data is None:
            print(f"  step {step}: no results found, skipping")
            continue

        wandb_summary, wandb_metrics = sanitize_results(data)
        run.summary.update(wandb_summary)

        step_metrics = {"OptimizerStep": step}
        run.log({**wandb_metrics, **step_metrics}, commit=True)
        print(f"  step {step}: uploaded {len(wandb_metrics)} metrics")

    run.finish()


if __name__ == "__main__":
    main()
