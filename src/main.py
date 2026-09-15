"""
FlowGen — Main entry point.

Runs an experiment based on YAML configuration.

Usage:
    python -m src.main --config config/experiments/raw_baseline.yaml
    python -m src.main --config config/experiments/waterfall_baseline.yaml
    python -m src.main  # Uses default.yaml

    # Tasks from .txt files
    python -m src.main --task tasks/task_001.txt
    python -m src.main --tasks tasks/
    python -m src.main --tasks tasks/ --process waterfall --model qwen:0.5b
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from src.benchmarks.base import Problem
from src.benchmarks.loaders import create_benchmark
from src.config import ExperimentConfig
from src.evaluation.evaluator import Evaluator
from src.evaluation.statistics import aggregate_results
from src.llm.factory import create_llm_provider
from src.logging_config import LLMCallLogger, setup_logging
from src.orchestration.context import SharedContext
from src.orchestration.orchestrator import Orchestrator
from src.processes.factory import create_process
from src.results.report_generator import ReportGenerator
from src.results.result_store import ResultStore
from src.tasks.task_loader import TaskLoader

logger = logging.getLogger(__name__)


def _load_problems(config: ExperimentConfig) -> tuple[list[Problem], str]:
    """
    Load problems from the configured source (benchmark or tasks).

    Returns:
        Tuple of (list of Problems, source name for reporting).
    """
    # Check if tasks path is configured
    if config.tasks.path:
        loader = TaskLoader()
        tasks_path = Path(config.tasks.path)

        if tasks_path.is_file():
            tasks = [loader.load_file(tasks_path)]
        elif tasks_path.is_dir():
            tasks = loader.load_directory(tasks_path)
        else:
            raise FileNotFoundError(
                f"Tasks path not found: {tasks_path}"
            )

        problems = [t.to_problem() for t in tasks]
        source_name = "tasks"
        logger.info("Loaded %d tasks from: %s", len(problems), tasks_path)
        return problems, source_name

    # Fallback to benchmark
    benchmark = create_benchmark(config)
    problems = benchmark.load()

    if config.benchmark.subset:
        problems = benchmark.get_subset(config.benchmark.subset)

    source_name = config.benchmark.name
    logger.info("Benchmark: %s (%d problems)", source_name, len(problems))
    return problems, source_name


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """
    Run a complete experiment based on configuration.

    Args:
        config: Fully loaded ExperimentConfig.

    Returns:
        Summary dict with results.
    """
    # Setup logging
    setup_logging(
        log_dir=config.logging.log_dir,
        level=config.logging.level,
        experiment_name=config.name,
    )

    logger.info("=" * 60)
    logger.info("Starting experiment: %s", config.name)
    logger.info("Process: %s | Model: %s | Runs: %d", config.pipeline.type, config.llm.model, config.runs)
    logger.info("=" * 60)

    # Create LLM provider
    llm = create_llm_provider(config)
    logger.info("LLM provider: %s (%s)", config.llm.provider, config.llm.model)

    # Load problems from benchmark or tasks
    problems, source_name = _load_problems(config)

    # Create process model
    process = create_process(config, llm)
    logger.info("Process model: %s", process.process_name)

    # Create result store
    result_store = ResultStore(output_dir=config.results.output_dir)
    experiment_id = result_store.initialize_experiment(
        experiment_name=config.name,
        config_dict=config.to_dict(),
    )
    logger.info("Experiment ID: %s", experiment_id)

    # Create LLM call logger
    llm_call_logger = LLMCallLogger(
        log_path=Path(config.logging.log_dir) / "llm_calls.jsonl"
    )

    # Create evaluator
    evaluator = Evaluator(timeout=config.evaluation.timeout)

    # Run experiment
    all_run_results: list[dict[str, Any]] = []

    for run_number in range(1, config.runs + 1):
        logger.info("-" * 40)
        logger.info("Run %d/%d", run_number, config.runs)
        logger.info("-" * 40)

        run_start = time.perf_counter()

        # Reset LLM metrics for this run
        llm.reset_metrics()

        # Process each problem
        code_results: list[dict[str, Any]] = []
        orchestrator = Orchestrator(config)

        for problem in problems:
            logger.info("Problem: %s", problem.task_id)

            # Create fresh context
            context = SharedContext(
                problem=problem.prompt,
                problem_id=problem.task_id,
                entry_point=problem.entry_point,
                canonical_tests=problem.test,
            )

            # Reset orchestrator for each problem
            orchestrator.reset_for_run()

            # Run the process
            try:
                context = process.run(context, orchestrator)
            except Exception as e:
                logger.error("Error processing %s: %s", problem.task_id, e)
                context.code = ""

            # Record results
            code_results.append({
                "task_id": problem.task_id,
                "code": context.code,
            })

        # Evaluate this run
        eval_result = evaluator.evaluate_batch(code_results, problems)

        run_elapsed_ms = (time.perf_counter() - run_start) * 1000

        # Collect metrics
        llm_metrics = llm.get_metrics()
        run_result = {
            "run_number": run_number,
            "pass_at_1": eval_result["pass_at_1"],
            "total": eval_result["total"],
            "passed": eval_result["passed"],
            "failed": eval_result["failed"],
            "total_time_ms": run_elapsed_ms,
            "total_tokens": llm_metrics.get("total_tokens", 0),
            "llm_calls": llm_metrics.get("call_count", 0),
            "results": eval_result["results"],
        }

        all_run_results.append(run_result)

        # Save run results
        result_store.save_run(
            run_number=run_number,
            results=run_result,
            context_dict=None,  # Could save last context for debugging
            messages=[m.to_dict() for m in orchestrator.message_bus.all_messages],
            artifacts=[a.to_dict() for a in orchestrator.artifact_store.all_artifacts],
        )

        logger.info(
            "Run %d: Pass@1=%.4f (%d/%d passed) in %.1fs",
            run_number,
            eval_result["pass_at_1"],
            eval_result["passed"],
            eval_result["total"],
            run_elapsed_ms / 1000,
        )

    # Aggregate results across runs
    summary = aggregate_results(all_run_results)
    summary["experiment_id"] = experiment_id
    summary["experiment_name"] = config.name
    summary["model"] = config.llm.model
    summary["process"] = config.pipeline.type
    summary["benchmark"] = source_name

    # Save summary
    result_store.save_summary(summary)

    # Generate report
    report_data = [{
        "model": config.llm.model,
        "process": config.pipeline.type,
        "benchmark": source_name,
        "mean_pass_at_1": summary.get("mean_pass_at_1", 0.0),
        "std_pass_at_1": summary.get("std_pass_at_1", 0.0),
        "runs": summary.get("runs", 0),
        "total_llm_calls": summary.get("total_llm_calls", 0),
    }]

    report = ReportGenerator.generate_markdown_table(report_data)
    if result_store.experiment_dir:
        ReportGenerator.save_report(
            report, result_store.experiment_dir / "report.md"
        )

    logger.info("=" * 60)
    logger.info("Experiment completed: %s", experiment_id)
    logger.info("Mean Pass@1: %.4f (±%.4f)", summary.get("mean_pass_at_1", 0.0), summary.get("std_pass_at_1", 0.0))
    logger.info("Results saved to: %s", result_store.experiment_dir)
    logger.info("=" * 60)

    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="FlowGen — Multi-agent code generation experiment runner"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment config YAML (overrides default.yaml)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Override number of runs",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override LLM model name",
    )
    parser.add_argument(
        "--process",
        type=str,
        default=None,
        help="Override process type (raw, waterfall, tdd, scrum)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Path to a single .txt task file",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Path to a directory of .txt task files",
    )

    args = parser.parse_args()

    # Validate: --task and --tasks are mutually exclusive
    if args.task and args.tasks:
        parser.error("--task and --tasks are mutually exclusive. Use one or the other.")

    # Build overrides from CLI args
    overrides: dict[str, Any] = {}
    if args.runs is not None:
        overrides.setdefault("experiment", {})["runs"] = args.runs
    if args.model is not None:
        overrides.setdefault("llm", {})["model"] = args.model
    if args.process is not None:
        overrides.setdefault("pipeline", {})["type"] = args.process
    if args.task is not None:
        overrides.setdefault("tasks", {})["path"] = args.task
    if args.tasks is not None:
        overrides.setdefault("tasks", {})["path"] = args.tasks

    # Load config
    config = ExperimentConfig.load(
        config_path=args.config,
        overrides=overrides if overrides else None,
    )

    # Run experiment
    try:
        summary = run_experiment(config)
        print(f"\nExperiment complete. Pass@1: {summary.get('mean_pass_at_1', 0.0):.4f}")
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.exception("Experiment failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
