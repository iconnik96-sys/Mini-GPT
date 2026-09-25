"""
Master Orchestration Script for Phase 11 Evaluation.

Executes:
1. Loads 13 standardized evaluation prompts across programming categories.
2. Iterates across all 4 systems (MiniGPT Baseline, MiniGPT Programming, DistilGPT-2 Base, DistilGPT-2 + LoRA).
3. Generates responses using deterministic greedy decoding (and captures latency & throughput).
4. Saves raw output files to phase11_evaluation/results/raw/.
5. Compiles aggregate comparison JSON to phase11_evaluation/results/final/summary_comparison.json.
6. Generates full 15-section report to phase11_evaluation/reports/phase11_report.md.
"""

import os
import sys
import json
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from phase11_evaluation.config import Phase11Config
from phase11_evaluation.src.generation_eval import evaluate_system_prompts
from phase11_evaluation.src.report import compile_final_comparison_json, generate_markdown_report


def main():
    print("=" * 75)
    print("PHASE 11: CROSS-SYSTEM MODEL EVALUATION & COMPARISON")
    print("=" * 75)

    config = Phase11Config()

    print(f"\n[1/4] Loading Evaluation Prompts from {config.prompts_path}...")
    with open(config.prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} standardized benchmark prompts.")

    all_system_results = {}

    print("\n[2/4] Running Deterministic Evaluations Across 4 Systems...")
    for model_cfg in config.models:
        print(f"\n--- Evaluating: {model_cfg.display_name} ---")
        sys_res = evaluate_system_prompts(
            model_cfg=model_cfg,
            prompts=prompts,
            mode="deterministic",
            config=config
        )
        all_system_results[model_cfg.system_id] = sys_res
        aggs = sys_res["aggregates"]
        print(f"  Success: {aggs['successful_prompts']}/{aggs['total_prompts']} | "
              f"Avg Time: {aggs['average_generation_time_sec']}s | "
              f"Throughput: {aggs['average_tokens_per_sec']} tok/s | "
              f"Repetition: {aggs['average_repetition_rate']}")

    print("\n[3/4] Compiling Comparison Summary JSON...")
    final_json_path = os.path.join(config.final_results_dir, "summary_comparison.json")
    summary_data = compile_final_comparison_json(all_system_results, final_json_path)
    print(f"Summary saved to: {final_json_path}")

    print("\n[4/4] Generating Comprehensive Markdown Report...")
    report_path = os.path.join(config.reports_dir, "phase11_report.md")
    generate_markdown_report(summary_data, report_path)
    print(f"Markdown report generated: {report_path}")

    print("\n" + "=" * 75)
    print("EVALUATION COMPLETE - SCOREBOARD SUMMARY")
    print("=" * 75)
    print(f"{'System':<32} | {'Params':<10} | {'Trainable %':<12} | {'Latency':<9} | {'Throughput':<10}")
    print("-" * 75)
    for sys_id, res in all_system_results.items():
        m = res["metadata"]
        a = res["aggregates"]
        print(f"{m['display_name']:<32} | {m['total_parameters']:<10,d} | {m['trainable_percentage']:<11.4f}% | {a['average_generation_time_sec']:<8.3f}s | {a['average_tokens_per_sec']:<10.1f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
