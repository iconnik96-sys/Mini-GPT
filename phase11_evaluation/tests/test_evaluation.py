"""
Unit and Integration Tests for Phase 11 Evaluation Suite.

Tests cover:
- Model metadata loading and schema verification
- Parameter counting and trainable percentage calculation
- Result serialization to JSON
- Metric calculations (Distinct-1, Distinct-2, repetition rate, domain keywords, syntax score)
- Generation result structure
- Missing checkpoint handling (graceful FileNotFoundError)
- Reproducibility and configuration validation
"""

import os
import tempfile
import shutil
import json
import unittest

from phase11_evaluation.config import Phase11Config, ModelTargetConfig
from phase11_evaluation.src.loaders import load_system, get_checkpoint_size_mb
from phase11_evaluation.src.metrics import (
    tokenize_words,
    compute_distinct_n,
    compute_repetition_rate,
    compute_domain_keyword_coverage,
    compute_syntax_heuristic_score,
    analyze_text
)
from phase11_evaluation.src.generation_eval import (
    check_char_tokenizer_oov,
    generate_single_prompt
)
from phase11_evaluation.src.report import compile_final_comparison_json


class TestPhase11Evaluation(unittest.TestCase):
    def setUp(self):
        self.config = Phase11Config()

    def test_1_configuration_structure(self):
        """Verify Phase11Config defines all 4 systems and paths."""
        self.assertEqual(len(self.config.models), 4)
        sys_ids = [m.system_id for m in self.config.models]
        self.assertIn("minigpt_baseline", sys_ids)
        self.assertIn("minigpt_programming", sys_ids)
        self.assertIn("distilgpt2_base", sys_ids)
        self.assertIn("distilgpt2_lora", sys_ids)
        self.assertTrue(os.path.exists(self.config.prompts_path))

    def test_2_prompts_file_validity(self):
        """Verify evaluation_prompts.json contains expected structure and required categories."""
        with open(self.config.prompts_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        self.assertGreaterEqual(len(prompts), 10)
        categories = {p["category"] for p in prompts}
        self.assertIn("Java OOP", categories)
        self.assertIn("Spring Boot REST API", categories)
        self.assertIn("SQL", categories)
        for p in prompts:
            self.assertIn("id", p)
            self.assertIn("prompt", p)
            self.assertIn("category", p)

    def test_3_metadata_extraction_minigpt_baseline(self):
        """Verify metadata extraction for MiniGPT baseline model."""
        baseline_cfg = next(m for m in self.config.models if m.system_id == "minigpt_baseline")
        model, tokenizer, metadata = load_system(baseline_cfg, device="cpu")
        self.assertEqual(metadata["system_id"], "minigpt_baseline")
        self.assertEqual(metadata["total_parameters"], 110336)
        self.assertEqual(metadata["trainable_parameters"], 110336)
        self.assertEqual(metadata["trainable_percentage"], 100.0)
        self.assertEqual(metadata["vocabulary_size"], 52)
        self.assertGreater(metadata["checkpoint_size_mb"], 0.0)

    def test_4_metadata_extraction_minigpt_programming(self):
        """Verify metadata extraction for MiniGPT programming domain model."""
        prog_cfg = next(m for m in self.config.models if m.system_id == "minigpt_programming")
        model, tokenizer, metadata = load_system(prog_cfg, device="cpu")
        self.assertEqual(metadata["system_id"], "minigpt_programming")
        self.assertEqual(metadata["total_parameters"], 114944)
        self.assertEqual(metadata["trainable_percentage"], 100.0)
        self.assertEqual(metadata["vocabulary_size"], 88)

    def test_5_metadata_extraction_distilgpt2_lora(self):
        """Verify metadata extraction for DistilGPT-2 LoRA model."""
        lora_cfg = next(m for m in self.config.models if m.system_id == "distilgpt2_lora")
        model, tokenizer, metadata = load_system(lora_cfg, device="cpu")
        self.assertEqual(metadata["system_id"], "distilgpt2_lora")
        self.assertEqual(metadata["total_parameters"], 82060032)
        self.assertEqual(metadata["trainable_parameters"], 147456)
        self.assertAlmostEqual(metadata["trainable_percentage"], 0.1797, places=3)
        self.assertEqual(metadata["vocabulary_size"], 50257)

    def test_6_metric_distinct_n_and_repetition(self):
        """Verify calculation of Distinct-1, Distinct-2, and repetition rate."""
        # Repetitive words
        rep_words = ["the", "the", "the", "the"]
        self.assertEqual(compute_distinct_n(rep_words, 1), 0.25)
        self.assertEqual(compute_distinct_n(rep_words, 2), 0.3333)
        self.assertEqual(compute_repetition_rate(rep_words), 0.75)
        
        # Varied words
        varied_words = ["spring", "boot", "controller", "service"]
        self.assertEqual(compute_distinct_n(varied_words, 1), 1.0)
        self.assertEqual(compute_repetition_rate(varied_words), 0.0)

    def test_7_metric_domain_and_syntax_heuristics(self):
        """Verify domain keyword and syntax scoring heuristics."""
        code_sample = "public class UserService { public String getUser() { return name; } }"
        kw_res = compute_domain_keyword_coverage(code_sample)
        self.assertIn("class", kw_res["found_keywords"])
        self.assertIn("public", kw_res["found_keywords"])
        self.assertIn("return", kw_res["found_keywords"])
        self.assertGreater(kw_res["count"], 2)
        
        syntax_res = compute_syntax_heuristic_score(code_sample)
        self.assertTrue(syntax_res["braces_balanced"])
        self.assertTrue(syntax_res["parens_balanced"])
        self.assertTrue(syntax_res["has_semicolon"])
        self.assertEqual(syntax_res["syntax_score_out_of_4"], 4)

    def test_8_char_tokenizer_oov_rejection(self):
        """Verify unknown characters trigger graceful OOV detection without crashing."""
        baseline_cfg = next(m for m in self.config.models if m.system_id == "minigpt_baseline")
        model, tokenizer, _ = load_system(baseline_cfg, device="cpu")
        
        # Prompt with '@' which does not exist in Shakespeare vocab
        res = generate_single_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt="@RestController",
            model_cfg=baseline_cfg
        )
        self.assertEqual(res["status"], "oov_rejected")
        self.assertIn("vocabulary mismatch", res["error"])

    def test_9_missing_checkpoint_handling(self):
        """Verify FileNotFoundError is raised when checkpoint path does not exist."""
        bad_cfg = ModelTargetConfig(
            system_id="minigpt_baseline",
            display_name="Missing Model",
            category="from_scratch",
            checkpoint_path="checkpoints/does_not_exist.pt",
            tokenizer_type="character"
        )
        with self.assertRaises(FileNotFoundError):
            load_system(bad_cfg, device="cpu")

    def test_10_result_compilation_and_serialization(self):
        """Verify serialization of evaluation results into summary JSON."""
        temp_dir = os.path.join("phase11_evaluation", "results", "temp_test_dir")
        os.makedirs(temp_dir, exist_ok=True)
        try:
            fake_results = {
                "sys1": {
                    "metadata": {"system_id": "sys1", "display_name": "Sys1"},
                    "aggregates": {"average_generation_time_sec": 0.5},
                    "prompt_results": [{
                        "prompt_id": "p01",
                        "category": "Test",
                        "prompt": "Test Prompt",
                        "status": "success",
                        "output": "Test Output",
                        "generated_tokens": 10,
                        "generation_time_sec": 0.5,
                        "tokens_per_sec": 20.0,
                        "metrics": {"repetition_rate": 0.1, "distinct_1": 0.9}
                    }]
                }
            }
            out_file = os.path.join(temp_dir, "summary.json")
            compiled = compile_final_comparison_json(fake_results, out_file)
            self.assertTrue(os.path.exists(out_file))
            self.assertIn("evaluation_title", compiled)
            self.assertEqual(len(compiled["side_by_side_prompts"]), 1)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
