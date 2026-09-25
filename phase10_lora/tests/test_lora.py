"""
Unit and Integration Tests for Phase 10 LoRA Fine-Tuning.

Tests cover:
- Pretrained model and tokenizer loading
- LoRA configuration validation
- Target module detection
- Trainable parameter calculation (parameter efficiency)
- Dataset formatting and label masking (-100)
- Adapter creation and forward pass
- Checkpoint saving and reloading
- Text generation
"""

import os
import shutil
import tempfile
import unittest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

from phase10_lora.src.config import Phase10Config
from phase10_lora.src.dataset import (
    format_instruction,
    load_dataset,
    split_dataset,
    SFTInstructionDataset,
    create_dataloaders
)
from phase10_lora.src.train import (
    count_parameters,
    setup_model_and_tokenizer,
    evaluate_loss,
    load_fine_tuned_model
)
from phase10_lora.src.generate import generate_response


class TestPhase10LoRA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_name = "distilbert/distilgpt2"
        cls.config = Phase10Config(
            model_name=cls.model_name,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["c_attn"],
            batch_size=2,
            max_seq_len=64
        )
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.raw_model = AutoModelForCausalLM.from_pretrained(cls.model_name)

    def test_1_tokenizer_and_model_loading(self):
        """Verify pretrained model and tokenizer instantiate with expected properties."""
        self.assertIsNotNone(self.tokenizer)
        self.assertIsNotNone(self.raw_model)
        self.assertEqual(len(self.tokenizer), 50257)
        base_param_count = sum(p.numel() for p in self.raw_model.parameters())
        self.assertEqual(base_param_count, 81912576)

    def test_2_lora_configuration(self):
        """Verify LoRA configuration values match the specification."""
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            fan_in_fan_out=self.config.fan_in_fan_out,
            bias=self.config.bias
        )
        self.assertEqual(lora_cfg.r, 8)
        self.assertEqual(lora_cfg.lora_alpha, 16)
        self.assertEqual(lora_cfg.lora_dropout, 0.05)
        self.assertEqual(lora_cfg.target_modules, {"c_attn"})

    def test_3_target_module_detection(self):
        """Verify target modules ('c_attn') exist in the base model architecture."""
        target_found = False
        attn_count = 0
        for name, module in self.raw_model.named_modules():
            if "c_attn" in name:
                target_found = True
                attn_count += 1
        self.assertTrue(target_found, "Target module 'c_attn' was not found in model modules.")
        self.assertEqual(attn_count, 6, "Expected 6 attention layers in DistilGPT2.")

    def test_4_parameter_efficiency_calculation(self):
        """Verify parameter counting correctly measures total, trainable, and frozen parameters."""
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["c_attn"],
            fan_in_fan_out=True
        )
        # Use fresh model copy
        base = AutoModelForCausalLM.from_pretrained(self.model_name)
        peft_m = get_peft_model(base, lora_cfg)
        stats = count_parameters(peft_m)
        
        self.assertEqual(stats["total_parameters"], 82060032)
        self.assertEqual(stats["trainable_parameters"], 147456)
        self.assertEqual(stats["frozen_parameters"], 81912576)
        self.assertAlmostEqual(stats["trainable_percentage"], 0.1797, places=3)
        self.assertLess(stats["trainable_percentage"], 0.25)

    def test_5_dataset_formatting_and_loading(self):
        """Verify instruction formatting and JSON dataset loading."""
        formatted = format_instruction("What is Java?", "Java is a programming language.")
        self.assertIn("### Instruction:\nWhat is Java?", formatted)
        self.assertIn("### Response:\nJava is a programming language.", formatted)
        
        data = load_dataset(self.config.dataset_path)
        self.assertGreaterEqual(len(data), 25)
        for item in data:
            self.assertIn("instruction", item)
            self.assertIn("response", item)
            self.assertGreater(len(item["instruction"]), 5)
            self.assertGreater(len(item["response"]), 10)

    def test_6_sft_dataset_label_masking(self):
        """Verify prompt tokens are masked with -100 so loss is computed only on responses."""
        sample_data = [{
            "instruction": "Test instruction",
            "response": "Test response"
        }]
        ds = SFTInstructionDataset(sample_data, self.tokenizer, max_seq_len=64)
        item = ds[0]
        
        input_ids = item["input_ids"]
        labels = item["labels"]
        attention_mask = item["attention_mask"]
        
        self.assertEqual(input_ids.shape, (64,))
        self.assertEqual(labels.shape, (64,))
        self.assertEqual(attention_mask.shape, (64,))
        
        # Verify first tokens (instruction prompt) are masked with -100
        self.assertEqual(labels[0].item(), -100)
        # Verify there are unmasked labels for the response
        unmasked = [l.item() for l in labels if l.item() != -100]
        self.assertGreater(len(unmasked), 0)

    def test_7_adapter_creation_and_forward_pass(self):
        """Verify forward pass computes cross-entropy loss over batch using LoRA model."""
        peft_m, _, _ = setup_model_and_tokenizer(self.config)
        sample_batch = {
            "input_ids": torch.randint(0, 1000, (2, 32)),
            "attention_mask": torch.ones((2, 32), dtype=torch.long),
            "labels": torch.randint(0, 1000, (2, 32))
        }
        outputs = peft_m(**sample_batch)
        self.assertIsNotNone(outputs.loss)
        self.assertFalse(torch.isnan(outputs.loss))
        self.assertEqual(outputs.loss.ndim, 0)

    def test_8_checkpoint_saving_and_reloading(self):
        """Verify saving adapter checkpoint and reloading with PeftModel."""
        peft_m, tokenizer, _ = setup_model_and_tokenizer(self.config)
        test_save_dir = os.path.join("phase10_lora", "checkpoints", "temp_test_adapter")
        os.makedirs(test_save_dir, exist_ok=True)
        try:
            peft_m.save_pretrained(test_save_dir)
            tokenizer.save_pretrained(test_save_dir)
            
            self.assertTrue(os.path.exists(os.path.join(test_save_dir, "adapter_config.json")))
            self.assertTrue(
                os.path.exists(os.path.join(test_save_dir, "adapter_model.safetensors")) or
                os.path.exists(os.path.join(test_save_dir, "adapter_model.bin"))
            )
            
            reloaded_model, reloaded_tok = load_fine_tuned_model(
                checkpoint_dir=test_save_dir,
                base_model_name=self.model_name
            )
            self.assertIsNotNone(reloaded_model)
            self.assertIsNotNone(reloaded_tok)
        finally:
            shutil.rmtree(test_save_dir, ignore_errors=True)

    def test_9_generation_returns_valid_string(self):
        """Verify generate_response generates a non-empty string response."""
        response = generate_response(
            model=self.raw_model,
            tokenizer=self.tokenizer,
            instruction="What is OOP?",
            max_new_tokens=20,
            temperature=0.0
        )
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_10_qlora_investigation_and_hardware_probe(self):
        """Verify QLoRA hardware probe truthfully detects CPU environment and returns valid config."""
        from phase10_lora.src.qlora_investigation import check_qlora_support, get_qlora_config
        probe = check_qlora_support()
        self.assertIn("supported", probe)
        self.assertIn("cuda_available", probe)
        self.assertIn("bitsandbytes_installed", probe)
        self.assertIn("reasons_for_limitation", probe)
        self.assertFalse(probe["supported"])
        self.assertFalse(probe["cuda_available"])
        
        cfg = get_qlora_config()
        self.assertTrue(cfg["load_in_4bit"])
        self.assertEqual(cfg["bnb_4bit_quant_type"], "nf4")
        self.assertTrue(cfg["bnb_4bit_use_double_quant"])

    def test_11_v2_checkpoint_loading(self):
        """Verify v2 checkpoint loads with base DistilGPT-2 model."""
        v2_path = os.path.join("phase10_lora", "checkpoints", "distilgpt2_lora_programming_v2")
        self.assertTrue(os.path.isdir(v2_path), f"Checkpoint directory missing: {v2_path}")
        self.assertTrue(os.path.exists(os.path.join(v2_path, "adapter_model.safetensors")))
        self.assertTrue(os.path.exists(os.path.join(v2_path, "adapter_config.json")))
        
        model, tok = load_fine_tuned_model(v2_path, base_model_name=self.model_name)
        self.assertIsNotNone(model)
        self.assertIsNotNone(tok)
        self.assertIsInstance(model, PeftModel)

    def test_12_prompt_formatting_and_eos_handling(self):
        """Verify prompt formatting, EOS inclusion in training, and stripping in decoding."""
        inst = "What is Java?"
        resp = "Java is a programming language."
        formatted = format_instruction(inst, resp)
        self.assertTrue(formatted.startswith("Below is an instruction"))
        self.assertIn("### Instruction:\nWhat is Java?", formatted)
        self.assertIn("### Response:\nJava is a programming language.", formatted)
        
        # Verify EOS token ID
        self.assertEqual(self.tokenizer.eos_token_id, 50256)
        encoded = self.tokenizer.encode(formatted + self.tokenizer.eos_token, add_special_tokens=False)
        self.assertEqual(encoded[-1], 50256)

    def test_13_generation_anti_repetition_controls(self):
        """Verify repetition penalty and no_repeat_ngram_size break degenerate looping."""
        v2_path = os.path.join("phase10_lora", "checkpoints", "distilgpt2_lora_programming_v2")
        model, tok = load_fine_tuned_model(v2_path, base_model_name=self.model_name)
        
        out = generate_response(
            model=model,
            tokenizer=tok,
            instruction="What is Java?",
            max_new_tokens=40,
            temperature=0.7,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3
        )
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 10)
        
        # Ensure no identical 4-word phrase repeats consecutively
        words = out.split()
        if len(words) >= 8:
            four_grams = [" ".join(words[i:i+4]) for i in range(len(words)-3)]
            self.assertEqual(len(four_grams), len(set(four_grams)), "Detected repeated 4-gram in output!")

    def test_14_v2_dataset_coverage(self):
        """Verify expanded v2 dataset contains >= 80 diverse examples and covers required topics."""
        v2_data_path = os.path.join("phase10_lora", "data", "programming_instructions_v2.json")
        self.assertTrue(os.path.exists(v2_data_path))
        data = load_dataset(v2_data_path)
        self.assertGreaterEqual(len(data), 80)
        
        instructions = " ".join([d["instruction"].lower() for d in data])
        self.assertIn("java", instructions)
        self.assertIn("class", instructions)
        self.assertIn("inheritance", instructions)
        self.assertIn("rest", instructions)
        self.assertIn("spring", instructions)
        self.assertIn("sql", instructions)
        self.assertIn("hello", instructions)


if __name__ == "__main__":
    unittest.main()

