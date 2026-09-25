"""
Instruction Dataset and Tokenization for Phase 10 LoRA Fine-Tuning.

Implements:
- Prompt formatting with standard instruction/response markers
- Prompt-loss masking (labels set to -100 for the instruction tokens)
- Train / Validation dataset splitting with reproducible seed
- PyTorch Dataset and DataLoader creation
"""

import json
import random
from typing import List, Dict, Any, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase


PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def format_instruction(instruction: str, response: Optional[str] = None) -> str:
    """Format an instruction and optional response into the canonical template."""
    prompt = PROMPT_TEMPLATE.format(instruction=instruction.strip())
    if response is not None:
        return prompt + response.strip()
    return prompt


def load_dataset(json_path: str) -> List[Dict[str, str]]:
    """Load instruction-response pairs from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Expected a non-empty list of instruction dicts in {json_path}")
    for item in data:
        if "instruction" not in item or "response" not in item:
            raise ValueError(f"Each item must contain 'instruction' and 'response' keys, got: {list(item.keys())}")
    return data


def split_dataset(
    data: List[Dict[str, str]],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Split dataset deterministically into train and validation sets."""
    rng = random.Random(seed)
    shuffled = list(data)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    split_idx = max(1, min(split_idx, len(shuffled) - 1))
    return shuffled[:split_idx], shuffled[split_idx:]


class SFTInstructionDataset(Dataset):
    """
    PyTorch Dataset for Supervised Fine-Tuning (SFT) with LoRA.
    
    Masks instruction prompt tokens with -100 so cross-entropy loss is computed
    only over the target response tokens and the EOS token.
    """
    def __init__(
        self,
        examples: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int = 256
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        instruction = ex["instruction"].strip()
        response = ex["response"].strip()
        
        prompt_text = format_instruction(instruction, None)
        full_text = format_instruction(instruction, response) + self.tokenizer.eos_token
        
        prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        full_tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
        
        if len(full_tokens) > self.max_seq_len:
            full_tokens = full_tokens[:self.max_seq_len]
            
        pad_len = self.max_seq_len - len(full_tokens)
        
        input_ids = full_tokens + [self.tokenizer.pad_token_id] * pad_len
        attention_mask = [1] * len(full_tokens) + [0] * pad_len
        
        # Build labels with -100 mask for prompt tokens and pad tokens
        prompt_len = min(len(prompt_tokens), len(full_tokens))
        labels = [-100] * prompt_len + full_tokens[prompt_len:] + [-100] * pad_len
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


def create_dataloaders(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 4,
    train_ratio: float = 0.8,
    seed: int = 42,
    max_seq_len: int = 256
) -> Tuple[DataLoader, DataLoader, List[Dict[str, str]], List[Dict[str, str]]]:
    """Create train and validation DataLoaders with reproducible split."""
    raw_data = load_dataset(dataset_path)
    train_data, val_data = split_dataset(raw_data, train_ratio=train_ratio, seed=seed)
    
    train_dataset = SFTInstructionDataset(train_data, tokenizer, max_seq_len=max_seq_len)
    val_dataset = SFTInstructionDataset(val_data, tokenizer, max_seq_len=max_seq_len)
    
    # Deterministic generator for DataLoader shuffling
    gen = torch.Generator()
    gen.manual_seed(seed)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, train_data, val_data
