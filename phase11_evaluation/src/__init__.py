"""
Phase 11: Cross-System Evaluation and Comparison.
"""

from phase11_evaluation.config import Phase11Config, ModelTargetConfig
from phase11_evaluation.src.loaders import load_system, get_checkpoint_size_mb
from phase11_evaluation.src.metrics import (
    DOMAIN_KEYWORDS,
    tokenize_words,
    compute_distinct_n,
    compute_repetition_rate,
    compute_domain_keyword_coverage,
    compute_syntax_heuristic_score,
    analyze_text
)
from phase11_evaluation.src.generation_eval import (
    check_char_tokenizer_oov,
    generate_single_prompt,
    evaluate_system_prompts
)
from phase11_evaluation.src.report import (
    compile_final_comparison_json,
    generate_markdown_report
)

__all__ = [
    "Phase11Config",
    "ModelTargetConfig",
    "load_system",
    "get_checkpoint_size_mb",
    "DOMAIN_KEYWORDS",
    "tokenize_words",
    "compute_distinct_n",
    "compute_repetition_rate",
    "compute_domain_keyword_coverage",
    "compute_syntax_heuristic_score",
    "analyze_text",
    "check_char_tokenizer_oov",
    "generate_single_prompt",
    "evaluate_system_prompts",
    "compile_final_comparison_json",
    "generate_markdown_report"
]
