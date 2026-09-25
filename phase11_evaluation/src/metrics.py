"""
Quantitative and Heuristic Generation Metrics for Phase 11 Evaluation.

Implements:
- Lexical diversity (Distinct-1, Distinct-2)
- Repetition rate
- Domain keyword coverage (heuristic detection of programming terms)
- Structural syntax-oriented checks for Java/code (braces, parens, semicolons)
- Aggregation helpers across prompts and models

IMPORTANT: These are heuristic, automated proxies for text diversity and syntactic features,
not human quality scores or correctness judgments.
"""

import re
from typing import List, Dict, Any, Tuple


# Key domain terms to check heuristic coverage
DOMAIN_KEYWORDS = [
    "class", "interface", "public", "private", "return", "spring", "boot",
    "controller", "service", "repository", "jpa", "sql", "join", "select",
    "exception", "http", "status", "null", "method", "table", "database",
    "inheritance", "polymorphism", "encapsulation", "transactional", "autowired"
]


def tokenize_words(text: str) -> List[str]:
    """Extract lowercase word tokens from text."""
    return re.findall(r"\b\w+\b", text.lower())


def compute_distinct_n(words: List[str], n: int = 1) -> float:
    """
    Compute Distinct-N metric (ratio of unique n-grams to total n-grams).
    Higher values indicate greater lexical diversity; lower values indicate repetitive text.
    """
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return round(len(set(ngrams)) / len(ngrams), 4)


def compute_repetition_rate(words: List[str]) -> float:
    """
    Compute unigram repetition rate (1 - Distinct-1).
    0.0 means completely unique words; 1.0 means infinite repetition of identical tokens.
    """
    if not words:
        return 0.0
    distinct_1 = len(set(words)) / len(words)
    return round(1.0 - distinct_1, 4)


def compute_domain_keyword_coverage(text: str) -> Dict[str, Any]:
    """
    Detect presence of domain programming keywords in the generated response.
    """
    words_set = set(tokenize_words(text))
    found = [kw for kw in DOMAIN_KEYWORDS if kw in words_set]
    ratio = round(len(found) / len(DOMAIN_KEYWORDS), 4)
    return {
        "found_keywords": found,
        "count": len(found),
        "ratio": ratio
    }


def compute_syntax_heuristic_score(text: str) -> Dict[str, Any]:
    """
    Heuristic syntactic checks for generated code/Java responses:
    - Balanced curly braces {}
    - Balanced parentheses ()
    - Presence of statement terminators (semicolon ;)
    - CamelCase / identifier structure
    """
    open_brace = text.count("{")
    close_brace = text.count("}")
    braces_balanced = (open_brace > 0) and (open_brace == close_brace)
    
    open_paren = text.count("(")
    close_paren = text.count(")")
    parens_balanced = (open_paren > 0) and (open_paren == close_paren)
    
    has_semicolon = ";" in text
    has_code_keywords = any(kw in text for kw in ["public ", "class ", "void ", "return ", "String "])
    
    score = 0
    if open_brace > 0 and braces_balanced:
        score += 1
    if open_paren > 0 and parens_balanced:
        score += 1
    if has_semicolon:
        score += 1
    if has_code_keywords:
        score += 1
        
    return {
        "braces_balanced": braces_balanced,
        "parens_balanced": parens_balanced,
        "has_semicolon": has_semicolon,
        "has_code_keywords": has_code_keywords,
        "syntax_score_out_of_4": score
    }


def analyze_text(text: str) -> Dict[str, Any]:
    """Run full suite of heuristic metrics on a single generated response."""
    words = tokenize_words(text)
    d1 = compute_distinct_n(words, 1)
    d2 = compute_distinct_n(words, 2)
    rep_rate = compute_repetition_rate(words)
    kw_stats = compute_domain_keyword_coverage(text)
    syntax_stats = compute_syntax_heuristic_score(text)
    
    return {
        "char_length": len(text),
        "word_count": len(words),
        "distinct_1": d1,
        "distinct_2": d2,
        "repetition_rate": rep_rate,
        "domain_keywords_found": kw_stats["count"],
        "domain_keywords_list": kw_stats["found_keywords"],
        "syntax_score": syntax_stats["syntax_score_out_of_4"]
    }
