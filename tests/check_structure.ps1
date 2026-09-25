# MiniGPT Phase 1 - Project Structure Verification Script

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "MiniGPT Phase 1: Structure Verification" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

$passed = $true

# 1. Required Directories
$dirs = @("data", "checkpoints", "src", "tests")
Write-Host "`n[1] Checking Required Directories:" -ForegroundColor Yellow
foreach ($d in $dirs) {
    if (Test-Path -Path $d -PathType Container) {
        Write-Host "  [PASS] Directory exists: $d" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] Directory missing: $d" -ForegroundColor Red
        $passed = $false
    }
}

# 2. Required Files
$files = @(
    "config.py",
    "requirements.txt",
    "README.md",
    ".gitignore",
    "data/.gitkeep",
    "checkpoints/.gitkeep",
    "src/__init__.py",
    "src/tokenizer.py",
    "src/dataset.py",
    "src/model.py",
    "src/train.py",
    "src/generate.py",
    "tests/__init__.py",
    "tests/test_structure.py"
)
Write-Host "`n[2] Checking Required Files:" -ForegroundColor Yellow
foreach ($f in $files) {
    if (Test-Path -Path $f -PathType Leaf) {
        Write-Host "  [PASS] File exists: $f" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] File missing: $f" -ForegroundColor Red
        $passed = $false
    }
}

# 3. Check config.py parameters
Write-Host "`n[3] Validating config.py Parameters:" -ForegroundColor Yellow
$configContent = Get-Content -Path "config.py" -Raw
$requiredParams = @(
    "batch_size",
    "block_size",
    "vocab_size",
    "n_embd",
    "n_head",
    "n_layer",
    "dropout",
    "learning_rate",
    "max_iters",
    "eval_interval",
    "eval_iters"
)
foreach ($param in $requiredParams) {
    if ($configContent -match "\b$param\b") {
        Write-Host "  [PASS] Parameter defined: $param" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] Parameter missing in config.py: $param" -ForegroundColor Red
        $passed = $false
    }
}

# 4. Check requirements.txt dependencies
Write-Host "`n[4] Validating requirements.txt Dependencies:" -ForegroundColor Yellow
$reqContent = Get-Content -Path "requirements.txt" -Raw
$requiredDeps = @("torch", "numpy", "tqdm", "requests")
foreach ($dep in $requiredDeps) {
    if ($reqContent -match "\b$dep\b") {
        Write-Host "  [PASS] Dependency present: $dep" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] Dependency missing in requirements.txt: $dep" -ForegroundColor Red
        $passed = $false
    }
}

# 5. Check README.md contents
Write-Host "`n[5] Validating README.md Documentation & Roadmap:" -ForegroundColor Yellow
$readmeContent = Get-Content -Path "README.md" -Raw
$readmeTerms = @(
    "Goal of MiniGPT",
    "Pretraining vs",
    "causal self-attention",
    "Transformer blocks",
    "Phase 1: Project foundation",
    "Phase 2: Tokenizer",
    "Phase 3: Dataset and batching",
    "Phase 4: Transformer architecture",
    "Phase 5: Training loop",
    "Phase 6: Validation and checkpoints",
    "Phase 7: Text generation",
    "Phase 8: Experiments and improvements"
)
foreach ($term in $readmeTerms) {
    if ($readmeContent -match [regex]::Escape($term)) {
        Write-Host "  [PASS] Section/Topic found: $term" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] Section/Topic missing: $term" -ForegroundColor Red
        $passed = $false
    }
}

# 6. Check .gitignore rules
Write-Host "`n[6] Validating .gitignore entries:" -ForegroundColor Yellow
$gitContent = Get-Content -Path ".gitignore" -Raw
$gitTerms = @(".venv", "__pycache__", "*.pyc", "checkpoints/*", "data/*", ".vscode")
foreach ($g in $gitTerms) {
    if ($gitContent.Contains($g)) {
        Write-Host "  [PASS] Ignore rule present: $g" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] Ignore rule missing: $g" -ForegroundColor Red
        $passed = $false
    }
}

Write-Host "`n==========================================" -ForegroundColor Cyan
if ($passed) {
    Write-Host "STRUCTURE CHECK PASSED: All requirements verified successfully!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "STRUCTURE CHECK FAILED: Please fix above issues." -ForegroundColor Red
    exit 1
}
