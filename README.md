# 🧠 CogGap: Cognitive Gap Evaluation Framework

This repository provides a unified framework for evaluating Large Language Models (LLMs) on cognitive gap tasks, including natural language inference (NLI) and factual correctness evaluation.

We implement a modular pipeline for running experiments across multiple LLM providers (OpenAI, Azure OpenAI, Google Gemini, Ollama) with support for different prompting strategies and comprehensive evaluation metrics.

---

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Running Experiments](#running-experiments)
  - [Evaluating Results](#evaluating-results)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Data Format](#data-format)
- [Reproducibility](#reproducibility)
- [Development](#development)
- [Citation](#citation)

---

## ✨ Features

- ✅ Unified command-line interface for experiments and evaluation
- ✅ Support for multiple LLM providers: OpenAI, Azure OpenAI, Google Gemini, and Ollama
- ✅ Two task types: Natural Language Inference (NLI) and Factual Correctness
- ✅ Multiple prompting strategies: Direct and Chain-of-Thought (CoT)
- ✅ Comprehensive evaluation metrics: Accuracy and F1 score
- ✅ Flexible output formats: Table, JSON, and LaTeX
- ✅ Modular, extensible architecture for easy customization

---

## 🛠️ Installation

### 1. Clone this repository

```bash
git clone <repository-url>
cd CogGap
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up configuration

Copy the example configuration file and add your API keys:

```bash
cp config/config.yaml.example config/config.yaml
```

Edit `config/config.yaml` with your API credentials:

```yaml
gemini:
  api_key: "your-gemini-api-key"

lunar-deepseek-r1:
  api_key: "your-azure-api-key"
  version: "2024-02-15-preview"
  endpoint: "https://your-endpoint.openai.azure.com/"

openai:
  api_key: "your-openai-api-key"
```

**Note:** API keys can also be provided via command-line arguments or environment variables. See [Configuration](#configuration) for details.

---

## 🚀 Usage

### 🔬 Running Experiments

Run experiments across different models, tasks, and prompting strategies:

```bash
python scripts/run_experiment.py \
    --model-type azure_openai \
    --model-name deepseek-r1 \
    --task-type nli \
    --prompt-type cot \
    --data-dir data/main_task \
    --output-dir results/main_task \
    --input-files causal.json comp_ground.json epistemic.json risk.json \
    --num-runs 10
```

#### Model Types

- `openai`: OpenAI API models (GPT-4o, o3, GPT-4o-mini)
- `azure_openai`: Azure OpenAI models (deepseek-r1, lunar-gpt-4o, etc.)
- `gemini`: Google Gemini models (gemini-2.5-pro)
- `ollama`: Local Ollama models (llama3.2, etc.)

#### Task Types

- `nli`: Natural Language Inference — predict relationship between premise and statement (entailment/neutral/contradiction)
- `factual`: Factual Correctness — determine if statement is factually correct (True/False)

#### Prompt Types

- `direct`: Direct prompting without reasoning steps
- `cot`: Chain-of-Thought prompting with explicit reasoning

#### Examples

**NLI task with OpenAI:**
```bash
python scripts/run_experiment.py \
    --model-type openai \
    --model-name gpt-4o \
    --task-type nli \
    --prompt-type direct \
    --data-dir data/main_task \
    --output-dir results/main_task \
    --input-files causal.json
```

**Factual task with Gemini:**
```bash
python scripts/run_experiment.py \
    --model-type gemini \
    --model-name gemini-2.5-pro \
    --task-type factual \
    --prompt-type cot \
    --data-dir data/ctrl_GKMRV \
    --output-dir results/ctrl_GKMRV \
    --input-files comp_ground.json
```

**Local model with Ollama:**
```bash
python scripts/run_experiment.py \
    --model-type ollama \
    --model-name llama3.2 \
    --task-type nli \
    --prompt-type cot \
    --data-dir data/main_task \
    --output-dir results/main_task \
    --input-files causal.json \
    --ollama-model-name llama3.2:latest
```

---

### 📊 Evaluating Results

Evaluate model predictions against gold labels:

```bash
python scripts/evaluate.py \
    --results-dir results/main_task \
    --data-dir data/main_task \
    --output-format table \
    --metric accuracy
```

#### Output Formats

- `table`: Human-readable table format
- `json`: JSON format for programmatic access
- `latex`: LaTeX table format for papers

#### Metrics

- `accuracy`: Accuracy score (correct / total)
- `f1`: F1 score (macro-averaged)

#### Examples

**Evaluate all models:**
```bash
python scripts/evaluate.py \
    --results-dir results/main_task \
    --data-dir data/main_task \
    --output-format table
```

**Evaluate specific model:**
```bash
python scripts/evaluate.py \
    --results-dir results/main_task \
    --data-dir data/main_task \
    --model gpt-4o \
    --output-format json
```

**Generate LaTeX table:**
```bash
python scripts/evaluate.py \
    --results-dir results/main_task \
    --data-dir data/main_task \
    --output-format latex \
    --metric f1
```

---

## 📁 Project Structure

```
.
├── config/                  # Configuration files
│   ├── config.yaml.example  # Example configuration template
│   └── config.yaml          # Your API keys (gitignored)
├── data/                    # Input data files
│   ├── main_task/          # Main task datasets
│   ├── GKMRV/             # GKMRV dataset
│   └── ctrl_GKMRV/        # Control GKMRV dataset
├── prompts/                # Prompt templates
│   ├── nli/               # NLI task prompts
│   │   ├── direct.txt
│   │   └── cot.txt
│   └── factual/           # Factual correctness prompts
│       ├── direct.txt
│       └── cot.txt
├── src/                    # Core Python modules
│   ├── models/            # LLM client implementations
│   │   ├── base.py
│   │   ├── openai_client.py
│   │   ├── gemini_client.py
│   │   └── ollama_client.py
│   ├── experiments/       # Experiment running logic
│   │   ├── config.py
│   │   └── runner.py
│   └── evaluation/        # Evaluation and metrics
│       ├── evaluator.py
│       └── metrics.py
├── scripts/               # CLI entry points
│   ├── run_experiment.py  # Run experiments
│   ├── evaluate.py        # Evaluate results
│   └── migrate_data.py    # Migrate old data structure
├── results/               # Experiment results (gitignored)
│   └── {dataset}/
│       └── {model}/
│           └── *.json
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Configuration

### Configuration File

The main configuration file is `config/config.yaml`. See `config/config.yaml.example` for a template.

### API Key Options

You can provide API keys in three ways (in order of precedence):

1. **Command-line argument:** `--api-key YOUR_KEY`
2. **Configuration file:** `config/config.yaml`
3. **Environment variable:** `OPENAI_API_KEY`, etc.

### Azure OpenAI Configuration

For Azure OpenAI models, you need:

```yaml
lunar-{model-name}:
  api_key: "your-azure-api-key"
  version: "2024-02-15-preview"
  endpoint: "https://your-resource.openai.azure.com/"
```

---

## 📄 Data Format

### Input Data

Input data files should be JSON with the following structure:

```json
{
  "example_id_1": {
    "premise": "Clinical information or context...",
    "statement": "Statement to evaluate...",
    "label": "entailment"  // For NLI: "entailment", "neutral", "contradiction"
                           // For factual: "True", "False"
  },
  "example_id_2": {
    ...
  }
}
```

### Results Format

Results are saved as JSON files:

```json
{
  "example_id_1": {
    "premise": "...",
    "statement": "...",
    "label": "...",
    "responses": [
      "Model response for run 1...",
      "Model response for run 2...",
      ...
    ]
  }
}
```

---

## 🔁 Reproducibility

To reproduce experiments:

1. **Set up environment:**
   ```bash
   pip install -r requirements.txt
   cp config/config.yaml.example config/config.yaml
   # Edit config.yaml with your API keys
   ```

2. **Run experiments:**
   ```bash
   python scripts/run_experiment.py \
       --model-type azure_openai \
       --model-name deepseek-r1 \
       --task-type nli \
       --prompt-type cot \
       --data-dir data/main_task \
       --output-dir results/main_task \
       --input-files causal.json comp_ground.json epistemic.json risk.json \
       --num-runs 10
   ```

3. **Evaluate results:**
   ```bash
   python scripts/evaluate.py \
       --results-dir results/main_task \
       --data-dir data/main_task \
       --output-format table
   ```

### Migrating from Old Structure

If you have data in the old `cl_results_and_prompts/` structure, use the migration script:

```bash
python scripts/migrate_data.py
```

See `MIGRATION_GUIDE.md` for detailed migration instructions.

---

## 🔧 Development

### Adding a New Model Provider

1. Create a new client class in `src/models/` inheriting from `BaseModel`:

```python
from .base import BaseModel

class NewModelClient(BaseModel):
    def generate(self, prompt: str, **kwargs) -> str:
        # Implement generation logic
        pass
```

2. Register it in `src/experiments/runner.py`:

```python
def create_model_client(config: ExperimentConfig) -> BaseModel:
    # ... existing code ...
    elif config.model_type == "new_provider":
        return NewModelClient(...)
```

3. Update `scripts/run_experiment.py` to support the new provider.

### Adding a New Evaluation Metric

1. Add the metric function to `src/evaluation/metrics.py`:

```python
def calculate_new_metric(y_true, y_pred):
    # Implementation
    return score
```

2. Update `src/evaluation/evaluator.py` to use the new metric.

3. Add the metric option to `scripts/evaluate.py`.

---

## 📄 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{coggap2025,
  title={CogGap: Cognitive Gap Evaluation Framework},
  author={[Your Name]},
  year={2025},
  howpublished={\url{https://github.com/[your-username]/CogGap}}
}
```

---

## 📝 License

[Specify your license here]

---

## 🙏 Acknowledgments

[Add any acknowledgments here]
