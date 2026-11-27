# rag_evaluation

A modular evaluation framework for RAG applications (built around the airport-kiosk assistant, usable with any RAG system).
It has two parts:

1. **`golden_dataset_generator.py`** – turns Markdown documents into a validated *golden dataset*
   (`question`, `contexts`, `ground_truth`, `answer`) using the RAGAS `TestsetGenerator`.
2. **`metrics/eval_*.py`** – ten independent metric modules (RAGAS + classic NLP metrics). Each one runs on its own
   from the command line, or is called by **`run_all_evaluations.py`**, which consolidates everything into one report.

```
rag_evaluation/
├── config.yaml                  # models, paths, thresholds (no secrets)
├── .env.example                 # copy to .env and add API keys
├── golden_dataset_generator.py
├── run_all_evaluations.py
├── requirements.txt
├── metrics/
│   ├── eval_common.py           # shared helpers (config, loading, reports, judge-LLM factory) - the only shared file
│   ├── eval_context_precision.py   eval_context_recall.py   eval_mrr.py   eval_ndcg.py   eval_accuracy.py
│   └── eval_answer_relevance.py    eval_faithfulness.py     eval_bleu.py  eval_rouge.py  eval_bertscore.py
├── sample_data/                 # airport_sample.md + a hand-made sample_golden_dataset.json (offline demo)
├── tests/                       # offline pytest suite (fake judge LLM, no API keys)
└── results/                     # reports are written here
```

## 1. Setup

Python 3.10, 3.11 or 3.12.

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env                 # Windows: Copy-Item .env.example .env
```

Put your key in `.env` (or export it). Keys are **never** read from `config.yaml`, which only stores the *name* of the variable.

```bash
export OPENAI_API_KEY=...            # PowerShell: $env:OPENAI_API_KEY = "..."
export ANTHROPIC_API_KEY=...         # only if you use Anthropic as judge/generator
```

`bert-score` needs PyTorch. `pip install -r requirements.txt` installs the default build; for a smaller CPU-only install use
`pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu` *before* the requirements file.
The first BERTScore run downloads the encoder from Hugging Face (cached afterwards).

### Choose the judge / generator model (OpenAI ⇄ Anthropic)

All LLM-judged metrics (context precision/recall, faithfulness, answer relevance) read `judge_llm` in `config.yaml`;
synthetic question generation reads `generator_llm`. Switch provider without editing files:

```bash
RAG_EVAL_JUDGE_PROVIDER=anthropic RAG_EVAL_JUDGE_MODEL=claude-sonnet-4-5 python run_all_evaluations.py
```

Anthropic has no embeddings API, so `embeddings.provider` stays `openai` or `huggingface` (local; requires
`pip install langchain-huggingface sentence-transformers`). Faithfulness is the hallucination / factual-consistency judge.

## 2. Create the golden dataset

```bash
# Synthetic: 20 questions from one Markdown file (or a folder of .md files)
python golden_dataset_generator.py --source sample_data/airport_sample.md --num-questions 20 --output golden_dataset.json

# Control the question mix; also write a JSON Lines copy for Hugging Face `datasets`
python golden_dataset_generator.py --source ../kiosk/data -n 50 \
    --distribution simple=0.5,reasoning=0.25,multi_context=0.25 --jsonl

# Evaluate the real kiosk retriever + generator instead of the LLM baseline
python golden_dataset_generator.py --source ../kiosk/data --answer-provider kiosk_app

# Skip synthesis and use hand-written questions (JSON list: question, ground_truth, reference_contexts)
python golden_dataset_generator.py --source docs.md --seed-file my_questions.json
```

**Question types** map to RAGAS query synthesizers: `simple` → single-hop specific, `reasoning` → multi-hop abstract,
`multi_context` → multi-hop specific. Weights are re-normalised. If a small document cannot support multi-hop questions the
generator retries with `simple` only.

**Where `contexts` and `answer` come from** (`generation.answer_provider`):

| Provider | `contexts` | `answer` | Use it for |
|---|---|---|---|
| `llm_baseline` (default) | RAGAS ground-truth chunks | generator LLM answers only from them | sanity check; generation metrics. Retrieval metrics are trivially perfect here. |
| `kiosk_app` | what the kiosk retriever returns (rank order) | kiosk generator, after the kiosk's evidence gate and validator (fallback phrase when it refuses) | measuring the real system. Set `kiosk_app.root` and give it the kiosk's own env vars. |

You can plug in any other system by implementing `AnswerProvider.answer(question, reference_contexts)`.

**Output** (`golden_dataset.json`):

```json
{ "schema_version": "1.0",
  "metadata": { "ragas_version": "...", "realised_question_types": {...}, "validation": {...} },
  "samples": [ { "question": "...", "contexts": ["..."], "ground_truth": "...", "answer": "...",
                 "reference_contexts": ["..."], "question_type": "simple", "source": "airport_sample.md" } ] }
```

**Validation** runs before export: schema and types, null/blank fields, blank contexts, too-short questions, exact and near
duplicates (`validation.near_duplicate_threshold`). Invalid records are dropped and listed in `metadata.validation.issues`;
`--strict` raises instead. Every metric module accepts this file, a bare JSON list, or JSON Lines, and tolerates the RAGAS field
names (`user_input`, `response`, `reference`, `retrieved_contexts`).

> The tiny sample document is only a smoke test. RAGAS builds a knowledge graph and needs richer, longer documents
> (for example the kiosk's `data/` folder) to produce good multi-hop questions. Review generated questions before trusting them.

## 3. Run evaluations

### One metric (standalone)

```bash
python metrics/eval_mrr.py --input golden_dataset.json --output results/mrr_report.json
python metrics/eval_faithfulness.py --input golden_dataset.json --output results/faithfulness_report.json
python metrics/eval_context_precision.py --input golden_dataset.json --no-llm     # Precision@K only, no API key
python metrics/eval_bleu.py --backend nltk --scale percent
python metrics/eval_rouge.py --types rouge1 rougeL --no-stemmer
python metrics/eval_accuracy.py --mode exact_match
python metrics/eval_bertscore.py --model-type roberta-large --device cuda
```

Every module accepts `--input`, `--output`, `--config`, `--limit N`, `--no-csv`, `--log-level`, plus its own options (`--help`).
Defaults come from `config.yaml`. Exit code `0` = ok, `2` = configuration/dataset/dependency problem, `3` = nothing could be scored.

### Full pipeline

```bash
python run_all_evaluations.py --input golden_dataset.json
python run_all_evaluations.py --include mrr ndcg rouge bleu            # only these
python run_all_evaluations.py --exclude bertscore faithfulness         # everything else
python run_all_evaluations.py --skip-llm                               # no judge LLM / no API key
python run_all_evaluations.py --formats json html --limit 10
python run_all_evaluations.py --list
```

A metric that fails (e.g. missing key) is recorded as `failed` in the summary and the others still run (`--fail-on-error` makes it an error exit).
Outputs in `results/`: `<metric>_report.json/.csv`, `summary_report.json/.csv/.html`, and `per_sample_scores.csv`.

### Report format

```json
{ "metric": "mrr", "headline": {"name": "mrr", "value": 0.667},
  "aggregate": { "mrr": 0.667, "hit_rate@3": 0.833, "count_scored": 6 },
  "metadata": { "dataset": "...", "duration_seconds": 0.02, "relevance_mode": "auto" },
  "per_sample": [ { "index": 0, "question": "...", "reciprocal_rank": 1.0 } ] }
```

## 4. Metric reference

| Module | Metric(s) | Library | Judge LLM |
|---|---|---|---|
| `eval_context_precision.py` | RAGAS Context Precision; Precision@K | ragas + custom | yes (`--no-llm` skips it) |
| `eval_context_recall.py` | RAGAS Context Recall; Recall@K | ragas + custom | yes (`--no-llm` skips it) |
| `eval_mrr.py` | MRR, Hit-Rate@K | custom | no |
| `eval_ndcg.py` | NDCG@K (binary or `--graded`) | scikit-learn | no |
| `eval_accuracy.py` | accuracy by exact match / contains / token-F1 | scikit-learn + custom | no |
| `eval_answer_relevance.py` | RAGAS Answer Relevancy | ragas | yes + embeddings |
| `eval_faithfulness.py` | RAGAS Faithfulness (hallucination flag below 0.8) | ragas | yes |
| `eval_bleu.py` | sentence + corpus BLEU | sacrebleu / nltk | no |
| `eval_rouge.py` | ROUGE-1/2/L P/R/F1 | rouge-score | no |
| `eval_bertscore.py` | BERTScore P/R/F1 | bert-score | no (local model) |

**Relevance labels for rank metrics.** Precision@K, Recall@K, MRR and NDCG need to know which retrieved chunks are relevant.
They are labelled by *text overlap* (`evaluation.relevance` in config): a retrieved chunk is relevant if it overlaps a ground-truth
chunk (`reference_contexts`) by at least `threshold`, or, when a sample has none, if it contains that share of the reference answer's
content words. Overlap is boundary-tolerant because the kiosk and RAGAS chunk documents differently, but it is lexical: tune the
threshold on a few samples you inspected by hand, or supply your own labels.

**RAGAS-specific configuration** (commented in the code): RAGAS 0.4 metrics live in `ragas.metrics.collections` and take an
*instructor-based* LLM from `llm_factory(model, provider, client)`, built here with **async** OpenAI/Anthropic clients; Answer
Relevancy additionally takes `embedding_factory(...)`. `TestsetGenerator` still uses the LangChain wrappers
(`LangchainLLMWrapper`, `LangchainEmbeddingsWrapper`), which is why both styles appear in the code. LangChain is pinned to 0.3.x because
RAGAS 0.4.3 imports a module that `langchain-community` 0.4 removed.

## 5. Tests

```bash
python -m pytest tests -q        # offline: fake judge LLM, no keys, no network
```

## 6. Known limitations

* LLM-judged scores cost tokens and vary slightly between runs/judges; keep `temperature: 0` and compare runs with the same judge.
* The kiosk's fallback answer ("I can't confirm that…") correctly scores ~0 on answer relevance; read it together with `accuracy`/`faithfulness`.
* The `kiosk_app` provider bypasses the kiosk guardrail and intent router on purpose (it evaluates the RAG path, not the policy rules).
