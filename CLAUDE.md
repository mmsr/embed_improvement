# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Numeracy-focused embedding fine-tuning project: improves how sentence embedding models
(ModernBERT, `answerdotai/ModernBERT-base`) numerically discriminate text via contrastive/triplet
learning. The pipeline builds anchor/positive/negative triplets that differ mainly in numeric
values and magnitude, rewrites them with an LLM for lexical diversity, trains on Colab, and
evaluates numeracy-aware retrieval (Recall@K, MRR, triplet accuracy).

## Commands

Dependency management is via `uv` (Python >=3.10, see `uv.lock`).

- Install deps: `uv sync`
- Run any script: `uv run python <script>.py`
- No test suite, linter, or CI is configured in this repo — do not assume pytest/ruff/etc. exist.

Key standalone scripts (all plain argparse/script entry points, run with `uv run python`):
- `utils/build_dataset.py` — builds triplets from the raw `Numeracy_600K_comment.json` source,
  sampling by target magnitude distribution.
- `utils/build_sentences.py` — substitutes anchor/positive/negative numbers into `comment`
  templates to produce sentence-level triplets.
- `data_rewriter/main.py` — async LLM rewriting pass (OpenAI `gpt-4.1-nano` by default) over
  sentence triplets; checkpointed/resumable, reads/writes JSONL.
- `data_rewriter/data_splitter.py` — splits rewritten triplets into train/val/test files.
- `utils/analyze_magnitude_relationships.py` / `analyze_dataset.py` — QA/health-check reports
  over a triplet JSONL file at any pipeline stage (magnitude coverage, duplicates, degenerate
  triplets, tokenization impact).
- `sample_numeracy.py` — random reproducible sampling of N records from the raw dataset.
- `utils/evaluate.py` — loads a HF/ModernBERT checkpoint and computes embedding retrieval
  metrics (Recall@K, MRR, triplet accuracy, cosine gap) over a triplet JSONL file.

Model training itself happens on Colab (not runnable locally as-is — needs `torch`/`transformers`/
`peft`/`accelerate`, none of which are in `pyproject.toml`; these are installed inline in Colab).
The actual training logic lives in `training/` (importable from Colab after cloning/syncing this
repo onto Drive), not in the notebooks themselves:

- `training/config.py` — `RunConfig` dataclass; every hyperparameter that varies between
  experiments (LoRA shape, loss choice, LRs, data files) lives here, not hardcoded in a notebook.
- `training/data.py` — `TripletDataset`, `make_triplet_collator`.
- `training/model.py` — `ContrastiveModel`, `build_model()` (attaches a fresh LoRA adapter from
  config, or continues training an existing one via `config.init_from_adapter`).
- `training/losses.py` — `LOSS_REGISTRY`; loss variants selected by name via `RunConfig.loss_fn`.
- `training/train.py` — `train_one_run(config, work_dir)`, the full training loop.
- `training/experiment.py` — `new_run_id`, `save_run`, `save_checkpoint`, `log_result`,
  `load_registry`: run naming, checkpoint saving, and results logging.
- `training/evaluate.py` — triplet metrics + numeric-ordering probe suites (Spearman vs
  log-distance); `evaluate_adapter()` returns a flat metrics dict for `log_result`. Supersedes
  the ad-hoc eval in `embed_eval_clean.ipynb` and `utils/evaluate.py` (which loads a model at
  import time — don't import it as a library).
- `colab/train_run.ipynb` — thin Colab entry point: mount Drive, sync repo, set a `RunConfig`,
  train, evaluate, and log to the registry in one session.

See `EXPERIMENTS.md` for why this exists and how it maps onto the original notebooks.

## Architecture: pipeline stages

Data flows through these stages in order; each stage's script/notebook reads the previous
stage's JSONL output from `data/`:

1. **Raw source** — `data/Numeracy_600K_comment.json` (~128MB raw financial-comment dataset).
2. **Triplet building** — `utils/build_dataset.py` (or the older prototype in
   `data_explore.ipynb`) samples anchors and generates positive/negative numbers per anchor by
   magnitude shift, validating log-distance signal quality.
3. **Sentence building** — `utils/build_sentences.py` substitutes those numbers into the
   `comment` text to produce `positive`/`negative` sentence fields (`train_sentences.jsonl`).
4. **LLM rewriting** — `data_rewriter/` (`main.py` orchestrator, `llm/` clients for
   OpenAI/Gemini, `rewrite/` prompt+validation, `utils/` batching+checkpoint helpers) paraphrases
   sentences for lexical diversity while preserving numbers exactly, to reduce shortcut learning.
   `langgraph_text_formater.ipynb` is an exploratory LangGraph/Ollama alternative to this stage.
5. **Splitting** — `data_rewriter/data_splitter.py` splits rewritten triplets into
   train/val/test-same/test-general files.
6. **QA/health checks** — `utils/analyze_magnitude_relationships.py` and `analyze_dataset.py`
   can be run against the output of any stage above to check magnitude coverage, duplicates, and
   degenerate triplets before training.
7. **Training** — `colab/train_run.ipynb` (thin entry point) + `training/train.py`
   (`train_one_run`) fine-tune ModernBERT with LoRA on Colab GPU using the split triplet data,
   config-driven via `training/config.py`. `colab/embed_trainer*.ipynb` (7 notebooks, near-
   duplicates of each other with only LoRA config/loss/data differing) are the original
   implementation this was extracted from — kept as reference, superseded, not deleted. See
   `EXPERIMENTS.md`.
8. **Evaluation** — `embed_eval_clean.ipynb` / `colab/embed_eval_clean (1).ipynb` /
   `utils/evaluate.py` compute Recall@K, MRR, and hand-built numeric-bias probes (e.g.
   left-to-right digit bias, monotonic magnitude decay) against test splits. Results historically
   were pasted as raw text into `colab/test_results/*.xlsx`; new results should go through
   `training.experiment.log_result` into a `runs/registry.jsonl` instead (see `EXPERIMENTS.md`).

Root-level `data_explore*.ipynb`, `data_train_explorer.ipynb`, and `Untitled.ipynb` are
exploratory/scratch notebooks, not pipeline entry points — treat them as prototypes/history for
the scripts above rather than sources of truth. `bot_interview` is an unrelated plaintext notes
file, not a module.
