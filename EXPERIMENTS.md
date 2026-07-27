# Organizing LoRA training runs and results

## What went wrong

Inspected all 7 notebooks in `colab/` and both files in `colab/test_results/`. Two concrete root causes explain "didn't save results and model names properly":

1. **The final save cell in every trainer notebook is hardcoded and ignores the run's own config.**
   Every variant (`embed_trainer.ipynb`, `embed_trainer_2.ipynb`, `embed_trainer_3.ipynb`,
   `embed_trainer_contrastive_corrected_CDL_GPTl.ipynb`,
   `Copy of embed_trainer_contrastive_CDL_GPTl_hyper.ipynb`,
   `embed_trainer_infosec_4132026.ipynb`) ends with:
   ```python
   unwrapped.encoder.save_pretrained("modernbert_lora_contrastive_corrected_dynamic")
   tokenizer.save_pretrained("modernbert_lora_contrastive_corrected_dynamic")
   ```
   This literal string has nothing to do with each notebook's own `LORA_MODEL` variable
   (e.g. `modernbert_lora_train3_4132026`, `modernbert_lora_4112026`, ...), and it saves to the
   **local Colab runtime disk**, not `WORK_DIR` on Drive — so it's wiped the moment the runtime
   disconnects unless someone remembers to copy it out. A second, better save cell later in some
   notebooks does write to Drive (`WORK_DIR/trained_model/..._epoch_{epoch+1}`), but the base name
   there (`modernbert_lora_contrastive_corrected_ML`) is also copy-pasted identically across
   variants with materially different LoRA configs/losses/data, so different runs collide on the
   same folder name on Drive.

2. **Results are hand copy-pasted as raw console text into Excel**, with columns named things like
   `ModrenBERT`, `CDL3_r`, `CLD3 old`, `CLD1 Old`, `corrected_ML_epoch_2`, `4/12/2025 epoch4`, or
   a bare date — no consistent link from a results column back to the LoRA config/checkpoint that
   produced it. `utils/evaluate.py` already computes clean numeric metrics
   (`recall@1/5/10`, `mrr@1/5/10`, `triplet_accuracy`, `cosine_gap`) — that output was being
   discarded in favor of pasting console text.

The 7 trainer notebooks are otherwise ~90% identical (Drive mount, imports, dataset/model classes,
training loop) with only the LoRA config, loss function, or data file differing — each new
experiment was a full copy of the previous notebook rather than a config change.

## Structure

`embed_trainer_3.ipynb` (the latest/current trainer) has been fully extracted into a proper
`training/` package. The other 6 notebooks are untouched and kept only as reference.

```
training/
  config.py      # RunConfig — every hyperparameter that varies between experiments
  data.py        # TripletDataset, make_triplet_collator
  model.py       # mean_pooling, ContrastiveModel, build_model() (fresh LoRA or continue an adapter)
  losses.py      # LOSS_REGISTRY — loss variants from the notebooks, selected by name
  train.py       # train_one_run(config, work_dir) — the whole training loop
  experiment.py  # new_run_id, save_run, save_checkpoint, log_result, load_registry
colab/
  train_run.ipynb      # new: thin entry point, config cell + train_one_run() call
  embed_trainer_3.ipynb # unchanged, kept as reference
  <other 5 trainer notebooks>  # unchanged, superseded by the above
  test_results/
runs/                   # created on Drive at runtime, e.g. WORK_DIR/runs/
  registry.jsonl        # one line per run: full config + eval metrics, append-only
  <run_id>/
    adapter/             # final LoRA adapter + tokenizer (save_run)
    checkpoints/
      epoch_<n>/          # per-epoch adapter snapshot (save_checkpoint) — inspect/compare any epoch
    accelerator_state/    # raw resumable training state (optimizer, RNG) — not a model checkpoint
    config.json           # exact RunConfig used, alongside the final adapter
```

JSONL (not CSV) for the registry so new hyperparameters/metrics can be added over time without a
fixed schema breaking old rows.

### What changed vs. the notebook

- **LoRA config and loss function are now config, not code.** `RunConfig.lora_r/lora_alpha/...`
  and `RunConfig.target_modules` drive `build_model()`; `RunConfig.loss_fn` (a string) selects a
  function from `training.losses.LOSS_REGISTRY`. Trying a new combination is a `RunConfig(...)`
  edit in the notebook's config cell, not a copy of the whole notebook. Add a new loss variant by
  writing one function in `losses.py` and adding it to `LOSS_REGISTRY` — nothing else changes.
- **`embed_trainer_3.ipynb` actually had two definitions of `improved_numeric_loss`**; the second
  (cell 13) silently shadowed the first at runtime, so the first was dead code, and it wasn't
  obvious which one had actually trained past runs. Both are preserved under honest names —
  `dynamic_margin_sigmoid_loss` (the first, unused-in-practice version) and
  `dynamic_margin_softplus_loss` (the second, the one that was actually training) — so either can
  be selected deliberately, and it's explicit in `registry.jsonl` which one produced a given run.
- **`embed_trainer_infosec_4132026.ipynb`** (the latest loss code tried) has the same
  `improved_numeric_loss` math as `dynamic_margin_softplus_loss` with one real difference: the
  rank-loss margin is `0.3` there, not `0.1`. Added as a third registry entry,
  `dynamic_margin_softplus_infosec_loss` (`loss_fn="dynamic_margin_softplus_infosec"`), rather
  than parameterizing the existing function, so both stay selectable and neither's behavior
  changes silently. That notebook's full recipe (this loss + `tok_embeddings` LoRA + unfreezing
  ModernBERT's norm layers via `modules_to_save`) scored ordering rho=0.73 in the earlier
  analysis — worse than the untouched trainer_3-era CLD3-old's 0.93 — so it isn't the default;
  it's there to A/B against `dynamic_margin_softplus` directly by changing one config field.
  The LoRA side of that notebook (adapting `tok_embeddings`, plus `modules_to_save=["attn_norm",
  "mlp_norm", "emb_norm"]` to fully fine-tune ModernBERT's LayerNorms alongside the LoRA adapters)
  is now expressible via `RunConfig.target_modules` / `RunConfig.modules_to_save` (new field,
  `None` by default = LoRA-only, matching `embed_trainer_3.ipynb`).
- **Every run gets its own `run_id` and its own directory**, generated once via `new_run_id()` and
  reused consistently for the accelerator resume state, every epoch checkpoint, and the final
  adapter. Previously `CHECKPOINT_DIR = WORK_DIR/checkpoints_final_margin` was one fixed path
  shared by every notebook variant, so resuming or checkpointing one experiment could silently
  read/overwrite state left by a completely different experiment.
- **Resume is now explicit and correct.** Set `RunConfig(resume=True, resume_run_id="<the id>")`
  to continue that specific run from its own `accelerator_state/`. The old notebook worked out
  in-epoch resume via a hardcoded `if epoch == 1 and step < 800: continue`; `train.py` computes
  the equivalent skip generically from the saved `global_step`, so it isn't tied to one specific
  interrupted run.
- **`wandb.init()` is now tagged with the run's own `name=run_id` and `config=asdict(config)`**,
  instead of every variant sharing `WANDB_PROJECT = "modernbert-numeracy-lora-corrected-final"`
  with no per-run distinction in the dashboard.

## Running it on Colab

Use `colab/train_run.ipynb` — it mounts Drive, clones/pulls this repo onto Drive so `training/` is
importable, installs Colab-only deps (`transformers peft accelerate wandb`), then has a single
CONFIG cell building a `RunConfig` and one cell calling `train_one_run(config, WORK_DIR)`. Update
`REPO_URL` in the sync cell to point at wherever you push this repo.

To start a new experiment: change the `RunConfig` fields in that one cell (LoRA shape, `loss_fn`,
data files, LRs) and re-run — no notebook copying. To resume an interrupted run: set
`resume=True, resume_run_id="<id from the earlier run>"`.

After training, run `utils/evaluate.py`'s `evaluate(triplet_file)` against
`runs/<run_id>/adapter` and log the result:

```python
from training.experiment import log_result
metrics = evaluate(triplet_file)
log_result(WORK_DIR, run_id, config, metrics, eval_file=triplet_file)
```

Compare all runs with `training.experiment.load_registry(WORK_DIR)` (a pandas DataFrame) instead
of an Excel sheet of pasted console text.

## Results ↔ notebook mapping (reconstructed 2026-07-26)

Ordering score ρ = Spearman correlation between similarity ranking and true |log ratio|
ranking across the probe suites in `Test Results*.xlsx` (1.0 = perfect numeric ordering).
Mapping reconstructed from notebook CONFIG cells, save paths, and dates — confidence noted.

| Excel column | ρ (mean) | Notebook / checkpoint | Train data | Confidence |
|---|---|---|---|---|
| ModrenBERT | −0.13 | base `answerdotai/ModernBERT-base`, no fine-tune | — | high |
| constrastive_dynamic_numeric | 0.73 | `embed_trainer.ipynb` (attn-only LoRA, pure dynamic-margin triplet loss, save name `…contrastive_corrected_dynamic`) | train_extracted.jsonl | high |
| corrected_ML_epoch_0 / _2 | 0.79 / 0.74 | `embed_trainer_2.ipynb` per-epoch saves (`…corrected_ML_epoch_{n}`) | train_extracted_improved_nodup (hard negatives) | high |
| CDL_GPTl_epoch_9 | 0.87 | a 9-epoch CDL_GPTl run: `embed_trainer_contrastive_corrected_CDL_GPTl.ipynb` or `embed_trainer_infosec_4132026.ipynb` (both loop EPOCHS+6, both +tok_embeddings) | train.jsonl | medium |
| CDL3_r | 0.84 | probably `embed_trainer_3.ipynb` ("3"; `_r` = rewritten/resumed?) | train.jsonl | low |
| CLD3 / CLD3 old / CLD1 Old | 0.78 / **0.93** / 0.88 | "CLD" ≈ typo of CDL; cannot be pinned to a notebook or epoch. **CLD3 old is the best model ever produced and is unattributable** — the motivating case for this registry. | unknown (likely improved_nodup era) | low |
| 4/12/2025 epoch4 / epoch3 | 0.79 / (partial) | `…corrected_CDL_GPTl.ipynb` (`modernbert_lora_4112026`, run 4/11–4/12), epochs 3–4 | train.jsonl | medium |
| 2025-04-14 | 0.73 | `embed_trainer_infosec_4132026.ipynb` (dated 4/13) | train.jsonl | medium |

Key data fact behind the scores: `train_triplets_04092026.jsonl` has negatives at median
50× from the anchor (1.4% within 1.2×) → near-tie ordering collapses; the older
`train_extracted_improved_nodup.jsonl` had 33.6% of negatives within 1.2×.

## Data preparation: lineage, root causes, and the 2026-07-26 fix

Two prep chains produced the training files:

- **Latest chain** (April 2026, fed trainer_3 / CDL_GPTl / infosec runs):
  `utils/build_dataset.py` → `train_triplets_04092026.jsonl` → `utils/build_sentences.py` →
  `train_sentences.jsonl` → `data_rewriter/main.py` (LLM paraphrase) → `data_rewriter/data_splitter.py`
  → `train/val/test_same/test_general.jsonl`.
- **Older chain** (best-performing era): prototype triplet gen in `data_explore.ipynb` →
  rewriting → number re-extraction via `extract_diff_numbers` + `.00`-record patching in
  `data_explore_numbers.ipynb` → dedup in `data_train_explorer.ipynb` →
  `train_extracted_improved_nodup.jsonl`.

Root causes found (all verified against the data):

1. **59.7% of negatives in `train_triplets_04092026.jsonl` were the literal fallback
   `round(max(anchor,pos)*50, 4)`.** The generator's `MIN_NEG_LOG_FACTOR=3.0` (measured on
   log1p of *linear* distance) was mutually unsatisfiable with the 10–20x magnitude ratio caps,
   so every strategy failed and fell through to the 50x last resort — which checked neither
   constraint. Hence median neg log-ratio = ln(50) = 3.91 and no near-tie training signal.
2. **`build_sentences.py` replaced the first *string match*, not the anchor occurrence** —
   e.g. "…ENDED JUNE 24.2, 2014 … TO $24.2 MILLION" mutated the date, not the amount. The
   source's `offset`/`length` fields (100% reliable, verified) were never used. Some of the
   old chain's "hard negatives" were actually artifacts of this bug plus re-extraction.
3. **`length` often excludes trailing zeros** ("5.5" span inside "5.50"), gluing leftover
   digits onto replacements — the bug `data_explore_numbers.ipynb` was patching post-hoc.
4. **Failed generations were written as nulls** (5,345 rows in 04092026) and silently skipped
   by every later stage.

Fixes (in `utils/build_dataset.py` and `utils/build_sentences.py`):

- Negatives now sampled directly by |log ratio| from graded bands `NEG_RATIO_BANDS`
  (40% at 1.05–1.5x, 30% at 1.5–5x, 30% at 5–50x), positives band-matched tighter so every
  triplet is valid by construction in both log-ratio and the loss's log1p space; `neg_band`
  recorded per row for ablations. Failures are dropped and counted, not written as nulls.
- Substitution uses the exact `offset`/`length` span, extended over the full number token;
  records whose span can't be verified (anchor embedded in a longer number, 96 rows = 0.1%)
  are dropped rather than risk corrupt text.

Regenerated (seed 42): `data/train_triplets_graded.jsonl` (96,000) →
`data/train_sentences_graded.jsonl` (95,904; 100% offset-exact, **0 substitution mismatches**).
Negative hardness: 14.9% within 1.2x, 40.3% within 1.5x, median 2.2x
(was 1.4% / 2.6% / 50x). Not yet done downstream: the LLM rewriting pass
(`data_rewriter/main.py`, costs OpenAI credits) — until then the `positive`/`negative`
fields can be trained on directly (lexical variety comes only from the rewriter). Known
quirk kept: `MAG_TARGETS` sums to 0.96, so `--total 100000` yields 96k records.

**Splits (2026-07-27, `utils/split_dataset.py`, seed 42):** the graded file had two leakage
risks a naive split would hit — 7,230 source records duplicated by replacement
oversampling (one comment appears 34x), and 1,213 records (1.3%) sharing their anchor
comment with the old `test_*_extracted.jsonl`/`val_extracted.jsonl` files. The splitter
drops the contaminated records (so old test sets stay valid for old-vs-new comparisons)
and splits by comment-group so no comment straddles splits (verified: 0 leaks). Outputs:
`data/graded_train.jsonl` (75,921), `data/graded_val.jsonl` (9,433),
`data/graded_test.jsonl` (9,337) — near/mid/far band mix preserved in each. Use
`graded_val.jsonl` for validation: the old `val.jsonl`'s negatives are the 50x kind, so
val loss on it cannot see near-tie learning.

## Core-problem diagnosis (2026-07-26) and the fixes implemented

Why many loss variants (simple triplet → infosec's complex multi-term) showed no eval
improvement — responsibility estimate:

1. **Train/eval space mismatch (~40%).** All metric losses shaped
   `normalize(metric_proj(emb))`; only the encoder was saved/evaluated. Evidence: the
   log-distance term demands cosine spread > 1.0, measured encoder-space spread was
   0.07–0.12 across every fine-tune. Loss changes were filtered through a space eval never
   saw, so they all looked alike. **Fix:** `RunConfig.use_metric_proj=False` (new default) —
   losses apply to the normalized pooled encoder output; `metric_proj` only exists when
   explicitly enabled; auxiliary heads persisted per run (`aux_heads.pt`) and restored by
   `init_from_adapter`.
2. **Data (~35%).** The 50x-fallback negatives (see Data preparation section) meant near-tie
   ordering was unconstrained. **Fix:** graded data already regenerated
   (`train_sentences_graded.jsonl`).
3. **Loss form (~20%).** The log-distance MSE is regression-to-cosine — CoSENT showed
   ranking-only objectives beat it for exactly this reason; triplets are pairwise and can't
   express listwise monotonicity; the unbounded margin kept gradient on easy far pairs;
   head_loss (~60 at init) dominated early training. **Fixes (new registry entries, existing
   losses untouched):** `cosent_log_ratio` (graded CoSENT over in-batch pairs, target
   −|log ratio|, τ=`loss_tau`), `cosent_plus_head` (+ downweighted normalized head), and
   `dynamic_margin_softplus_capped` (margin cap 0.8 + head targets /20).
4. **LoRA settings (~5%).** r=16/α=32 is fine and not the bottleneck; literature shows LoRA
   matches full fine-tuning for contrastive embedding adaptation. Default `target_modules`
   corrected to honest names `["Wqkv","Wo","Wi"]` (identical adapted set — "out_proj"
   matched nothing). tok_embeddings/norm-unfreezing (infosec) stays opt-in; it regressed.
5. **Loop hygiene:** `seed`, warmup+cosine schedule (`lr_schedule="none"` restores constant
   LR), `grad_clip`, and per-epoch ordering-ρ probes (`ordering_eval_each_epoch`) logged to
   wandb so checkpoint selection uses the metric that matters.

Key references: [CoSENT (TASLP 2024)](https://penghao-bdsc.github.io/papers/CoSENT_TASLP2024.pdf),
[MNR/in-batch negatives](https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/losses/MultipleNegativesRankingLoss.py),
[Wallace et al. 2019 numeracy probing](https://aclanthology.org/D19-1534/),
[xVal continuous number encoding](https://arxiv.org/abs/2310.02989) (longer-term option for
tokenization artifacts).

First recommended run: graded data + `use_metric_proj=False` + `loss_fn="cosent_log_ratio"`
(the pre-filled config in `colab/train_run.ipynb`), compared against the Phase-0 baselines.

## LoRA hyperparameters (researched 2026-07-27, set as defaults)

- **Learning rate `lora_lr=2e-4`** (was 5e-5): the LoRA optimum is consistently ~10x the
  full-fine-tuning LR, independent of base model ([Thinking Machines "LoRA Without
  Regret"](https://thinkingmachines.ai/blog/lora/); [LR scaling across
  ranks](https://arxiv.org/html/2602.06204v1)). The old notebooks used the full-FT-style
  2e-5. Safe with the warmup+cosine schedule and grad clipping now in the loop.
- **`target_modules=["Wqkv","Wo","Wi"]` = all linear layers**: attention-only LoRA
  significantly underperforms even at matched parameter count — MLP layers are where the
  capacity lives (Thinking Machines; also the QLoRA finding echoed in the
  [Unsloth guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)).
- **`r=16, alpha=32` (alpha = 2r)**: the standard best-practice ratio (Raschka); rank has
  little effect at this data scale (~76k examples), and optimal LR is roughly
  rank-independent due to the alpha/r prefactor. Not worth sweeping before the loss/data
  results are in.
- **`use_rslora` flag added (default False)**: alpha/sqrt(r) scaling matters only for
  r >= 64, where standard alpha/r collapses gradients
  ([rsLoRA paper](https://arxiv.org/abs/2312.03732)) — enable it for the high-rank ablation.
- **`batch_size=32` kept**: LoRA tolerates large batches worse than full fine-tuning
  (gap grows with batch size, not mitigated by rank — Thinking Machines), while 32 triplets
  already yield 64 in-batch pairs (~2k ranking comparisons) for the cosent losses. 64 is a
  reasonable A100 variant.

## Recommended Colab run sequence

All runs: graded splits, `use_metric_proj=False`, defaults above; one variable per run.

0. **Baselines (eval-only, once):** base ModernBERT (`adapter_path=None`) through the eval
   section; if the Drive copy of the CLD3-old-era adapter can be found, evaluate it too.
1. **run1 (pre-filled in the notebook):** `loss_fn="cosent_log_ratio"` — the headline recipe.
2. **run2:** `loss_fn="cosent_plus_head"` — does the auxiliary magnitude head help?
3. **run3:** `loss_fn="dynamic_margin_softplus_capped"` — best margin-family variant, for the
   loss-family comparison row.
4. **run4 (baseline row):** `loss_fn="dynamic_margin_softplus"`, `lr_schedule="none"`,
   `lora_lr=5e-5`, `use_metric_proj=True` — the old trainer_3 recipe on the new data, so the
   paper can attribute gains to loss/space fixes vs data fixes separately.
5. Then: tau ablation (0.03/0.1) on the run1-vs-run2 winner, and the LoRA ablations
   (r ∈ {8, 64 + rslora}, +tok_embeddings) only after the loss winner is fixed.

Checkpoint selection: per-epoch `ordering_mean` (logged automatically), not val loss.

## Experiment plan (for the paper)

Every run goes through `colab/train_run.ipynb` → `runs/registry.jsonl`. One change per run.

- **Phase 0 — anchors.** (a) Base ModernBERT through the eval section only (`adapter_path=None`);
  (b) exact `embed_trainer_3` recipe re-run through the new pipeline (`tag="trainer3-repro"`).
  These two registry rows are the reference points for everything else.
- **Phase 1 — fixes** (config-flagged, one at a time vs. Phase 0b): metric losses on encoder
  output instead of discarded `metric_proj`; margin cap in `dynamic_margin_softplus`;
  normalized head-regression targets; honest `target_modules=["Wqkv","Wo","Wi"]`.
- **Phase 2 — data** (largest expected gain): regenerate triplets in `utils/build_dataset.py`
  with graded negatives (~40% within 1.05–1.5×, ~30% 1.5–5×, ~30% >5×); drop null-number
  records. Ablate easy-only vs hard-only vs graded.
- **Phase 3 — loss**: add CoSENT-style listwise ordering loss (pairs ranked by |log ratio|,
  temperature τ ∈ {0.03, 0.05, 0.1}) to `LOSS_REGISTRY`; ablate alone and combined with head.
- **Phase 4 — LoRA ablations**: r ∈ {8, 16, 32}; ± `tok_embeddings`; attention-only vs +FFN.
- **Phase 5 — tokenization**: number canonicalization at input (fixed format / scientific
  notation); measure on digit_bias + permutation_trap suites specifically.

Composing results: main table (models × triplet_acc / recall@k / ordering_mean /
ordering_mean_random / sim_spread, 3 seeds mean±std for headline rows); ablation tables per
phase — all generated from `load_registry(WORK_DIR)` with `df.to_latex()`, never hand-pasted.
Figures: similarity-vs-log-ratio curve for anchor 500 (base vs best); ordering ρ vs epoch from
the per-epoch checkpoints (shows the overtraining degradation seen in corrected_ML). Add one
general-purpose sentence embedder as an external baseline and position against xVal and
Wallace et al. (2019) numeracy probing.

## Evaluation code

`training/evaluate.py` replaces the ad-hoc eval notebooks: `load_eval_model` (base or
adapter), triplet metrics (seeded negative sampling — old versions were unseeded, so
recall@k wasn't reproducible), the 5 fixed probe suites (kept verbatim for comparability),
`make_random_suites()` for randomized robustness probes, and `ordering_scores()` computing
the Spearman-vs-log-distance metric used in the analysis above. `evaluate_adapter()` returns
one flat dict ready for `log_result`. The eval section at the end of `colab/train_run.ipynb`
runs it in the same session as training; `embed_eval_clean.ipynb` remains as reference.

## Migration status

`training/*.py` and `colab/train_run.ipynb` are new; none of the 7 existing trainer notebooks or
`colab/test_results/*.xlsx` were modified or deleted. Once `train_run.ipynb` has reproduced your
current best setup, the old notebooks can be moved into `colab/archive/` (`git mv`, preserving
history) — do that only once you've verified equivalence, since they're still the only record of
exactly how earlier checkpoints were produced. `colab/test_results/*.xlsx` is worth keeping as a
historical record but not worth migrating row-by-row: its columns can't be reliably attributed
back to a specific checkpoint.
