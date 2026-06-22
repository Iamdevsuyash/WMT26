"""
train.py
--------
Fine-tunes facebook/nllb-200-distilled-1.3B with a single multilingual LoRA
covering Khasi and Nyishi (both directions) in one checkpoint each.

Usage:
    python train.py --direction en_indic   # EN → Khasi/Nyishi
    python train.py --direction indic_en   # Khasi/Nyishi → EN
    python train.py --direction both       # run both sequentially

Before running:
    1. Run prepare_data.py to generate data/processed/combined_*.jsonl
    2. pip install -r requirements.txt

Why NLLB-200 over IndicTrans2:
  - Standard HuggingFace seq2seq — no custom tokenizer class, no overridden
    set_input_embeddings(), no ONNX import hacks needed.
  - kha_Latn (Khasi) is already in NLLB-200's vocab.
  - nif_Latn (Nyishi) is not in vocab but add_tokens() works cleanly because
    NLLB uses a standard SentencePiece tokenizer backed by NllbTokenizer —
    resize_token_embeddings() works out of the box.
  - Forced BOS token approach: NLLB uses forced_bos_token_id at inference
    time to select the target language. We set this in generation_config.
"""

import argparse
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from dataset import Seq2SeqCollator, TranslationDataset

# ── GPU detection ─────────────────────────────────────────────────────────────
_bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
_DTYPE   = torch.bfloat16 if _bf16_ok else torch.float16

print(f"[GPU] CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[GPU] Device         : {torch.cuda.get_device_name(0)}")
    print(f"[GPU] VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"[GPU] Using dtype    : {_DTYPE}")

# ── Configuration ─────────────────────────────────────────────────────────────

# 1.3B distilled is the sweet spot for an 8 GB VRAM card with LoRA.
# Swap to "facebook/nllb-200-1.3B" (non-distilled) if you want to try the
# denser model — same interface, ~same VRAM at 4-bit/LoRA config below.
MODEL_ID = "facebook/nllb-200-distilled-1.3B"

DATA = {
    "train": "data/processed/combined_train.jsonl",
    "val":   "data/processed/combined_val.jsonl",
}

OUTPUT_BASE = Path("checkpoints")

# ── Language codes ─────────────────────────────────────────────────────────────
# NLLB-200 uses FLORES-200 language codes (ISO 639-3 + script tag).
#
#   eng_Latn  — English   (already in NLLB vocab)
#   kha_Latn  — Khasi     (already in NLLB vocab — verified in flores200 list)
#   nif_Latn  — Nyishi    (NOT in NLLB vocab — we add it below)
#
# At inference time NLLB selects the output language via forced_bos_token_id,
# so the lang code token must be in the tokenizer vocab.
# ──────────────────────────────────────────────────────────────────────────────
LANG_CODES = {
    "english": "eng_Latn",
    "khasi":   "kha_Latn",
    "nyishi":  "nif_Latn",
}

VALID_LANGS: set[str] = set(LANG_CODES.values())

# ── LoRA config ────────────────────────────────────────────────────────────────
# NLLB-200 is built on top of the mBART architecture (transformer enc-dec).
# Attention projection names: q_proj, k_proj, v_proj, out_proj — same as
# before. These names are stable across NLLB checkpoints.
LORA_CFG = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    bias="none",
)

# ── Training hyperparameters ───────────────────────────────────────────────────
TRAIN_KWARGS = dict(
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,        # effective batch = 32
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.06,
    bf16=_bf16_ok,
    fp16=not _bf16_ok,
    tf32=True,
    save_strategy="steps",
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    predict_with_generate=False,
    logging_steps=25,
    # Keep at 0: forked workers won't see tokenizer state for newly added tokens.
    # NLLB uses Python-side SentencePiece — same caveat as IT2 here.
    dataloader_num_workers=0,
    dataloader_pin_memory=True,
    report_to="none",
    save_total_limit=2,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

MAX_SRC_LEN = 256
MAX_TGT_LEN = 256


# ── Helpers ────────────────────────────────────────────────────────────────────

def extend_tokenizer_for_new_langs(tokenizer, lang_codes: set[str]) -> list[str]:
    """
    Add any lang-code tokens missing from the NLLB tokenizer.

    NLLB stores lang tokens as regular special tokens (e.g. "kha_Latn").
    Unlike IndicTrans2, NllbTokenizer exposes a proper get_vocab() so we can
    check membership cleanly. add_tokens(..., special_tokens=True) then
    registers them so they're never split by SentencePiece.

    Returns the list of tokens actually added.
    """
    vocab = tokenizer.get_vocab()
    to_add = [code for code in sorted(lang_codes) if code not in vocab]

    if to_add:
        tokenizer.add_tokens(to_add, special_tokens=True)
        print(f"  [tokenizer] Added {len(to_add)} new lang token(s): {to_add}")
    else:
        print("  [tokenizer] All lang codes already in vocab — no changes needed.")

    return to_add


def resize_model_embeddings(model, tokenizer) -> None:
    """
    Resize the model's token embeddings to match the (possibly extended)
    tokenizer vocabulary.

    Unlike IndicTrans2, NLLB's resize_token_embeddings() works out of the
    box — it's the standard HuggingFace implementation backed by a shared
    embedding matrix. New rows are mean-initialised internally by HF.
    """
    old_size = model.config.vocab_size
    new_size = len(tokenizer)
    if new_size != old_size:
        model.resize_token_embeddings(new_size)
        print(f"  [model] Embeddings resized: {old_size} → {new_size} tokens")
    else:
        print(f"  [model] Embedding size unchanged ({old_size})")


def set_src_lang(tokenizer, direction: str) -> None:
    """
    Set tokenizer.src_lang for the training direction.

    NllbTokenizer prepends the source-language token automatically when
    src_lang is set. For a multilingual dataset with mixed source langs,
    the collator / dataset should override this per-batch; here we set a
    sensible default for the direction.
    """
    if direction == "en_indic":
        tokenizer.src_lang = LANG_CODES["english"]
    else:
        # indic_en: src is Khasi or Nyishi — default to Khasi; per-sample
        # override is handled in TranslationDataset / Seq2SeqCollator.
        tokenizer.src_lang = LANG_CODES["khasi"]


def get_forced_bos_token_id(tokenizer, direction: str) -> int | None:
    """
    Return the forced_bos_token_id for generation.

    NLLB requires forced_bos_token_id at inference time to pick the target
    language. For training with teacher forcing this doesn't matter, but we
    save it in generation_config so the checkpoint is ready for inference.

    For a multilingual target (Khasi + Nyishi in the same model), forced_bos
    must be overridden per sample at inference — return None here to avoid
    locking the checkpoint to one target language.
    """
    return None   # caller sets per-sample at inference time


def count_trainable(model) -> str:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct       = 100 * trainable / total
    return f"{trainable:,} / {total:,}  ({pct:.2f} %)"


def inspect_module_names(model) -> None:
    """Run once to verify LoRA target_modules match your checkpoint."""
    print("\nAttention-related module names:")
    for name, _ in model.named_modules():
        if any(k in name for k in ["proj", "attn", "self", "query", "key", "value"]):
            print(f"  {name}")


# ── Main training loop ─────────────────────────────────────────────────────────

def train_direction(direction: str) -> None:
    out_dir = OUTPUT_BASE / direction
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Direction : {direction}")
    print(f"  Base model: {MODEL_ID}")
    print(f"  Output    : {out_dir}")
    print(f"{'='*60}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # kha_Latn is in NLLB vocab; nif_Latn (Nyishi) is not — add it.
    added_tokens = extend_tokenizer_for_new_langs(tokenizer, VALID_LANGS)

    # Set default src_lang (NllbTokenizer uses this to prepend the lang token).
    set_src_lang(tokenizer, direction)

    # ── Model ──────────────────────────────────────────────────────────────────
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        torch_dtype=_DTYPE,
        device_map="cuda",
    )

    # Standard HF resize — works cleanly with NLLB (no NotImplementedError).
    resize_model_embeddings(model, tokenizer)

    # Uncomment to verify LoRA target_modules names before training:
    # inspect_module_names(model)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LORA_CFG)

    print(f"\n  Trainable params: {count_trainable(model)}")
    model.print_trainable_parameters()

    # ── Datasets ───────────────────────────────────────────────────────────────
    print("\nLoading datasets …")
    train_ds = TranslationDataset(
        DATA["train"], tokenizer, MAX_SRC_LEN, MAX_TGT_LEN, valid_langs=VALID_LANGS
    )
    val_ds = TranslationDataset(
        DATA["val"], tokenizer, MAX_SRC_LEN, MAX_TGT_LEN, valid_langs=VALID_LANGS
    )
    collator = Seq2SeqCollator(tokenizer)

    # ── Trainer ────────────────────────────────────────────────────────────────
    args = Seq2SeqTrainingArguments(output_dir=str(out_dir), **TRAIN_KWARGS)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nStarting training …")
    trainer.train(
    resume_from_checkpoint=True
)
    # ── Save adapter + extended tokenizer ─────────────────────────────────────
    final = out_dir / "final"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"\n  Saved to {final}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--direction",
        choices=["en_indic", "indic_en", "both"],
        default="both",
    )
    args = parser.parse_args()

    if args.direction == "both":
        train_direction("en_indic")
        train_direction("indic_en")
    else:
        train_direction(args.direction)


if __name__ == "__main__":
    main()