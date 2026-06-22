import torch
import pandas as pd
from tqdm import tqdm
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import os

BASE_MODEL  = "facebook/nllb-200-distilled-1.3B"
CKPT        = r"as_mni_finetune\as_finetune\checkpoints\en_indic\checkpoint-1750"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SRC_LANG    = "eng_Latn"
TGT_LANG    = "mni_Mtei"
INPUT_FILE  = r"English - Meitei\en-mni-Mtei Test.xlsx"
OUTPUT_FILE = "results/RozarNLP-cr7_primary_en_to_mni.txt"
BATCH_SIZE  = 16

# ── Tokenizer ─────────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = NllbTokenizer.from_pretrained(CKPT, use_fast=False)
tokenizer.src_lang = SRC_LANG


# ── Model ─────────────────────────────────────────────────────────────────────
print("Loading model...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)

base_model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(base_model, CKPT)
model = model.to(DEVICE)
model.eval()
print(f"Model ready on {DEVICE}")

# ── Validate BOS token ────────────────────────────────────────────────────────
bos_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
assert bos_id != tokenizer.unk_token_id, f"'{TGT_LANG}' resolved to <unk>!"
print(f"Target lang '{TGT_LANG}' → token id {bos_id} ✓")

# ── Load input ────────────────────────────────────────────────────────────────
df = pd.read_excel(INPUT_FILE)
sentences = df["English Sentences"].astype(str).tolist()
print(f"Translating {len(sentences)} sentences...")

# ── Inference ─────────────────────────────────────────────────────────────────
LANG_PREFIXES = ["mni_Mtei", "eng_Latn"]

def strip_lang_prefix(text: str) -> str:
    for prefix in LANG_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return text.strip()

translations = []

for i in tqdm(range(0, len(sentences), BATCH_SIZE)):
    batch = sentences[i:i + BATCH_SIZE]

    tokenizer.src_lang = SRC_LANG
    inputs = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(DEVICE)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=bos_id,
            max_new_tokens=256,
            num_beams=3,
        )

    outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    outputs = [strip_lang_prefix(o) for o in outputs]
    translations.extend(outputs)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for line in translations:
        f.write(line.strip() + "\n")

print(f"Saved {len(translations)} translations → {OUTPUT_FILE}")