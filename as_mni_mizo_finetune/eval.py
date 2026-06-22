import json
import torch
from tqdm import tqdm
from collections import defaultdict
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from sacrebleu.metrics import BLEU, CHRF

# ── Config ────────────────────────────────────────────────────────────────────
VAL_FILE   = "data/processed/combined_val.jsonl"
BASE_MODEL = "facebook/nllb-200-distilled-1.3B"
CKPT = "C:/Users/CSE_SDPL/Desktop/WMT26/as_mni_finetune/checkpoints/en_indic/checkpoint-5500"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SAMPLES = None   # set to an int to cap evaluation

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = NllbTokenizer.from_pretrained(CKPT, use_fast=False)

# ── Model ─────────────────────────────────────────────────────────────────────
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)
base_model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(base_model, CKPT)
model = model.to(DEVICE)
model.eval()

# ── Load val records ──────────────────────────────────────────────────────────
with open(VAL_FILE, encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

if MAX_SAMPLES is not None:
    records = records[:MAX_SAMPLES]

print(f"Evaluating on {len(records)} records  |  device: {DEVICE}\n")

# ── Inference ─────────────────────────────────────────────────────────────────
# Bucket results by (src_lang, tgt_lang) pair
pair_preds: dict[tuple, list[str]] = defaultdict(list)
pair_refs:  dict[tuple, list[str]] = defaultdict(list)
pair_samples: dict[tuple, list[dict]] = defaultdict(list)  # for printing examples

for rec in tqdm(records):
    src      = rec["src"]
    tgt      = rec["tgt"]
    src_lang = rec["src_lang"]
    tgt_lang = rec["tgt_lang"]

    tokenizer.src_lang = src_lang

    inputs = tokenizer(
        src,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(DEVICE)

    bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    assert bos_id != tokenizer.unk_token_id, (
        f"tgt_lang '{tgt_lang}' resolved to <unk>. "
        "Check that this token was saved into the checkpoint's tokenizer."
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=bos_id,
            max_new_tokens=256,
            num_beams=5,
        )

    pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    key = (src_lang, tgt_lang)
    pair_preds[key].append(pred)
    pair_refs[key].append(tgt)
    pair_samples[key].append({"src": src, "ref": tgt, "pred": pred})

# ── Per-language-pair metrics ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"{'Direction':<35} {'N':>5}  {'BLEU':>7}  {'chrF':>7}")
print("=" * 70)

all_preds: list[str] = []
all_refs:  list[str] = []

# Friendly display names for NLLB lang codes
DISPLAY = {
    "eng_Latn": "English",
    "kha_Latn": "Khasi",
    "nif_Latn": "Nyishi",
    "asm_Beng": "Assamese",
    "mni_Mtei": "Meitei",
    "lus_Latn": "Mizo",
    "mni_Beng" : "Manipuri",
}

def label(code: str) -> str:
    return DISPLAY.get(code, code)

for key in sorted(pair_preds.keys()):
    src_lang, tgt_lang = key
    preds = pair_preds[key]
    refs  = pair_refs[key]

    bleu = BLEU().corpus_score(preds, [refs])
    chrf = CHRF().corpus_score(preds, [refs])

    direction = f"{label(src_lang)} → {label(tgt_lang)}"
    print(f"{direction:<35} {len(preds):>5}  {bleu.score:>7.2f}  {chrf.score:>7.2f}")

    all_preds.extend(preds)
    all_refs.extend(refs)

# ── Overall aggregate ─────────────────────────────────────────────────────────
print("-" * 70)
overall_bleu = BLEU().corpus_score(all_preds, [all_refs])
overall_chrf = CHRF().corpus_score(all_preds, [all_refs])
print(f"{'OVERALL':<35} {len(all_preds):>5}  {overall_bleu.score:>7.2f}  {overall_chrf.score:>7.2f}")
print("=" * 70)

# ── Sample predictions per pair ───────────────────────────────────────────────
print("\n\nSample predictions per direction (up to 5 each):\n")
for key in sorted(pair_samples.keys()):
    src_lang, tgt_lang = key
    direction = f"{label(src_lang)} → {label(tgt_lang)}"
    print(f"\n{'─'*70}")
    print(f"  {direction}  ({len(pair_preds[key])} samples)")
    print(f"{'─'*70}")
    for item in pair_samples[key][:5]:
        print(f"  SRC : {item['src']}")
        print(f"  REF : {item['ref']}")
        print(f"  PRED: {item['pred']}")
        print()