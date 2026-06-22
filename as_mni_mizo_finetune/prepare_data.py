import pandas as pd
import os

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset

# df = pd.read_csv("English-Meitei-Mayek Transining Data 2026.xlsx - eng-mtei-train.csv")

# os.makedirs(
#     "data/train/eng_Latn-mtei_Mtei",
#     exist_ok=True
# )

# with open(
#     "data/train/eng_Latn-mtei_Mtei/train.eng_Latn",
#     "w",
#     encoding="utf-8"
# ) as f:
#     for x in df["eng-mtei-eng-train"]:
#         f.write(str(x).strip() + "\n")

# with open(
#     "data/train/eng_Latn-mtei_Mtei/train.mtei_Mtei",
#     "w",
#     encoding="utf-8"
# ) as f:
#     for x in df["eng-mtei-mtei-train"]:
#         f.write(str(x).strip() + "\n")



from pathlib import Path

src = Path("data/train/eng_Latn-mtei_Mtei/train.eng_Latn")
tgt = Path("data/train/eng_Latn-mtei_Mtei/train.mtei_Mtei")

src_lines = src.read_text(encoding="utf8").splitlines()
tgt_lines = tgt.read_text(encoding="utf8").splitlines()

n_dev = int(0.02 * len(src_lines))  # 2%

dev_src = src_lines[-n_dev:]
dev_tgt = tgt_lines[-n_dev:]

Path("data/dev/eng_Latn-mtei_Mtei").mkdir(parents=True, exist_ok=True)

with open("data/dev/eng_Latn-mtei_Mtei/dev.eng_Latn","w",encoding="utf8") as f:
    f.write("\n".join(dev_src))

with open("data/dev/eng_Latn-mtei_Mtei/dev.mtei_Mtei","w",encoding="utf8") as f:
    f.write("\n".join(dev_tgt))


"""
dataset.py
----------
PyTorch Dataset + DataCollator for NLLB-200 seq2seq fine-tuning.
Reads the JSONL files produced by prepare_data.py.

Languages in this run
---------------------
  eng_Latn  — English           (natively in NLLB vocab)
  asm_Beng  — Assamese          (natively in NLLB vocab)
  mni_Beng  — Meitei/Manipuri   (natively in NLLB vocab)
  lus_Latn  — Lushai/Mizo       (added via add_tokens() in train.py)

Root cause of the NoneType crash (and how we fix it)
-----------------------------------------------------
NllbTokenizerFast._switch_to_target_mode() looks up `self.tgt_lang` in an
internal dict (lang_code_to_id) that is populated at load time from the
original vocab.  Tokens added later via add_tokens() — like lus_Latn — are
NOT inserted into that dict, so any newly added lang code returns None,
causing int(None) → TypeError.

Two-part fix applied here:
  1. use_fast=False in train.py → loads NllbTokenizer (slow, Python/SP).
     The slow tokenizer does not have _switch_to_target_mode; it encodes
     target text by temporarily re-setting src_lang.
  2. Encode targets by setting tokenizer.src_lang = tgt_lang, calling the
     tokenizer normally, then restoring src_lang.  Never use text_target=
     with newly added lang codes — it internally calls _switch_to_target_mode
     even on the slow tokenizer in some HF versions.
"""




class TranslationDataset(Dataset):
    """
    Reads a JSONL file where each record has the shape:
        {"src": "...", "tgt": "...", "src_lang": "eng_Latn", "tgt_lang": "asm_Beng"}

    Parameters
    ----------
    jsonl_path   : path to the .jsonl file
    tokenizer    : NllbTokenizer slow (use_fast=False), already extended
    max_src_len  : max source token length (truncation)
    max_tgt_len  : max target token length (truncation)
    valid_langs  : if provided, records with unknown lang codes are skipped
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        max_src_len: int = 256,
        max_tgt_len: int = 256,
        valid_langs: Optional[set] = None,
    ):
        self.tokenizer   = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.records: list[dict] = []

        skipped = 0
        with Path(jsonl_path).open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  WARNING [{jsonl_path}:{lineno}] bad JSON, skipping: {e}")
                    skipped += 1
                    continue

                if valid_langs is not None:
                    bad = {rec.get("src_lang"), rec.get("tgt_lang")} - valid_langs
                    if bad:
                        print(
                            f"  WARNING [{jsonl_path}:{lineno}] "
                            f"unknown lang code(s) {bad}, skipping."
                        )
                        skipped += 1
                        continue

                self.records.append(rec)

        print(
            f"Loaded {len(self.records):,} records from {jsonl_path}"
            + (f"  ({skipped} skipped)" if skipped else "")
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        # ── Source tokenisation ───────────────────────────────────────────────
        # Set src_lang so the tokenizer prepends the correct lang-id token.
        # Safe because dataloader_num_workers=0 (single process, no races).
        self.tokenizer.src_lang = rec["src_lang"]

        model_inputs = self.tokenizer(
            rec["src"],
            max_length=self.max_src_len,
            truncation=True,
            padding=False,
        )

        # ── Target tokenisation ───────────────────────────────────────────────
        # We do NOT use text_target= here.
        #
        # Why: text_target= internally calls _switch_to_target_mode() on the
        # fast tokenizer (and some slow versions), which does a lang_code_to_id
        # lookup.  Tokens added via add_tokens() — e.g. lus_Latn — are absent
        # from that dict, so they return None → int(None) → TypeError.
        #
        # Safe alternative: temporarily set src_lang to the target lang code,
        # tokenize the target text normally (the tokenizer prepends the target
        # lang token just as it would for a source sentence), then restore the
        # original src_lang.  The resulting input_ids are identical to what
        # text_target= would produce for natively supported codes.
        original_src_lang = self.tokenizer.src_lang
        self.tokenizer.src_lang = rec["tgt_lang"]

        label_enc = self.tokenizer(
            rec["tgt"],
            max_length=self.max_tgt_len,
            truncation=True,
            padding=False,
        )

        self.tokenizer.src_lang = original_src_lang   # restore

        model_inputs["labels"] = label_enc["input_ids"]
        return model_inputs
        # Note: src_lang / tgt_lang strings are NOT stored in model_inputs.
        # Trainer expects only tensor-compatible values; strings cause a
        # collation error. If you need them for custom eval metrics, store
        # them in a parallel list on the dataset object instead.


@dataclass
class Seq2SeqCollator:
    """
    Pads inputs and labels to the longest sequence in the batch.
    Replaces padding positions in labels with -100 so cross-entropy
    ignores them.
    """
    tokenizer: object
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        input_ids      = [f["input_ids"]      for f in features]
        attention_mask = [f["attention_mask"]  for f in features]
        labels         = [f["labels"]          for f in features]

        # ── Pad encoder inputs ────────────────────────────────────────────────
        padded_inputs = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # ── Pad labels with -100 (ignored by cross-entropy loss) ──────────────
        max_label_len = max(len(l) for l in labels)
        if self.pad_to_multiple_of:
            rem = max_label_len % self.pad_to_multiple_of
            if rem:
                max_label_len += self.pad_to_multiple_of - rem

        padded_labels = [
            lbl + [-100] * (max_label_len - len(lbl))
            for lbl in labels
        ]

        padded_inputs["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return padded_inputs