import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

BASE_MODEL = "facebook/nllb-200-distilled-1.3B"
CKPT = "checkpoints/en_indic/checkpoint-2500"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, CKPT)
model.eval()

tokenizer.src_lang = "eng_Latn"

sentence = "The weather is pleasant today."

inputs = tokenizer(
    sentence,
    return_tensors="pt"
).to(model.device)

generated = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.convert_tokens_to_ids("kha_Latn"),
    max_new_tokens=128
)

print(tokenizer.decode(
    generated[0],
    skip_special_tokens=True
))