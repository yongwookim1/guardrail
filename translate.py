import os
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "K-intelligence/Midm-2.0-Base-Instruct"

ALL_DATASETS = [
    "ToxicChat",
    "HarmBenchPrompt",
    "OpenAIModeration",
    "AegisSafetyTest",
    "SimpleSafetyTests",
    "WildGuardTestPrompt",
    "HarmBenchResponse",
    "SafeRLHF",
    "BeaverTails",
    "XSTestReponseHarmful",
    "WildGuardTestResponse",
]

TRANSLATION_SYSTEM_PROMPT = (
    "You are a professional translator. "
    "Translate the following English text to Korean. "
    "Output only the translated text without any explanation or additional commentary."
)

HUMAN_LABEL = "Human user:\n"
AI_LABEL    = "\n\nAI assistant:\n"

parser = argparse.ArgumentParser(description="Translate benchmark datasets to Korean")
parser.add_argument(
    "--model_path",
    type=str,
    default=MODEL_PATH,
    help="Path to translation model",
)
parser.add_argument(
    "--benchmark_path",
    type=str,
    default="./data/benchmark/",
    help="Path to benchmark directory",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=4,
    help="Number of samples to process per batch",
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=4096,
    help="Maximum number of new tokens to generate",
)
args = parser.parse_args()

print(f"Loading model: {args.model_path}")
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()


def translate_batch(texts: list[str]) -> list[str]:
    messages_batch = [
        [
            {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        for text in texts
    ]

    prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_batch
    ]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(model.device)

    inputs.pop("token_type_ids", None)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    for output in outputs:
        generated = output[input_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        results.append(text)
    return results


def parse_input(text: str) -> tuple[str, str | None]:
    """Split a formatted input into (human_text, ai_text). ai_text is None if 'None'."""
    human_text = text[len(HUMAN_LABEL):text.find(AI_LABEL)]
    ai_raw = text[text.find(AI_LABEL) + len(AI_LABEL):].rstrip("\n")
    ai_text = None if ai_raw == "None" else ai_raw
    return human_text, ai_text


def reconstruct_input(human_ko: str, ai_ko: str | None) -> str:
    """Rebuild the labeled format with translated content."""
    ai_part = ai_ko if ai_ko is not None else "None"
    return f"Human user:\n{human_ko}\n\nAI assistant:\n{ai_part}\n\n"


def translate_texts_in_batches(texts: list[str]) -> list[str]:
    results = []
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start:start + args.batch_size]
        results.extend(translate_batch(batch))
        print(f"  {min(start + args.batch_size, len(texts))}/{len(texts)} done")
    return results


for data_name in ALL_DATASETS:
    src_path = os.path.join(args.benchmark_path, f"{data_name}.json")
    if not os.path.exists(src_path):
        print(f"[SKIP] {src_path} not found.")
        continue

    with open(src_path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"[{data_name}] Translating {len(data)} samples...")

    # Phase 1: Parse
    parsed = [parse_input(sample["input"]) for sample in data]

    # Phase 2: Build flat list (only actual text, skip None AI parts)
    flat_texts = []
    index_map = []
    for human_text, ai_text in parsed:
        h_idx = len(flat_texts)
        flat_texts.append(human_text)
        if ai_text is not None:
            a_idx = len(flat_texts)
            flat_texts.append(ai_text)
        else:
            a_idx = None
        index_map.append((h_idx, a_idx))

    # Phase 3: Translate in batches
    translated = translate_texts_in_batches(flat_texts)
    print(f"  [{data_name}] {len(flat_texts)} segments translated")

    # Phase 4: Reconstruct
    translated_data = []
    for sample, (h_idx, a_idx) in zip(data, index_map):
        new_sample = dict(sample)
        if "input" in new_sample:
            new_sample["input_en"] = sample["input"]
            human_ko = translated[h_idx]
            ai_ko = translated[a_idx] if a_idx is not None else None
            new_sample["input"] = reconstruct_input(human_ko, ai_ko)
        translated_data.append(new_sample)

    dst_path = os.path.join(args.benchmark_path, f"{data_name}_ko.json")
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=4)

    print(f"[{data_name}] Saved to {dst_path}")


# SPA_VL_Eval: translate human text and AI response inside messages structure
spa_path = os.path.join(args.benchmark_path, "SPA_VL_Eval.json")
if os.path.exists(spa_path):
    with open(spa_path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"[SPA_VL_Eval] Translating {len(data)} samples...")

    def extract_spa_texts(sample):
        content = sample["messages"][0]["content"]
        human_start = content.find("<image>") + len("<image>")
        ai_marker = "\n\nAI assistant:\n"
        ai_start = content.find(ai_marker) + len(ai_marker)
        human_text = content[human_start:content.find(ai_marker)].strip()
        ai_text = content[ai_start:].strip()
        return human_text, ai_text

    human_texts = [extract_spa_texts(s)[0] for s in data]
    ai_texts = [extract_spa_texts(s)[1] for s in data]

    translated_human = translate_texts_in_batches(human_texts)
    print(f"  [SPA_VL_Eval human] {len(human_texts)}/{len(human_texts)} done")

    translated_ai = translate_texts_in_batches(ai_texts)
    print(f"  [SPA_VL_Eval ai] {len(ai_texts)}/{len(ai_texts)} done")

    translated_data = []
    for sample, h_ko, a_ko in zip(data, translated_human, translated_ai):
        new_sample = dict(sample)
        content = sample["messages"][0]["content"]
        human_start = content.find("<image>") + len("<image>")
        ai_marker = "\n\nAI assistant:\n"
        new_content = (
            content[:human_start] + " " + h_ko + "\n"
            + ai_marker + a_ko + "\n"
        )
        new_sample["messages"] = [{"content": new_content, "role": "user"}, sample["messages"][1]]
        translated_data.append(new_sample)

    dst_path = os.path.join(args.benchmark_path, "SPA_VL_Eval_ko.json")
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=4)

    print(f"[SPA_VL_Eval] Saved to {dst_path}")

print("Translation complete.")
