import os
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = os.path.abspath("../models_cache/HyperCLOVAX-SEED-Think-32B")

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
    "HarmImageTest",
    "SPA_VL_Eval",
]

TRANSLATION_SYSTEM_PROMPT = (
    "You are a professional translator. "
    "Translate the following English text to Korean accurately and naturally. "
    "Output only the translated text without any explanation or additional commentary."
)

parser = argparse.ArgumentParser(description="Translate benchmark datasets to Korean")
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

datasets = ALL_DATASETS

print(f"Loading model: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
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
        truncation=True,
        max_length=1024,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    results = []
    for output in outputs:
        generated = output[input_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        results.append(text)
    return results


for data_name in datasets:
    src_path = os.path.join(args.benchmark_path, f"{data_name}.json")
    if not os.path.exists(src_path):
        print(f"[SKIP] {src_path} not found.")
        continue

    with open(src_path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"[{data_name}] Translating {len(data)} samples...")

    inputs_to_translate = [sample.get("input", "") for sample in data]

    translated_texts = []
    for start in range(0, len(inputs_to_translate), args.batch_size):
        batch = inputs_to_translate[start : start + args.batch_size]
        translated_texts.extend(translate_batch(batch))
        print(f"  [{data_name}] {min(start + args.batch_size, len(inputs_to_translate))}/{len(inputs_to_translate)} done")

    translated_data = []
    for sample, translated_input in zip(data, translated_texts):
        new_sample = dict(sample)
        if "input" in new_sample:
            new_sample["input_en"] = sample["input"]
            new_sample["input"] = translated_input
        translated_data.append(new_sample)

    dst_path = os.path.join(args.benchmark_path, f"{data_name}_ko.json")
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=4)

    print(f"[{data_name}] Saved to {dst_path}")

print("Translation complete.")
