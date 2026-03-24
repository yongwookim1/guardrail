import os
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

MODEL_PATH = "Qwen/Qwen3-8B"

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
    "Translate the following English text to Korean accurately and naturally. "
    "Output only the translated text without any explanation or additional commentary."
)

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

datasets = ALL_DATASETS

print(f"Loading model: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
generation_config = GenerationConfig.from_pretrained(args.model_path)
generation_config.max_new_tokens = args.max_new_tokens
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

    inputs.pop("token_type_ids", None)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
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

    translated_human = []
    for start in range(0, len(human_texts), args.batch_size):
        batch = human_texts[start:start + args.batch_size]
        translated_human.extend(translate_batch(batch))
        print(f"  [SPA_VL_Eval human] {min(start + args.batch_size, len(human_texts))}/{len(human_texts)} done")

    translated_ai = []
    for start in range(0, len(ai_texts), args.batch_size):
        batch = ai_texts[start:start + args.batch_size]
        translated_ai.extend(translate_batch(batch))
        print(f"  [SPA_VL_Eval ai] {min(start + args.batch_size, len(ai_texts))}/{len(ai_texts)} done")

    translated_data = []
    for sample, h_ko, a_ko in zip(data, translated_human, translated_ai):
        new_sample = dict(sample)
        content = sample["messages"][0]["content"]
        human_start = content.find("<image>") + len("<image>")
        ai_marker = "\n\nAI assistant:\n"
        ai_start = content.find(ai_marker) + len(ai_marker)
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
