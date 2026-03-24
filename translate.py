import os
import json
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

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
    "--max_new_tokens",
    type=int,
    default=4096,
    help="Maximum number of new tokens to generate",
)
args = parser.parse_args()

print(f"Loading model: {args.model_path}")
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
vllm_model = LLM(model=args.model_path, gpu_memory_utilization=0.70, max_num_seqs=256, enforce_eager=True)
sampling_params = SamplingParams(temperature=0., top_p=1.0, max_tokens=args.max_new_tokens)


def build_prompts(texts: list[str]) -> list[str]:
    prompts = []
    for text in texts:
        messages = [
            {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(prompt)
    return prompts


def translate_batch(texts: list[str]) -> list[str]:
    prompts = build_prompts(texts)
    outputs = vllm_model.generate(prompts, sampling_params=sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]


datasets = ALL_DATASETS

for data_name in datasets:
    src_path = os.path.join(args.benchmark_path, f"{data_name}.json")
    if not os.path.exists(src_path):
        print(f"[SKIP] {src_path} not found.")
        continue

    with open(src_path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"[{data_name}] Translating {len(data)} samples...")

    inputs_to_translate = [sample.get("input", "") for sample in data]
    translated_texts = translate_batch(inputs_to_translate)
    print(f"  [{data_name}] {len(inputs_to_translate)}/{len(inputs_to_translate)} done")

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

    translated_human = translate_batch(human_texts)
    print(f"  [SPA_VL_Eval human] {len(human_texts)}/{len(human_texts)} done")

    translated_ai = translate_batch(ai_texts)
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
