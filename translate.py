import os
import json
import argparse
from vllm import LLM, SamplingParams

MODEL_PATH = "../model_cache/HyperCLOVAX-SEED-Think-32B"

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
    "--dataset",
    type=str,
    default=None,
    help="Dataset name to translate (without .json). Translates all datasets if not specified.",
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
    default=64,
    help="Number of samples to process per batch",
)
args = parser.parse_args()

datasets = [args.dataset] if args.dataset else ALL_DATASETS

print(f"Loading model: {MODEL_PATH}")
vllm_model = LLM(
    model=MODEL_PATH,
    gpu_memory_utilization=0.80,
    max_num_seqs=128,
    trust_remote_code=True,
)
sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=4096)


def build_translation_prompt(text: str) -> str:
    return (
        f"<|im_start|>system\n{TRANSLATION_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


for data_name in datasets:
    src_path = os.path.join(args.benchmark_path, f"{data_name}.json")
    if not os.path.exists(src_path):
        print(f"[SKIP] {src_path} not found.")
        continue

    with open(src_path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"[{data_name}] Translating {len(data)} samples...")

    # Collect all inputs that need translation
    inputs_to_translate = []
    for sample in data:
        if "input" in sample:
            inputs_to_translate.append(sample["input"])
        else:
            inputs_to_translate.append("")

    # Build prompts
    prompts = [build_translation_prompt(text) for text in inputs_to_translate]

    # Generate translations in batches
    translated_texts = []
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start : start + args.batch_size]
        outputs = vllm_model.generate(batch, sampling_params)
        for out in outputs:
            translated_texts.append(out.outputs[0].text.strip())
        print(f"  [{data_name}] {min(start + args.batch_size, len(prompts))}/{len(prompts)} done")

    # Build translated dataset
    translated_data = []
    for sample, translated_input in zip(data, translated_texts):
        new_sample = dict(sample)
        if "input" in new_sample:
            new_sample["input"] = translated_input
            new_sample["input_en"] = sample["input"]
        translated_data.append(new_sample)

    # Save
    dst_path = os.path.join(args.benchmark_path, f"{data_name}_ko.json")
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=4)

    print(f"[{data_name}] Saved to {dst_path}")

print("Translation complete.")
