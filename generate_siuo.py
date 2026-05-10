import os
import re
import json
import argparse
from PIL import Image
from template import INSTRUCTION
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="yueliu1999/GuardReasoner-VL-3B")
parser.add_argument("--siuo_dir", type=str, default="./data/SIUO/")
args = parser.parse_args()

SAVE_PATH = f"./data/test/{args.model_path.split('/')[-1]}/SIUO"

vllm_model = LLM(
    model=args.model_path,
    gpu_memory_utilization=0.70,
    max_num_seqs=256,
    limit_mm_per_prompt={"image": 10, "video": 10},
    enforce_eager=True,
    max_model_len=32768,
)
sampling_params = SamplingParams(temperature=0., top_p=1.0, max_tokens=4096)
processor = AutoProcessor.from_pretrained(args.model_path)

with open(os.path.join(args.siuo_dir, "siuo_gen.json")) as f:
    data = json.load(f)

input_list = []
label_list = []
save_dict_list = []

for sample in data:
    image_path = os.path.join(args.siuo_dir, "images", sample["image"])
    image = Image.open(image_path).convert("RGB")

    messages = [
        {"role": "system", "content": INSTRUCTION},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sample["question"]},
            ],
        },
    ]

    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = prompt.replace("<|vision_start|><|image_pad|><|vision_end|>", "")
    prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")

    input_list.append({"prompt": prompt, "multi_modal_data": mm_data})
    label_list.append("harmful")  # all SIUO samples are harmful by design
    save_dict_list.append({
        "question_id": sample["question_id"],
        "text_input": sample["question"],
        "image": sample["image"],
        "category": sample["category"],
    })

outputs = vllm_model.generate(input_list, sampling_params=sampling_params)

for id, output in enumerate(outputs):
    text = output.outputs[0].text
    try:
        predict = re.search(r'Request:\s*(harmful|unharmful)', text, re.IGNORECASE).groups()[0].lower()
    except:
        predict = "unharmful"

    save_dict_list[id]["text_output"] = text
    save_dict_list[id]["label"] = label_list[id]
    save_dict_list[id]["predict"] = predict
    save_dict_list[id]["res_len"] = len(text)

os.makedirs(SAVE_PATH, exist_ok=True)
with open(f"{SAVE_PATH}/generated_predictions.jsonl", "w", encoding="utf-8") as f:
    for item in save_dict_list:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Saved {len(save_dict_list)} samples to {SAVE_PATH}/generated_predictions.jsonl")
