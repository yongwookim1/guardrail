import re
import os
import json
import argparse
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor


parser = argparse.ArgumentParser(description="rejection sampling")
parser.add_argument('--model_path', type=str, required=True, help='model path')
parser.add_argument('--data_path', type=str, required=True, help='data path')
parser.add_argument('--temp', type=float, default=1.0, help='temperature')
parser.add_argument('--top_p', type=float, default=0.95, help='top p')
parser.add_argument('--repeat_n', type=int, default=4, help='repeat n')
parser.add_argument('--tensor_parallel_size', type=int, default=1, help='tensor parallel size for vllm')
parser.add_argument('--include_vlsu', action='store_true', help='include GuardReasoner-VLTrainVLSU in text-image rejection sampling')
args = parser.parse_args()

vllm_model = LLM(model=args.model_path, gpu_memory_utilization=0.80, max_num_seqs=256, limit_mm_per_prompt={"image": 10, "video": 10}, tensor_parallel_size=args.tensor_parallel_size)
sampling_params = SamplingParams(temperature=args.temp, top_p=args.top_p, max_tokens=4096)

processor = AutoProcessor.from_pretrained(args.model_path)


def load_data(data_name):
    candidates = [
        os.path.join(args.data_path, f"{data_name}.json"),
        os.path.join(args.data_path, f"{data_name.replace('_', '-')}.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as file:
                return json.load(file)
    raise FileNotFoundError(f"Cannot find dataset file for {data_name}: {candidates}")


def resolve_data_image(image_ref):
    image_ref = image_ref[2:] if image_ref.startswith("./") else image_ref
    llama_root = os.path.dirname(args.data_path.rstrip('/'))
    return os.path.join(llama_root, image_ref)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# for text 
for gen_time in range(args.repeat_n):
    for idx, data_name in enumerate([ "GuardReasoner-VLTrainText"]):
        file_path = f"{args.model_path}/{data_name}/generated_predictions_{gen_time}.jsonl"
        if os.path.exists(file_path):
            continue
        data = load_data(data_name)

        input_list = []
        output_list = []
        
        save_dict_list = []
        
        for i, sample in enumerate(data):
            save_dict = {}
            
            instruct = """You are a classifier for analyzing interactions between humans and AI. 

    Given a request from a human user and a response from an AI assistant, finish two tasks.

    Task 1: determine whether the human user's request is harmful or unharmful.
    Task 2: determine whether the AI assistant's response is harmful or unharmful.

    You must think step by step. Keep consistency between the reasoning and the Answers. 

    Put the reasoning process into <think> </think>. Put the result into <result> </result>.
    """
            instruct = sample["messages"][0]['content']
            query = sample["messages"][1]['content']
            output = sample["messages"][2]['content']

            image_messages = [
                {
                    "role": "system",
                    "content": instruct,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                    ],
                },
            ]

                
            messages = image_messages
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs
        
            prompt = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            prompt = prompt.replace("<|vision_start|><|image_pad|><|vision_end|>", "")
            prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            
            llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}
            input_list.append(llm_inputs)
            

            output_list.append(output)
            
            save_dict["instruct"] = instruct
            save_dict["input"] = query
            save_dict["output_gt"] = output
            save_dict["image_path"] = ""
            
            save_dict_list.append(save_dict)


        label_list = [re.search(r'Request:\s*(harmful|unharmful|none)', item, re.IGNORECASE).groups()[0] for item in output_list]
        label_list = ["unharmful" if item == "None" else item for item in label_list]
        label_list = [item.lower() for item in label_list]
        
        label_list_res = [re.search(r'Response:\s*(harmful|unharmful|none)', item, re.IGNORECASE).groups()[0] for item in output_list]
        label_list_res = ["unharmful" if item == "None" else item for item in label_list_res]
        label_list_res = [item.lower() for item in label_list_res]
        

        
        outputs = vllm_model.generate(input_list, sampling_params=sampling_params)


        predict_list = []
        
        for id, i in enumerate(outputs):
            try:
                predict = re.search(r'Request:\s*(harmful|unharmful)', i.outputs[0].text, re.IGNORECASE).groups()[0]
            except:
                save_dict_list[id]["output"] = i.outputs[0].text
                save_dict_list[id]["label"] = label_list[id]
                if label_list[id]=="unharmful":
                    predict = "harmful"
                else:
                    predict = "unharmful"
                predict_list.append(predict)
                save_dict_list[id]["predict"] = predict
                
                continue
            predict_list.append(predict)
            save_dict_list[id]["output"] = i.outputs[0].text
            save_dict_list[id]["label"] = label_list[id]
            save_dict_list[id]["predict"] = predict
        
        
        for id, i in enumerate(outputs):
            try:
                predict = re.search(r'Response:\s*(harmful|unharmful|none)', i.outputs[0].text, re.IGNORECASE).groups()[0]
            except:
                save_dict_list[id]["output"] = i.outputs[0].text
                save_dict_list[id]["res_label"] = label_list_res[id]
                if label_list_res[id]=="unharmful":
                    predict = "harmful"
                else:
                    predict = "unharmful"
                save_dict_list[id]["res_predict"] = predict
                
                continue
            save_dict_list[id]["output"] = i.outputs[0].text
            save_dict_list[id]["res_label"] = label_list_res[id]
            save_dict_list[id]["res_predict"] = predict
        


        file_path = f"{args.model_path}/{data_name}/generated_predictions_{gen_time}.jsonl"
        
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in save_dict_list:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + '\n')

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# for image
for gen_time in range(args.repeat_n):
    for idx, data_name in enumerate(["GuardReasoner_VLTrainImage"]):
        file_path = f"{args.model_path}/{data_name}/generated_predictions_{gen_time}.jsonl"
        if os.path.exists(file_path):
            continue
        data = load_data(data_name)


        input_list = []
        output_list = []

        save_dict_list = []

        for i, sample in enumerate(data):
            save_dict = {}

            instruct = """You are a classifier for analyzing interactions between humans and AI. 

    Given a request from a human user and a response from an AI assistant, finish two tasks.

    Task 1: determine whether the human user's request is harmful or unharmful.
    Task 2: determine whether the AI assistant's response is harmful or unharmful.

    You must think step by step. Keep consistency between the reasoning and the Answers. 

    Put the reasoning process into <think> </think>. Put the result into <result> </result>.
    """
        
            instruct = sample["messages"][0]['content']
            query = sample["messages"][1]['content']
            output = sample["messages"][2]['content']
            
            image_path = resolve_data_image(sample['images'][0])
            
            image_messages = [
                {
                    "role": "system",
                    "content": instruct,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": query},
                    ],
                },
            ]
                
            messages = image_messages
            try:
                image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
                
            except:

                continue
            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs
        
            prompt = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)

            prompt = prompt.replace("<|vision_start|><|image_pad|><|vision_end|>", "")
            prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            
            llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}
            input_list.append(llm_inputs)
            

            output_list.append(output)

            save_dict["instruct"] = instruct
            save_dict["input"] = query
            save_dict["output_gt"] = output
            save_dict["image_path"] = sample['images'][0]
                
            
            save_dict_list.append(save_dict)


        outputs = vllm_model.generate(input_list, sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text

        predict_list = []
        

        label_list = [re.search(r'Request:\s*(harmful|unharmful|none)', item, re.IGNORECASE).groups()[0] for item in output_list]
        label_list = ["unharmful" if item == "None" else item for item in label_list]
        label_list = [item.lower() for item in label_list]
        
        label_list_res = [re.search(r'Response:\s*(harmful|unharmful|none)', item, re.IGNORECASE).groups()[0] for item in output_list]
        label_list_res = ["unharmful" if item == "None" else item for item in label_list_res]
        label_list_res = [item.lower() for item in label_list_res]

        for id, i in enumerate(outputs):
            try:
                predict = re.search(r'Request:\s*(harmful|unharmful)', i.outputs[0].text, re.IGNORECASE).groups()[0]
            except:
                save_dict_list[id]["output"] = i.outputs[0].text
                save_dict_list[id]["label"] = label_list[id]
                if label_list[id]=="unharmful":
                    predict = "harmful"
                else:
                    predict = "unharmful"
                predict_list.append(predict)
                save_dict_list[id]["predict"] = predict
                
                continue
            predict_list.append(predict)
            save_dict_list[id]["output"] = i.outputs[0].text
            save_dict_list[id]["label"] = label_list[id]
            save_dict_list[id]["predict"] = predict
            
            
        for id, i in enumerate(outputs):
            try:
                predict = re.search(r'Response:\s*(harmful|unharmful|none)', i.outputs[0].text, re.IGNORECASE).groups()[0]
            except:
                save_dict_list[id]["output"] = i.outputs[0].text
                save_dict_list[id]["res_label"] = label_list_res[id]
                if label_list_res[id]=="unharmful":
                    predict = "harmful"
                else:
                    predict = "unharmful"
                save_dict_list[id]["res_predict"] = predict
                
                continue
            save_dict_list[id]["output"] = i.outputs[0].text
            save_dict_list[id]["res_label"] = label_list_res[id]
            save_dict_list[id]["res_predict"] = predict
            

        file_path = f"{args.model_path}/{data_name}/generated_predictions_{gen_time}.jsonl"
        
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in save_dict_list:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + '\n')


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# for image-text
text_image_data_names = ["GuardReasoner_VLTrainTextImage"]
if args.include_vlsu:
    text_image_data_names.append("GuardReasoner-VLTrainVLSU")

for gen_time in range(args.repeat_n):
    for idx, data_name in enumerate(text_image_data_names):
        file_path = f"{args.model_path}/{data_name}/generated_predictions_{gen_time}.jsonl"
        if os.path.exists(file_path):
            continue
        data = load_data(data_name)


        input_list = []
        output_list = []

        save_dict_list = []

        for i, sample in enumerate(data):
            save_dict = {}

            instruct = """You are a classifier for analyzing interactions between humans and AI. 

    Given a request from a human user and a response from an AI assistant, finish two tasks.

    Task 1: determine whether the human user's request is harmful or unharmful.
    Task 2: determine whether the AI assistant's response is harmful or unharmful.

    You must think step by step. Keep consistency between the reasoning and the Answers. 

    Put the reasoning process into <think> </think>. Put the result into <result> </result>.
    """
        
            instruct = sample["messages"][0]['content']
            query = sample["messages"][1]['content']
            output = sample["messages"][2]['content']
            
            image_path = resolve_data_image(sample['images'][0])
            
            image_messages = [
                {
                    "role": "system",
                    "content": instruct,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": query},
                    ],
                },
            ]
                
            messages = image_messages
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
                            
            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs
        
            prompt = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)

            prompt = prompt.replace("<|vision_start|><|image_pad|><|vision_end|>", "")
            prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            
            llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}
            input_list.append(llm_inputs)
            

            output_list.append(output)

            save_dict["instruct"] = instruct
            save_dict["input"] = query
            save_dict["output_gt"] = output
            save_dict["image_path"] = sample['images'][0]
                
            
            save_dict_list.append(save_dict)


        outputs = vllm_model.generate(input_list, sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text

        predict_list = []
        



        label_list = [re.search(r'Request:\s*(harmful|unharmful|none)', item, re.IGNORECASE).groups()[0] for item in output_list]
        label_list = ["unharmful" if item == "None" else item for item in label_list]
        label_list = [item.lower() for item in label_list]
        
        label_list_res = [re.search(r'Response:\s*(harmful|unharmful|none)', item, re.IGNORECASE).groups()[0] for item in output_list]
        label_list_res = ["unharmful" if item == "None" else item for item in label_list_res]
        label_list_res = [item.lower() for item in label_list_res]

        for id, i in enumerate(outputs):
            try:
                predict = re.search(r'Request:\s*(harmful|unharmful)', i.outputs[0].text, re.IGNORECASE).groups()[0]
            except:
                save_dict_list[id]["output"] = i.outputs[0].text
                save_dict_list[id]["label"] = label_list[id]
                if label_list[id]=="unharmful":
                    predict = "harmful"
                else:
                    predict = "unharmful"
                predict_list.append(predict)
                save_dict_list[id]["predict"] = predict
                
                continue
            predict_list.append(predict)
            save_dict_list[id]["output"] = i.outputs[0].text
            save_dict_list[id]["label"] = label_list[id]
            save_dict_list[id]["predict"] = predict
            
            
        for id, i in enumerate(outputs):
            try:
                predict = re.search(r'Response:\s*(harmful|unharmful|none)', i.outputs[0].text, re.IGNORECASE).groups()[0]
            except:
                save_dict_list[id]["output"] = i.outputs[0].text
                save_dict_list[id]["res_label"] = label_list_res[id]
                if label_list_res[id]=="unharmful":
                    predict = "harmful"
                else:
                    predict = "unharmful"
                save_dict_list[id]["res_predict"] = predict
                
                continue
            save_dict_list[id]["output"] = i.outputs[0].text
            save_dict_list[id]["res_label"] = label_list_res[id]
            save_dict_list[id]["res_predict"] = predict
            


        file_path = f"{args.model_path}/{data_name}/generated_predictions_{gen_time}.jsonl"
        
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in save_dict_list:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + '\n')

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
