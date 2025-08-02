import json
from transformers import PreTrainedTokenizerBase, AutoConfig
from typing import List, Dict
from llm_eval.datasets.sharegpt import SharegptDataset
from llm_eval.datasets.longbench.longbench import LongbenchDataset
from llm_eval.utils.tokenizer import get_tokenizer
from llm_eval.utils.logger import init_global_log

init_global_log()

# Statistic by 'statistic_maas_in_out_map.py', 2024/11/15
IN_OUT_MAP = [
    # {"in_min": 5500, "in_max": 5700, "out_min": 0, "out_max": 1800, "num": 10000000, "count": 0},
    {"in_min": 5300, "in_max": 5500, "out_min": 0, "out_max": 1800, "num": 10000000, "count": 0},
    # {"in_min": 500, "in_max": 1000, "out_min": 0, "out_max": 250, "num": 180, "count": 0},
    # {"in_min": 1000, "in_max": 1500, "out_min": 0, "out_max": 250, "num": 27, "count": 0},
    # {"in_min": 1500, "in_max": 2000, "out_min": 0, "out_max": 250, "num": 11, "count": 0},
    # {"in_min": 2000, "in_max": 4000, "out_min": 0, "out_max": 2500, "num": 46, "count": 0},
]


def sample_sharegpt_requests(
    sharegpt_dataset: SharegptDataset,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Dict]:
    dataset = sharegpt_dataset.dataset
    print(dataset)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter sequences.
    filtered_dataset: List[Dict] = []    
    
    count = 0
    total_num = sum(item["num"] for item in IN_OUT_MAP)
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        
        for in_out_map in IN_OUT_MAP:
            if in_out_map["count"] < in_out_map["num"]:
                if (in_out_map["in_min"] < prompt_len <= in_out_map["in_max"]) and (in_out_map["out_min"] < output_len <= in_out_map["out_max"]):
                    filtered_dataset.append(
                        {
                            "prompt": prompt,
                            "prompt_len": prompt_len,
                            "output_len": output_len
                        }
                    )
                    in_out_map["count"] += 1
                    count += 1
                    break

        if count >= total_num:
            break
            

    return filtered_dataset


def sample_longbench_requests(
    longbench_dataset: LongbenchDataset,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Dict]:
    dataset = longbench_dataset.dataset
    print(dataset)
    # Tokenize the prompts and completions.
    prompts = [item['prompt'] for item in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    max_gen_list = [item['max_gen'] for item in dataset]
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = max_gen_list[i]
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter sequences.
    filtered_dataset: List[Dict] = []    
    
    count = 0
    total_num = sum(item["num"] for item in IN_OUT_MAP)
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        
        for in_out_map in IN_OUT_MAP:
            if in_out_map["count"] < in_out_map["num"]:
                if (in_out_map["in_min"] < prompt_len <= in_out_map["in_max"]) and (in_out_map["out_min"] < output_len <= in_out_map["out_max"]):
                    filtered_dataset.append(
                        {
                            "prompt": prompt,
                            "prompt_len": prompt_len,
                            "output_len": int(prompt_len/9)
                            # "output_len": output_len
                        }
                    )
                    in_out_map["count"] += 1
                    count += 1
                    break

        if count >= total_num:
            break
            

    return filtered_dataset


if __name__ == "__main__":
    """
        /share/datasets/public_models/Qwen_Qwen1.5-72B-Chat/
        /share/datasets/public_models/Qwen_Qwen2-72B-Instruct/
        /share/datasets/public_models/Qwen_Qwen2.5-72B-Instruct/
    """
    model_path = "/Users/howardchen/DeepSeek-R1"
    
    tokenizer = get_tokenizer(model_path, trust_remote_code=True)
    tokenizer_config = AutoConfig.from_pretrained(tokenizer.name_or_path, trust_remote_code=True)
    
    # sharegpt_dataset = SharegptDataset()
    # datasets = sample_sharegpt_requests(sharegpt_dataset, tokenizer)
    
    longbench_dataset = LongbenchDataset()
    datasets = sample_longbench_requests(longbench_dataset, tokenizer)
    
    # 初始化统计变量
    datasets = datasets * 10
    datasets_num = len(datasets)
    print(datasets_num)
    total_input_length = 0
    total_output_length = 0

    # 初始化用于统计区间分布的字典
    input_length_distribution = {}
    output_length_distribution = {}

    # 遍历数据集
    for dataset in datasets:
        prompt_len = dataset["prompt_len"]
        output_len = dataset["output_len"]

        # 累加总长度
        total_input_length += prompt_len
        total_output_length += output_len

        # 统计 prompt_len 的分布
        input_bin = (prompt_len // 250) * 250  # 计算所在的区间
        if input_bin not in input_length_distribution:
            input_length_distribution[input_bin] = 0
        input_length_distribution[input_bin] += 1

        # 统计 output_len 的分布
        output_bin = (output_len // 250) * 250  # 计算所在的区间
        if output_bin not in output_length_distribution:
            output_length_distribution[output_bin] = 0
        output_length_distribution[output_bin] += 1

    # 计算平均值
    avg_input_length = total_input_length / datasets_num
    avg_output_length = total_output_length / datasets_num

    # 打印结果
    print("-" * 40)
    print(f"Total datasets: {datasets_num}")
    print(f"Avg prompt_len: {avg_input_length}")
    print(f"Avg output_len: {avg_output_length}")

    # 打印 prompt_len 的分布
    print("Prompt length distribution:")
    for input_bin, count in sorted(input_length_distribution.items()):
        print(f"{input_bin}-{input_bin + 249}: {count} datasets")

    # 打印 output_len 的分布
    print("Output length distribution:")
    for output_bin, count in sorted(output_length_distribution.items()):
        print(f"{output_bin}-{output_bin + 249}: {count} datasets")
    
    print("-" * 40)
    
    file_name = f"maas_{tokenizer_config.model_type}_in{int(avg_input_length)}_out{int(avg_output_length)}_num{datasets_num}.json"
    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(datasets, file, ensure_ascii=False, indent=4)
    print(file_name)
    