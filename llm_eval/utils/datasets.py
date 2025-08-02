import os
import re
import json
import requests
from datasets import load_dataset
from modelscope.msdatasets import MsDataset
from tqdm import tqdm
from urllib.parse import urlparse
from transformers import PreTrainedTokenizerBase, AutoConfig
from llm_eval.datasets.base_dataset import PerfReq
from typing import List
import logging

USER_HOME = os.path.expanduser("~")
DEFAULT_DATA_FOLDER = os.path.join(USER_HOME, '.cache/llm_eval/original_datasets/')
PERF_DATA_FOLDER = os.path.join(USER_HOME, '.cache/llm_eval/perf_datasets/')


def sanitize_path_component(component: str) -> str:
    """
    Sanitize the path component by replacing specific special characters
    that are not suitable for directory/file names.
    """
    return re.sub(r'[<>:"/\\|?*]', '_', component)


def download_dataset_by_hub(path: str, name: str, split: str = "test"):
    # update the path with CACHE_DIR
    cache_dir = os.environ.get("LLM_EVAL_DATA_CACHE", DEFAULT_DATA_FOLDER)
    sanitized_path = sanitize_path_component(path)
    dataset_cache_path = os.path.join(cache_dir, sanitized_path, name, split)
    
    dataset_source = os.environ.get("DATASET_SOURCE", "")
    
    if os.path.exists(dataset_cache_path):
        logging.info(f"Loading local cached dataset: {name} from {cache_dir}")
        data = load_dataset(path, name=name, split=split, cache_dir=cache_dir)
    else:
        if dataset_source == "ModelScope":
            logging.info(f"Downloading dataset: {name} from ModelScope to {cache_dir}")
            data = MsDataset.load(path, subset_name=name, split=split, cache_dir=cache_dir)
        else:
            logging.info(f"Downloading dataset: {name} to {cache_dir}")
            data = load_dataset(path, name=name, split=split, cache_dir=cache_dir)
    
    return data


def dowanload_dataset_by_url(name: str, download_url: str) -> str:
    cache_dir = os.environ.get("LLM_EVAL_DATA_CACHE", DEFAULT_DATA_FOLDER)
    dataset_cache_path = os.path.join(cache_dir, name)
    
    filename = os.path.basename(urlparse(download_url).path)
    local_file = os.path.join(dataset_cache_path, filename)
    
    if os.path.exists(local_file):
        logging.info(f"Loading local cached dataset: {name} from {local_file}")
        return local_file
    
    logging.info(f"Download dataset from {download_url}.")
    os.makedirs(dataset_cache_path, exist_ok=True)
    response = requests.get(download_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(local_file, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            f.write(data)
            bar.update(len(data))

    logging.info(f"Dataset downloaded and saved to {local_file}")
    return local_file


"""
    perf_req dataset format:
    {"idx": 0, "prompt": "Hi!", "prompt_len": 256, "output_len": 128}
    {"idx": 1, "prompt": "Hi!", "prompt_len": 256, "output_len": 128}
    ...
"""
def cache_perf_dataset(name: str, tokenizer: PreTrainedTokenizerBase, data_list: List[PerfReq]) -> None:
    tokenizer_config = AutoConfig.from_pretrained(tokenizer.name_or_path, trust_remote_code=True)
    file_name = f"{name}_{tokenizer_config.model_type}_num_{len(data_list)}.jsonl"
    cache_dir = os.environ.get("LLM_EVAL_DATA_CACHE", PERF_DATA_FOLDER)
    dataset_cache_path = os.path.join(cache_dir, file_name)

    os.makedirs(cache_dir, exist_ok=True)
    
    with open(dataset_cache_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            json_line = {
                "idx": item.idx,
                "prompt": item.prompt,
                "prompt_len": item.prompt_len,
                "output_len": item.output_len
            }
            f.write(json.dumps(json_line) + '\n')
    
    logging.info(f"Dataset has been cached at {dataset_cache_path}")
    

def get_perf_dataset_by_cache(name: str, tokenizer: PreTrainedTokenizerBase, num_requests: int) -> List[PerfReq]:
    tokenizer_config = AutoConfig.from_pretrained(tokenizer.name_or_path, trust_remote_code=True)
    file_name = f"{name}_{tokenizer_config.model_type}_num_{num_requests}.jsonl"
    cache_dir = os.environ.get("LLM_EVAL_DATA_CACHE", PERF_DATA_FOLDER)
    dataset_cache_path = os.path.join(cache_dir, file_name)
    logging.info(f"Try to get cache perf dataset from {dataset_cache_path}")
    
    if not os.path.isfile(dataset_cache_path):
        logging.info(f"The cache perf dataset from {dataset_cache_path} is empty.")
        return None
    
    data_list = []
    with open(dataset_cache_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                json_line = json.loads(line)
                data_list.append(PerfReq(
                    idx=json_line["idx"],
                    prompt=json_line["prompt"],
                    prompt_len=json_line["prompt_len"],
                    output_len=json_line["output_len"]
                ))
            except (json.JSONDecodeError, KeyError):
                print(f"Skipping invalid line: {line}")
                continue
    
    return data_list if data_list else None

