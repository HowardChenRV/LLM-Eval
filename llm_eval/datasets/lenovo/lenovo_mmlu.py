import pandas as pd
import os
import json
import logging
from datasets import Dataset
from typing import Iterator
from ..base_dataset import DatasetBase, PerfReq
from transformers import PreTrainedTokenizerBase
from llm_eval.utils.registry import register_dataset
from llm_eval.utils.datasets import cache_perf_dataset, get_perf_dataset_by_cache


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_NAME = "lenovo_mmlu"

@register_dataset(DATASET_NAME)
class LenovoMMLU(DatasetBase):
    """MMLU dataset implementation for Lenovo evaluation"""
    
    @staticmethod
    def load(path: str = "") -> Dataset:
        """Load MMLU dataset from json file"""
        if path:
            data_path = path
        else:
            data_path = os.path.join(CURRENT_DIR, "mmlu_case.json")
            
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to list of messages format
        samples = []
        for key, value in data.items():
            messages = []
            for prompt in value["origin_prompt"]:
                role_mapping = {
                    "BOT": "assistant",
                    "SYSTEM": "system",
                    "HUMAN": "user"
                }
                role = role_mapping.get(prompt["role"], "user")
                messages.append({
                    "role": role,
                    "content": prompt["prompt"]
                })
            
            samples.append({
                "idx": key,
                "messages": messages
            })
        
        df = pd.DataFrame(samples)
        dataset = Dataset.from_pandas(df)
        return dataset
    
    def req_generator(self, num_requests: int) -> Iterator[dict]:
        """Generate requests in messages format"""
        dataset_num = len(self.dataset)
        assert dataset_num >= num_requests, (
            f"Dataset num not enough, expect={num_requests}, "
            f"actual={dataset_num}, Please reduce the num_requests."
        )
        
        count = 0
        for data in self.dataset:
            if count >= num_requests:
                break
            yield {
                "idx": data["idx"],
                "messages": data["messages"]
            }
            count += 1
            

    def perf_req_generator(
        self, 
        num_requests: int, 
        tokenizer: PreTrainedTokenizerBase
    ) -> Iterator[PerfReq]:
        assert tokenizer is not None, "Tokenizer must be provided for Lenovo MMLU dataset."
        # get dataset by cache
        data_list = get_perf_dataset_by_cache(DATASET_NAME, tokenizer=tokenizer, num_requests=num_requests)
        
        if data_list is None or len(data_list) < num_requests:
            logging.info("Didn't found the perf dataset int cache dir, starting filter data...")
            
            filter_data_list = []
            count = 0
            for data in self.dataset:
                if count >= num_requests:
                    break  # Stop if we've reached the request limit
                
                prompt = data["messages"][0]["content"]
                
                filter_data_list.append(PerfReq(
                    idx=count,
                    prompt=prompt,
                    prompt_len=len(tokenizer(prompt).input_ids),
                    output_len=1024
                ))
                count += 1
            
            assert count >= num_requests, f"Dataset num not enough, expect={num_requests}, actural={count}, Please reduce the num_requests."
            # Caching tokenized_dataset as a '.jsonl' file to the cache path.
            cache_perf_dataset(DATASET_NAME, tokenizer=tokenizer, data_list=filter_data_list)
            data_list = filter_data_list
        
        for perf_req in data_list:
            yield perf_req