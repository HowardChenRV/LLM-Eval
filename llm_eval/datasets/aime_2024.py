from datasets import load_dataset, Dataset
from typing import Iterator
from transformers import PreTrainedTokenizerBase
from .base_dataset import DatasetBase, PerfReq
from llm_eval.utils.registry import register_dataset
from llm_eval.utils.datasets import download_dataset_by_hub, cache_perf_dataset, get_perf_dataset_by_cache
import logging

DATASET_NAME = "aime_2024"

@register_dataset(DATASET_NAME)
class AIME2024(DatasetBase):
    """
        Datasets: https://huggingface.co/datasets/Maxwell-Jia/AIME_2024
    """
    
    @staticmethod
    def load(path: str="") -> Dataset:
        if path:
            dataset = load_dataset("json", data_files=path)
        else:
            dataset = download_dataset_by_hub("Maxwell-Jia/AIME_2024", name="default", split="train")
        return dataset
    
    
    def perf_req_generator(
        self, 
        num_requests: int, 
        tokenizer: PreTrainedTokenizerBase
    ) -> Iterator[PerfReq]:
        assert tokenizer is not None, "Tokenizer must be provided for AIME 2024 dataset."
        # get dataset by cache
        data_list = get_perf_dataset_by_cache(DATASET_NAME, tokenizer=tokenizer, num_requests=num_requests)
        
        if data_list is None or len(data_list) < num_requests:
            logging.info("Didn't found the perf dataset int cache dir, starting filter data...")
            
            filter_data_list = []
            count = 0
            for data in self.dataset:
                if count >= num_requests:
                    break  # Stop if we've reached the request limit
                
                prompt = data["Problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{{}}."
                
                filter_data_list.append(PerfReq(
                    idx=count,
                    prompt=prompt,
                    prompt_len=len(tokenizer(prompt).input_ids),
                    # output_len=len(tokenizer(f"Solution:\n{data['Solution']}\nAnswer:\n{data['Answer']}").input_ids)
                    output_len=32768
                ))
                count += 1
            
            assert count >= num_requests, f"Dataset num not enough, expect={num_requests}, actural={count}, Please reduce the num_requests."
            # Caching tokenized_dataset as a '.jsonl' file to the cache path.
            cache_perf_dataset(DATASET_NAME, tokenizer=tokenizer, data_list=filter_data_list)
            data_list = filter_data_list
        
        for perf_req in data_list:
            yield perf_req
            