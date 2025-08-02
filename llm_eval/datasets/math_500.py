from datasets import load_dataset, Dataset
from typing import Iterator
from transformers import PreTrainedTokenizerBase
from .base_dataset import DatasetBase, PerfReq
from llm_eval.utils.registry import register_dataset
from llm_eval.utils.datasets import dowanload_dataset_by_url, cache_perf_dataset, get_perf_dataset_by_cache
import logging

DATASET_NAME = "math-500"

@register_dataset(DATASET_NAME)
class Math500(DatasetBase):
    """
        Datasets: https://www.modelscope.cn/datasets/AI-ModelScope/MATH-500/resolve/master/test.jsonl
    """
    
    @staticmethod
    def load(path: str="") -> Dataset:
        if path:
            file_path = path
        else:
            file_path = dowanload_dataset_by_url(
                name = DATASET_NAME,
                download_url = "https://www.modelscope.cn/datasets/AI-ModelScope/MATH-500/resolve/master/test.jsonl"
            )
            
        dataset = load_dataset("json", data_files=file_path)
        return dataset["train"]
    
    
    def perf_req_generator(
        self, 
        num_requests: int, 
        tokenizer: PreTrainedTokenizerBase
    ) -> Iterator[PerfReq]:
        assert tokenizer is not None, "Tokenizer must be provided for math-500 dataset."
        # get dataset by cache
        data_list = get_perf_dataset_by_cache(DATASET_NAME, tokenizer=tokenizer, num_requests=num_requests)
        
        system_prompt = "\nPlease reason step by step, and put your final answer within \boxed{}."
        
        if data_list is None or len(data_list) < num_requests:
            logging.info("Didn't found the perf dataset int cache dir, starting filter data...")
            
            filter_data_list = []
            count = 0
            for data in self.dataset:
                if count >= num_requests:
                    break  # Stop if we've reached the request limit
                
                filter_data_list.append(PerfReq(
                    idx=count,
                    prompt=data["problem"]+system_prompt,
                    prompt_len=len(tokenizer(data["problem"]+system_prompt).input_ids),
                    # output_len=len(tokenizer(data["solution"]+data["answer"]).input_ids)
                    output_len=4096
                ))
                count += 1
            
            assert count >= num_requests, f"Dataset num not enough, expect={num_requests}, actural={count}, Please reduce the num_requests."
            # Caching tokenized_dataset as a '.jsonl' file to the cache path.
            cache_perf_dataset(DATASET_NAME, tokenizer=tokenizer, data_list=filter_data_list)
            data_list = filter_data_list
        
        for perf_req in data_list:
            yield perf_req
            