import os
import json
import logging
from ..base_dataset import DatasetBase
from llm_eval.utils.registry import register_dataset
from datasets import Dataset
from typing import Dict, AsyncIterator
from transformers import PreTrainedTokenizerBase
from llm_eval.utils.datasets import download_dataset_by_hub

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_NAME = "longbench"

@register_dataset(DATASET_NAME)
class LongbenchDataset(DatasetBase):
    """
        Datasets: https://huggingface.co/datasets/THUDM/LongBench
    """
    
    @staticmethod
    def load() -> Dataset:
        dataset_name = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
        dataset2prompt = json.load(open(os.path.join(CURRENT_DIR, "dataset2prompt.json"), "r"))
        dataset2maxlen = json.load(open(os.path.join(CURRENT_DIR, "dataset2maxlen.json"), "r"))
        dataset_list = []
        for name in dataset_name:
            dataset = download_dataset_by_hub("THUDM/LongBench", name=name, split="test")
            prompt_format = dataset2prompt[name]
            max_gen = dataset2maxlen[name]        
            for data in dataset:
                dataset_list.append({
                    "prompt": prompt_format.format(**data),
                    "max_gen": max_gen,
                    "dataset": data["dataset"],
                    "language": data["language"],
                    "_id": data["_id"]
                })   
        return Dataset.from_list(dataset_list)
    
    
    async def perf_req_generator(
        self, 
        num_requests: int, 
        tokenizer: PreTrainedTokenizerBase
    ) -> AsyncIterator[Dict]:
        assert tokenizer is not None, "Tokenizer must be provided for Longbench dataset."
        
        # get dataset by cache
        data_list = self.get_perf_dataset_by_cache(DATASET_NAME, tokenizer=tokenizer, num_requests=num_requests)
        
        if data_list is None or len(data_list) < num_requests:
            logging.info("Didn't found the perf dataset in cache dir, starting filter data...")
            filter_data_list = []
            count = 0
            
            for data in self.dataset:
                if count >= num_requests:
                    break  # Stop if we've reached the request limit
                
                prompt = data["prompt"]
                max_gen = data["max_gen"]
                
                filter_data_list.append({
                    "idx": count,
                    "prompt": prompt,
                    "prompt_len": len(tokenizer(prompt).input_ids),
                    "output_len": max_gen
                })
                count += 1
            
            assert count >= num_requests, f"Dataset num not enough, expect={num_requests}, actual={count}, Please reduce the num_requests."
            # Caching tokenized_dataset as a '.jsonl' file to the cache path.
            self.cache_perf_dataset(DATASET_NAME, tokenizer=tokenizer, data_list=filter_data_list)
            data_list = filter_data_list
        
        for perf_req in data_list:
            yield perf_req
