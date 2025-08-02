from datasets import load_dataset, Dataset
from typing import Iterator
from transformers import PreTrainedTokenizerBase
from .base_dataset import DatasetBase, PerfReq
from llm_eval.utils.registry import register_dataset
from llm_eval.utils.datasets import dowanload_dataset_by_url, cache_perf_dataset, get_perf_dataset_by_cache
import logging

DATASET_NAME = "sharegpt"

@register_dataset(DATASET_NAME)
class SharegptDataset(DatasetBase):
    """
        Datasets: https://www.modelscope.cn/datasets/otavia/ShareGPT_Vicuna_unfiltered/resolve/master/ShareGPT_V3_unfiltered_cleaned_split.json
    """
    
    @staticmethod
    def load(path: str="") -> Dataset:
        if path:
            file_path = path
        else:
            file_path = dowanload_dataset_by_url(
                name = DATASET_NAME,
                download_url = "https://www.modelscope.cn/datasets/otavia/ShareGPT_Vicuna_unfiltered/resolve/master/ShareGPT_V3_unfiltered_cleaned_split.json"
            )
            
        dataset = load_dataset("json", data_files=file_path)
        return dataset["train"]
    
    
    def perf_req_generator(
        self, 
        num_requests: int, 
        tokenizer: PreTrainedTokenizerBase
    ) -> Iterator[PerfReq]:
        assert tokenizer is not None, "Tokenizer must be provided for sharegpt dataset."
        # get dataset by cache
        data_list = get_perf_dataset_by_cache(DATASET_NAME, tokenizer=tokenizer, num_requests=num_requests)
        
        if data_list is None or len(data_list) < num_requests:
            logging.info("Didn't found the perf dataset int cache dir, starting filter data...")
            filter_data_list = []
            # Filter out the conversations with less than 2 turns.
            dataset = [data for data in self.dataset if len(data["conversations"]) >= 2]
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

            # Filter out too long sequences.
            count = 0
            for prompt, prompt_token_ids, output_len in tokenized_dataset:
                if count >= num_requests:
                    break  # Stop if we've reached the request limit
                
                prompt_len = len(prompt_token_ids)
                if prompt_len < 4 or output_len < 4:
                    # Prune too short sequences.
                    # This is because TGI causes errors when the input or output length
                    # is too short.
                    continue
                if prompt_len > 1024 or prompt_len + output_len > 2048:
                    # Prune too long sequences.
                    continue
                filter_data_list.append(PerfReq(
                    idx=count,
                    prompt=prompt,
                    prompt_len=prompt_len,
                    output_len=output_len
                ))
                count += 1
            
            assert count >= num_requests, f"Dataset num not enough, expect={num_requests}, actural={count}, Please reduce the num_requests."
            # Caching tokenized_dataset as a '.jsonl' file to the cache path.
            cache_perf_dataset(DATASET_NAME, tokenizer=tokenizer, data_list=filter_data_list)
            data_list = filter_data_list
        
        for perf_req in data_list:
            yield perf_req
            