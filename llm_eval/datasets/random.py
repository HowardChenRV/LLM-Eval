from datasets import Dataset
from typing import Iterator, Dict
from transformers import PreTrainedTokenizerBase
from .base_dataset import DatasetBase, PerfReq
import numpy as np
from llm_eval.utils.registry import register_dataset
from llm_eval.utils.datasets import cache_perf_dataset, get_perf_dataset_by_cache
import logging

DATASET_NAME = "random"

@register_dataset(DATASET_NAME)
class RandomDataset(DatasetBase):
    DEFAULT_PREFIX_LEN = 0
    DEFAULT_RANGE_RATIO = 0.0
    DEFAULT_INPUT_LEN = 1024
    DEFAULT_OUTPUT_LEN = 128

    @staticmethod
    def load(
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        range_ratio: float = DEFAULT_RANGE_RATIO,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
    ) -> Dict:
        assert tokenizer is not None, "Tokenizer must be provided for random dataset."
        # Enforce range_ratio < 1
        assert range_ratio < 1.0, (
            "random_range_ratio must be < 1.0 to ensure a valid sampling range"
        )
        dataset_name = f"random_prefix_{prefix_len}_range_{range_ratio}_input_{input_len}_output_{output_len}"

        vocab_size = tokenizer.vocab_size
        num_special_tokens = tokenizer.num_special_tokens_to_add()
        real_input_len = input_len - num_special_tokens

        prefix_token_ids = (
            np.random.randint(0, vocab_size, size=prefix_len).tolist()
            if prefix_len > 0
            else []
        )

        # New sampling logic: [X * (1 - b), X * (1 + b)]
        input_low = int(real_input_len * (1 - range_ratio))
        input_high = int(real_input_len * (1 + range_ratio))
        output_low = int(output_len * (1 - range_ratio))
        output_high = int(output_len * (1 + range_ratio))

        # Add logging for debugging
        logging.info("Sampling input_len from [%s, %s]", input_low, input_high)
        logging.info("Sampling output_len from [%s, %s]", output_low, output_high)

        input_lens = np.random.randint(input_low, input_high + 1, size=num_requests)
        output_lens = np.random.randint(output_low, output_high + 1, size=num_requests)
        offsets = np.random.randint(0, vocab_size, size=num_requests)
        
        requests = []
        for i in range(num_requests):
            inner_seq = (
                (offsets[i] + i + np.arange(input_lens[i])) % vocab_size
            ).tolist()
            token_sequence = prefix_token_ids + inner_seq
            prompt = tokenizer.decode(token_sequence)
            # After decoding the prompt we have to encode and decode it again.
            # This is done because in some cases N consecutive tokens
            # give a string tokenized into != N number of tokens.
            # For example for GPT2Tokenizer:
            # [6880, 6881] -> ['Ġcalls', 'here'] ->
            # [1650, 939, 486] -> ['Ġcall', 'sh', 'ere']
            # To avoid uncontrolled change of the prompt length,
            # the encoded sequence is truncated before being decode again.
            re_encoded_sequence = tokenizer.encode(prompt, add_special_tokens=False)[
                : input_lens[i]
            ]
            prompt = tokenizer.decode(re_encoded_sequence)
            total_input_len = len(re_encoded_sequence)
            requests.append({
                "prompt": prompt,
                "prompt_len": total_input_len,
                "output_len": int(output_lens[i])
            })
        # Convert requests list to Dataset
        dataset = Dataset.from_list(requests)
        # Add dataset name as metadata
        dataset.info.description = dataset_name
        return dataset
        
    
    def perf_req_generator(
        self, 
        num_requests: int, 
        tokenizer: PreTrainedTokenizerBase
    ) -> Iterator[PerfReq]:
        # get dataset by cache
        dataset_name = self.dataset.info.description
        data_list = get_perf_dataset_by_cache(dataset_name, tokenizer=tokenizer, num_requests=num_requests)
                
        if data_list is None or len(data_list) < num_requests:
            logging.info("Didn't found the perf dataset int cache dir, starting filter data...")
            
            filter_data_list = []
            count = 0
            for data in self.dataset:
                if count >= num_requests:
                    break  # Stop if we've reached the request limit
                
                # Convert Dataset row to PerfReq
                perf_req = PerfReq(
                    idx=count,
                    prompt=data["prompt"],
                    prompt_len=data["prompt_len"],
                    output_len=data["output_len"]
                )
                filter_data_list.append(perf_req)
                count += 1
            
            assert count >= num_requests, f"Dataset num not enough, expect={num_requests}, actural={count}, Please reduce the num_requests."
            # Caching tokenized_dataset as a '.jsonl' file to the cache path.
            cache_perf_dataset(dataset_name, tokenizer=tokenizer, data_list=filter_data_list)
            data_list = filter_data_list
        
        for perf_req in data_list:
            yield perf_req
            