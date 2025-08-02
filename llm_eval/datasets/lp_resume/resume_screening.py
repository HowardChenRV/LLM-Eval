from datasets import Dataset
from faker import Faker
import pandas as pd
from typing import Iterator
from transformers import PreTrainedTokenizerBase
from ..base_dataset import DatasetBase, PerfReq
from llm_eval.utils.registry import register_dataset
from llm_eval.utils.datasets import cache_perf_dataset, get_perf_dataset_by_cache
import logging
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_NAME = "resume_screening"

@register_dataset(DATASET_NAME)
class ResumeScreening(DatasetBase):
    
    template = """
    候选人简历：
    \"""
    "{{CV}}"
    \"""
    招聘职位说明：
    \"""
    "{{JD}}"
    \"""
    你是一个经验丰富的招聘专家，已经仔细审阅了候选人的简历和招聘职位说明。请结合简历信息和招聘职位说明判断是否匹配，并给出分析过程。
    ##输出格式
    ```
    匹配结果: 是或否。
    分析过程: ***。
    ```
    """
 
    @staticmethod
    def load(path: str="") -> Dataset:
        if path:
            sample_path = path
        else:
            sample_path = os.path.join(CURRENT_DIR, "e2e_resume_sample.xlsx")
            
        df = pd.read_excel(sample_path)
        dataset = Dataset.from_pandas(df)
    
        return dataset
    
    
    def perf_req_generator(
        self, 
        num_requests: int, 
        tokenizer: PreTrainedTokenizerBase
    ) -> Iterator[PerfReq]:
        assert tokenizer is not None, "Tokenizer must be provided for resume screening dataset."
        # get dataset by cache
        data_list = get_perf_dataset_by_cache(DATASET_NAME, tokenizer=tokenizer, num_requests=num_requests)
        
        if data_list is None or len(data_list) < num_requests:
            logging.info("Didn't found the perf dataset int cache dir, starting filter data...")
            filter_data_list = []
            fake = Faker()
            
            sample_num = len(self.dataset)
            for count in range(num_requests):
                sample_data = self.dataset[count % sample_num]
                prompt = self.template.replace("{{CV}}", sample_data["CV"]).replace("{{JD}}", sample_data["JD"])
                prompt = "候选人姓名：" + fake.name() + "\n" + prompt
                answer = sample_data["人工标注结果"]
                prompt_len = len(tokenizer(prompt).input_ids)
                # output_len = len(tokenizer(answer).input_ids)
                filter_data_list.append(PerfReq(
                    idx=count,
                    prompt=prompt,
                    prompt_len=prompt_len,
                    output_len=1024
                ))
                count += 1
            
            # Caching tokenized_dataset as a '.jsonl' file to the cache path.
            cache_perf_dataset(DATASET_NAME, tokenizer=tokenizer, data_list=filter_data_list)
            data_list = filter_data_list
        
        for perf_req in data_list:
            yield perf_req
            