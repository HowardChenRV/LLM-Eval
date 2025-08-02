import os
import logging
import json
from datasets import load_dataset, Dataset
from typing import Iterator
from transformers import PreTrainedTokenizerBase
from .base_dataset import DatasetBase, PerfReq
from llm_eval.utils.registry import register_dataset


PERF_DATASET_NAME = "local"     # local perf dataset, use for serving performance benchmark
CUSTOM_BODY_DATASET_NAME = "custom"  # custom payload dataset


@register_dataset(PERF_DATASET_NAME)
class LocalPerfDataset(DatasetBase):
    """本地性能测试数据集加载器，支持JSON和JSONL格式
    
    数据集格式要求：
    [
        {
            "prompt": "问题文本",  # 必填，输入提示文本
            "prompt_len": 512,    # 必填，输入token长度
            "output_len": 128     # 必填，期望输出token长度
        },
        ...
    ]
    
    支持的文件格式：
    - .json: 标准JSON格式文件
    - .jsonl: JSON Lines格式文件（每行一个JSON对象）
    """
    
    @staticmethod
    def load(path: str="") -> Dataset:
        """加载本地数据集文件
        
        Args:
            path: 数据集文件路径
            
        Returns:
            Dataset: 加载后的数据集对象
        """
        filename = LocalPerfDataset.check_local_dataset(path)
        logging.info(f"Loading local dataset: {filename}")
        dataset = load_dataset("json", data_files=path)
        return dataset["train"]
    
    @staticmethod
    def check_local_dataset(path: str) -> str:
        """检查本地数据集文件有效性
        
        Args:
            path: 数据集文件路径
            
        Returns:
            str: 数据集文件名（不含扩展名）
            
        Raises:
            ValueError: 如果文件路径无效或格式不支持
        """
        if not os.path.exists(path):
            raise ValueError(f"Local dataset path does not exist: {path}")
        
        if not os.path.isfile(path):
            raise ValueError(f"Local dataset path is not a file: {path}")
        
        if not (path.lower().endswith('.json') or path.lower().endswith('.jsonl')):
            raise ValueError(f"Local dataset file must be JSON or JSONL format: {path}")
        
        return os.path.splitext(os.path.basename(path))[0]
        
    def perf_req_generator(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase
    ) -> Iterator[PerfReq]:
        """生成性能测试请求
        
        Args:
            num_requests: 需要生成的请求数量
            tokenizer: 用于tokenize的tokenizer对象
            
        Yields:
            PerfReq: 性能测试请求对象
            
        Raises:
            AssertionError: 如果数据集数量不足
        """
        dataset_num = len(self.dataset)
        assert dataset_num >= num_requests, f"Dataset num not enough, expect={num_requests}, actual={dataset_num}, Please reduce the num_requests."
            
        for count, data in enumerate(self.dataset):
            if count >= num_requests:
                break  # 达到请求数量后立即停止迭代
                
            yield PerfReq(
                idx=count,
                prompt=data["prompt"],
                prompt_len=data["prompt_len"],
                output_len=data["output_len"]
            )


@register_dataset(CUSTOM_BODY_DATASET_NAME)
class CustomBodyDataset(DatasetBase):
    """
        .jsonl
        ...
        {"idx": "request-1", "prompt_body": {"model": "gpt-4o-mini", "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is 2+2?"}]}}
        {"idx": "request-2", "prompt_body": {"model": "gpt-4o-mini", "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is 2+2?"}]}}
        ...
    """
    
    @staticmethod
    def load(path: str="") -> Dataset:
        filename = CustomBodyDataset.check_local_dataset(path)
        logging.info(f"Loading custom body dataset: {filename}")
        dataset = load_dataset("json", data_files=path)
        
        # 转换prompt_body为字典
        def transform_prompt_body(example):
            if 'prompt_body' in example:
                if isinstance(example['prompt_body'], dict):
                    example['body'] = example['prompt_body']
                elif isinstance(example['prompt_body'], str):
                    try:
                        example['body'] = json.loads(example['prompt_body'])
                    except json.JSONDecodeError:
                        logging.warning(f"Failed to parse prompt_body: {example['prompt_body']}")
                        example['body'] = {}
                else:
                    logging.warning(f"Unexpected prompt_body type: {type(example['prompt_body'])}")
                    example['body'] = {}
            return example
            
        dataset = dataset['train'].map(transform_prompt_body)
        return dataset
    
    @staticmethod
    def check_local_dataset(path: str) -> str:
        if not os.path.exists(path):
            raise ValueError(f"Custom body dataset path does not exist: {path}")
        
        if not os.path.isfile(path):
            raise ValueError(f"Custom body dataset path is not a file: {path}")
        
        if not path.lower().endswith('.jsonl'):
            raise ValueError(f"Custom body dataset file is not a JSONL file: {path}")
        
        return os.path.splitext(os.path.basename(path))[0]
        
    