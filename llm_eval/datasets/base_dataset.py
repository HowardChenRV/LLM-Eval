from abc import ABC, abstractmethod
from datasets import Dataset, DatasetDict
from typing import Union, Iterator, Dict
from dataclasses import dataclass


@dataclass
class PerfReq:
    idx: int
    prompt: str
    prompt_len: int
    output_len: int


class DatasetBase(ABC):
    """ 
    Abstract base class for datasets.
    This class provides a template for loading datasets and generating performance requests.
    """
    
    def __init__(self, **kwargs):
        """
        Initializes the dataset by loading it using the load method.

        Args:
            **kwargs: Additional keyword arguments passed to the load method.
        """
        self.dataset = self.load(**kwargs)
    
    
    @staticmethod
    @abstractmethod
    def load(**kwargs) -> Union[Dataset, DatasetDict, Dict]:
        """
        Abstract method to load the dataset.

        Args:
            **kwargs: Additional keyword arguments for loading the dataset.

        Returns:
            Union[Dataset, DatasetDict, Dict]: The loaded dataset or dataset dictionary.
        """
        raise NotImplementedError
    
    
    def req_generator(self, num_requests: int) -> Iterator[Dict]:
        """
        Generator for dataset requests.
        
        Yields each element from the dataset as a dictionary.
        
        Args:
            num_requests: Number of requests to generate
            
        Yields:
            Dict: A dictionary containing the dataset element
        """
        dataset_num = len(self.dataset)
        assert dataset_num >= num_requests, f"Dataset num not enough, expect={num_requests}, actual={dataset_num}, Please reduce the num_requests."
        
        count = 0
        for data in self.dataset:
            if count >= num_requests:
                break
            # print(f"Generating request {count + 1}/{num_requests}: {data}")
            yield dict(data)
            count += 1
    
    
    """
        perf_req dataset format:
        {"idx": 0, "prompt": "Hi!", "prompt_len": 1, "output_len": 128}
        {"idx": 1, "prompt": "Hi!", "prompt_len": 1, "output_len": 128}
        ...
    """
    def perf_req_generator(self, num_requests: int) -> Iterator[PerfReq]:
        """
        Generator for performance test requests.

        This method simulates generating performance requests, yielding a new request.
        """
        for i in range(num_requests):
            yield PerfReq(
                idx=i, 
                prompt="Hi!", 
                prompt_len=1,
                output_len=128
            )

