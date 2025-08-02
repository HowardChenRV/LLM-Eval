from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from llm_eval.utils.tokenizer import get_tokenizer


@dataclass
class Chunk:
    chunk_time: float   # time.perf_counter()
    response: Any


@dataclass
class RequestFuncInput:
    prompt: str = ""
    idx: Optional[int] = 0
    prompt_len: Optional[int] = None
    
    
@dataclass
class ServerResponses:
    request: RequestFuncInput
    start_time: float           # time.perf_counter()
    end_time: float             # time.perf_counter()
    success: bool = False
    chunk_list: List[Chunk] = field(default_factory=list)
    error_msg: str = ""
    status_code: int = 0
    response_type: str = ""
    

@dataclass
class RequestFuncOutput:
    idx: int
    start_time: float           # time.perf_counter()
    end_time: float             # time.perf_counter()
    chunk_list: List[Chunk] = field(default_factory=list)
    generated_text: str = ""
    prompt_len: int = 0
    output_len: int = 0
    reasoning_text: str = ""
    reasoning_len: int = 0
    audit_hit: bool = False
    success: bool = False
    error: str = ""
    extra: Optional[Dict] = None


class ApiBase(ABC):
    
    ENDPOINT = "/v1/chat/completions"
    
    def __init__(self, api_url: str, model_path: str = None) -> None:
        self.api_url = api_url
        assert api_url.endswith(
            self.ENDPOINT
        ), f"{self.__class__.__name__} API URL must end with '{self.ENDPOINT}'."
        
        if model_path:
            self.tokenizer = get_tokenizer(model_path)
        else:
            self.tokenizer = None
    
    
    @abstractmethod
    def create_request_input(self, **kwargs) -> RequestFuncInput:
        raise NotImplementedError
    

    @abstractmethod
    def build_request(self, param: RequestFuncInput) -> Dict:
        raise NotImplementedError
    
    
    @abstractmethod
    def parse_responses(self, server_responses: ServerResponses, **kwargs: Any) -> RequestFuncOutput:
        raise NotImplementedError 


