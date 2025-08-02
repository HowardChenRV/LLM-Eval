from dataclasses import dataclass, field
from typing import Dict, List, Optional, get_type_hints
from llm_eval.utils.registry import register_api
from .openai_api import OpenaiApi, OpenaiRequestFuncInput
from .base_api import RequestFuncOutput, ServerResponses


@dataclass
class VllmRequestFuncInput(OpenaiRequestFuncInput):
    ignore_eos: bool = False
    stop_token_ids: Optional[List[int]] = field(default_factory=list)
    

@register_api("vllm")
class VllmApi(OpenaiApi):
        
    def __init__(self, api_url: str, model_path: str = None) -> None:
        super().__init__(api_url=api_url, model_path=model_path)


    def create_request_input(self, **kwargs) -> VllmRequestFuncInput:
        filtered_dict = {k: v for k, v in kwargs.items() if k in get_type_hints(VllmRequestFuncInput)}
        return VllmRequestFuncInput(**filtered_dict)
    
    
    def build_request(self, param: VllmRequestFuncInput) -> Dict:
        payload = super().build_request(param)
        
        if not param.custom_body:  # if custom_body is not provided, add the vllm parameters
            payload.update({
                "ignore_eos": param.ignore_eos,
                "stop_token_ids": param.stop_token_ids
            })
        return payload
    
    
    def parse_responses(self, server_responses: ServerResponses) -> RequestFuncOutput:
        return super().parse_responses(server_responses=server_responses)
    