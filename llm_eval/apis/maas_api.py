import json
from dataclasses import dataclass
from typing import Dict, get_type_hints
from llm_eval.utils.registry import register_api
from .openai_api import OpenaiApi, OpenaiRequestFuncInput
from .base_api import RequestFuncOutput, ServerResponses


@dataclass
class MaasRequestFuncInput(OpenaiRequestFuncInput):
    ignore_eos: bool = False


@register_api("maas")
class MaasApi(OpenaiApi):
    
    ENDPOINT = "/chat/completions"
        
    def __init__(self, api_url: str, model_path: str = None) -> None:
        super().__init__(api_url=api_url, model_path=model_path)
    
    
    def create_request_input(self, **kwargs) -> MaasRequestFuncInput:
        filtered_dict = {k: v for k, v in kwargs.items() if k in get_type_hints(MaasRequestFuncInput)}
        return MaasRequestFuncInput(**filtered_dict)
    
    
    def build_request(self, param: MaasRequestFuncInput) -> Dict:
        payload = super().build_request(param)
        
        if not param.custom_body:   # if custom_body is not provided, add the maas parameter
            payload["ignore_eos"] = param.ignore_eos
        return payload
    
    
    def parse_responses(self, server_responses: ServerResponses) -> RequestFuncOutput:
        output = super().parse_responses(server_responses=server_responses)
        chunk_list = server_responses.chunk_list
        
        # add blocked
        for chunk in chunk_list:
            json_res = json.loads(chunk.response)
            if "blocked" in json_res and json_res["blocked"] == True:
                output.generated_text = "Request was blocked by security review. Please disable the security logic and try again."
                output.output_len = 0
                output.audit_hit = True
                break
        
        return output
    