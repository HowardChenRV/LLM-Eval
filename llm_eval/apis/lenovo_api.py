import json
import logging
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, get_type_hints
from llm_eval.utils.registry import register_api
from .base_api import ApiBase, RequestFuncInput, RequestFuncOutput, ServerResponses


@dataclass
class LenovoRequestFuncInput(RequestFuncInput):
    stream: bool = True
    n_ctx: Optional[int] = None
    n_predict: Optional[int] = None
    seed: Optional[int] = None
    temp: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repeat_penalty: Optional[float] = None
    

@register_api("lenovo")
class LenovoApi(ApiBase):
    
    ENDPOINT = "/app/v1/infer/llm/chunked"
    
    def __init__(self, api_url: str, model_path: str = None) -> None:
        super().__init__(api_url=api_url, model_path=model_path)


    def create_request_input(self, **kwargs) -> LenovoRequestFuncInput:
        filtered_dict = {k: v for k, v in kwargs.items() if k in get_type_hints(LenovoRequestFuncInput)}
        return LenovoRequestFuncInput(**filtered_dict)

    
    def build_request(self, param: LenovoRequestFuncInput) -> Dict:
        payload = {
            "prompt": param.prompt,
            "stream": param.stream,
            "n_ctx": param.n_ctx,
            "n_predict": param.n_predict,
            "seed": param.seed,
            "temp": param.temp,
            "top_k": param.top_k,
            "top_p": param.top_p,
            "repeat_penalty": param.repeat_penalty
        }
        return {k: v for k, v in payload.items() if v is not None}
    
    
    def parse_responses(self, server_responses: ServerResponses) -> RequestFuncOutput:
        request = server_responses.request
        chunk_list = server_responses.chunk_list
        output = RequestFuncOutput(
            idx = request.idx,
            start_time = server_responses.start_time,
            end_time = server_responses.end_time
        )
        output.chunk_list = chunk_list
        timings = None
        
        try:
            for chunk in chunk_list:
                # print(f"chunk: {chunk}")
                json_res = json.loads(chunk.response)
                output.generated_text += json_res.get("content", "").encode('utf-8').decode('unicode-escape')
                if output.generated_text == "":
                    output.success = False
                    output.error = "Empty content in response"
                    return output
                
                if "timings" in json_res and json_res["timings"]:
                    timings = json_res["timings"]

                    
            if timings:
                output.prompt_len = timings.get("prompt_n", 0)
                output.output_len = timings.get("predicted_n", 0)
                output.extra = {"timings": timings}
            else:
                output.success = False
                output.error = "No timings in response"
                return output
                
            output.success = True

        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
            
        return output
    