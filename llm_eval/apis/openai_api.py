import json
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, get_type_hints
from llm_eval.utils.registry import register_api
from .base_api import ApiBase, RequestFuncInput, RequestFuncOutput, ServerResponses


@dataclass
class OpenaiRequestFuncInput(RequestFuncInput):
    model: str = ""
    stream: bool = True
    include_usage: bool = True
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = None     # Be going to deprecated
    max_completion_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop: Optional[Union[str, List]] = None
    custom_body: Optional[Dict] = None  # custom payload, if provided, will override all other fields

    
@register_api("openai")
class OpenaiApi(ApiBase):
    
    ENDPOINT = "/v1/chat/completions"
    
    def __init__(self, api_url: str, model_path: str = None) -> None:
        super().__init__(api_url=api_url, model_path=model_path)


    def create_request_input(self, **kwargs) -> OpenaiRequestFuncInput:
        filtered_dict = {k: v for k, v in kwargs.items() if k in get_type_hints(OpenaiRequestFuncInput)}
        return OpenaiRequestFuncInput(**filtered_dict)

    
    def build_request(self, param: OpenaiRequestFuncInput) -> Dict:
        # if custom_body is provided, use it as the payload
        if param.custom_body:
            payload = param.custom_body
            payload["model"] = param.model  # override the model
            return payload
        
        content = [{"type": "text", "text": param.prompt}]
        
        payload = {
            "model": param.model,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                },
            ]
        }
        optional_params = {
            "temperature": param.temperature,
            "top_p": param.top_p,
            "stop": param.stop,
            "max_completion_tokens": param.max_completion_tokens,
            "max_tokens": param.max_tokens,  # Deprecated, but still supported for compatibility
            "stream": param.stream,
        }
        payload.update({k: v for k, v in optional_params.items() if v is not None})
        if param.stream and param.include_usage is not None:
            payload["stream_options"] = {
                "include_usage": param.include_usage
            }
        return payload
    
    
    def parse_responses(self, server_responses: ServerResponses) -> RequestFuncOutput:
        request = server_responses.request
        chunk_list = server_responses.chunk_list
        output = RequestFuncOutput(
            idx = request.idx,
            start_time = server_responses.start_time,
            end_time = server_responses.end_time
        )
        usage = None
        
        try:
            for chunk in chunk_list:
                # print(f"chunk: {chunk}")
                json_res = json.loads(chunk.response)
                if server_responses.response_type == "stream":   # stream == True
                    if "choices" in json_res and len(json_res["choices"]) > 0:
                        choice = json_res["choices"][0]     # Only support n==1
                        if "delta" in choice:
                            delta = choice["delta"]
                            if delta.get("content", None):
                                output.generated_text += delta["content"]
                            else:
                                # compatible reasoning content for deepseek-r1、qwq
                                if delta.get("reasoning_content", None):
                                    output.reasoning_text += delta["reasoning_content"]
                    if "usage" in json_res:
                        usage = json_res["usage"]
                else:   # chat.completion , stream == False
                    choice = json_res["choices"][0]     # Only support n==1
                    output.generated_text = choice["message"]["content"]
                    if "reasoning_content" in choice["message"]:
                        output.reasoning_text = choice["message"]["reasoning_content"]
                    if "usage" in json_res:
                        usage = json_res["usage"]
                    break   #   len(responses)==1 when non stream
                        
            if usage:
                # Note: There will be additional tokens for some groups of chat templates.
                output.prompt_len = usage.get("prompt_tokens", 0)
                # Note: the completion_tokens maybe include the reasoning_content
                output.output_len = usage.get("completion_tokens", 0)
                if "completion_tokens_details" in usage:
                    output.reasoning_len = usage["completion_tokens_details"].get("reasoning_tokens", 0)

            # calculate the prompt length
            if output.prompt_len == 0 or output.prompt_len is None:
                if request.prompt_len:
                    output.prompt_len = request.prompt_len
                elif self.tokenizer:
                    if request.custom_body and "messages" in request.custom_body:
                        chat_template_prompt_tokens = self.tokenizer.apply_chat_template(request.custom_body["messages"])
                        output.prompt_len = len(chat_template_prompt_tokens)
                    elif request.prompt:
                        output.prompt_len = len(self.tokenizer.encode(request.prompt))
            
            # if no usage and user provided a tokenizer, use it to calculate the output and reasoning length, cover the usage
            if output.output_len == 0 or output.output_len is None:
                # calculate output_len
                if output.generated_text and self.tokenizer:
                    output.output_len = len(self.tokenizer.encode(output.generated_text))
                else:
                    output.output_len = 0
            
            if output.reasoning_len == 0 or output.reasoning_len is None:
                # calculate reasoning_len
                if output.reasoning_text and self.tokenizer:
                    output.reasoning_len = len(self.tokenizer.encode(output.reasoning_text))
                else:
                    output.reasoning_len = 0
            
            output.output_len = output.output_len + output.reasoning_len
            output.chunk_list = chunk_list
            output.success = True
            
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
            
        return output
    