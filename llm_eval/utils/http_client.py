import aiohttp
import logging
import requests
import json
import time
import sys
import traceback
import platform
if platform.system() != 'Windows':
    import resource
from dataclasses import dataclass
from typing import Dict, Optional
from llm_eval.apis.base_api import ApiBase, RequestFuncInput, ServerResponses, Chunk
import logging


@dataclass
class ServerSentEvent:
    """
        See more information about Server-Sent Events (SSE) in https://hpbn.co/server-sent-events-sse/
    """
    data: Optional[str] = None
    event: Optional[str] = None
    id: Optional[str] = None
    retry: Optional[str] = None
    
    
def stream_sse_decoder(line: str) -> ServerSentEvent:
    if not line:
        return None
    
    line = line.decode("utf-8")
    sse = ServerSentEvent()
    sse_type, _, sse_value = line.partition(":")

    if sse_type == "event":
        sse.event = sse_value
    elif sse_type == "data":
        sse.data = sse_value.strip()     # compatible with openai api prifix 'data: '
    elif sse_type == "id":
        sse.id = sse_value
    elif sse_type == "retry":
        sse.retry = sse_value
    else:
        return None

    return sse


async def on_request_start(session, trace_config_ctx, params):
    logging.debug(f'Starting request: <{params}>')


async def on_request_chunk_sent(session, trace_config_ctx, params):
    logging.debug(f'Request body: {params}')


async def on_response_chunk_received(session, trace_config_ctx, params):
    logging.debug(f'Response info: <{params}>')


def enhance_soft_limit(max_file_limit: int) -> None:
    try:
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft_limit < max_file_limit:
            new_soft_limit = min(max_file_limit, hard_limit)
            logging.info(f"Enhance soft_limit {soft_limit} -> {new_soft_limit}")
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_limit, hard_limit))
    except Exception as e:
        logging.error(f"Failed to set the soft limit: {e}")


class AioHttpClient:
    
    CONNECT_TIMEOOUT = 60 * 60  # for non-streaming requests, set connect timeout to 10 minutes
    TOTAL_TIMEOUT = 60 * 60
    MAX_FILE_DESCRIPTORS = 65535    # soft limit
    TCP_LIMIT = 65535
    READ_BUFSIZE = 2**24    # set read_bufsize to 16MB for 30K input/output
    
    def __init__(self, api: ApiBase, headers: Dict=None, debug: bool=False):
        self.api = api
        client_timeout = aiohttp.ClientTimeout(total=self.TOTAL_TIMEOUT, connect=self.CONNECT_TIMEOOUT)
        tcp_connector = aiohttp.TCPConnector(limit=self.TCP_LIMIT)
        
        self.debug = debug
        if debug:
            logging.setLevel(level=logging.DEBUG)
            # Note: See more aiohttp trace information in: https://docs.aiohttp.org/en/stable/tracing_reference.html#aiohttp-client-tracing-reference
            trace_config = aiohttp.TraceConfig()
            trace_config.on_request_start.append(on_request_start)
            trace_config.on_request_chunk_sent.append(on_request_chunk_sent)
            trace_config.on_response_chunk_received.append(on_response_chunk_received)
            self.trace_config = trace_config
            
        self.session = aiohttp.ClientSession(
            trace_configs=[trace_config] if debug else [],
            connector=tcp_connector,
            timeout=client_timeout,
            read_bufsize=self.READ_BUFSIZE
        )
        self.headers = {
            "user-agent": "llm-eval",
            "Content-Type": "application/json"
        }
        if headers:
            self.headers.update(headers)
        
        self.url = api.api_url
        
        if platform.system() != 'Windows':
            enhance_soft_limit(self.MAX_FILE_DESCRIPTORS)


    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.session.close()
        
    
    async def post(self, request_input: RequestFuncInput) -> ServerResponses:
        payload = self.api.build_request(request_input)
        server_responses = ServerResponses(
            request=request_input,
            start_time=time.perf_counter(),
            end_time=0,
            success=False,
            chunk_list=[],
            error_msg="",
            status_code=0
        )
        # print(f"payload: {payload}", flush=True)
        # print(f"headers: {self.headers}", flush=True)
        
        try:
            async with self.session.post(url=self.url, json=payload, headers=self.headers) as response:
                server_responses.status_code = response.status
                if response.status == 200 and "text/event-stream" in response.content_type:
                    server_responses.response_type = "stream"
                    async for line in response.content:
                        # print(line, flush=True)
                        if self.debug:
                            logging.debug(line)
                        sse = stream_sse_decoder(line)
                        if sse:
                            if sse.event and sse.event == "error":
                                server_responses.success = False
                                server_responses.error_msg = "Encountered event=error while handling Server-Sent Events (SSE)."
                                logging.error(f"async post Exception {server_responses.error_msg}")
                                break

                            if sse.data:
                                if sse.data.startswith("[DONE]"):  # openai api completed
                                    server_responses.end_time = time.perf_counter()
                                else:
                                    server_responses.chunk_list.append(Chunk(chunk_time=time.perf_counter(), response=sse.data))
                                    server_responses.end_time = time.perf_counter()
                    server_responses.success = True
                elif response.status == 200 and "application/json" in response.content_type:
                    server_responses.response_type = "application/json"
                    json_res = await response.json()
                    server_responses.chunk_list.append(Chunk(chunk_time=time.perf_counter(), response=json.dumps(json_res)))
                    server_responses.end_time = time.perf_counter()
                    if "object" in json_res and json_res["object"] == "error":
                        server_responses.success = False
                        server_responses.error_msg = f"code={json_res['code']}, message={json_res['message']}"
                        logging.error(f"async post Exception {server_responses.error_msg}")
                    else:
                        server_responses.success = True
                else:
                    server_responses.success = False
                    server_responses.error_msg = await response.text()
                    logging.error(f"async post Exception: response.status={response.status}, response.content_type={response.content_type}, response.hearders={response.headers}, response.text={server_responses.error_msg}")
                    
        except Exception:
            server_responses.success = False
            exc_info = sys.exc_info()
            server_responses.error_msg = "".join(traceback.format_exception(*exc_info))
            logging.error(f"async post Exception {server_responses.error_msg}")
        
        if server_responses.end_time == 0:
            server_responses.end_time = time.perf_counter()
            
        return server_responses


class HttpClient:
    
    CONNECT_TIMEOUT = 60 * 60
    TOTAL_TIMEOUT = 60 * 60
    MAX_FILE_DESCRIPTORS = 65535    # soft limit
    READ_BUFSIZE = 2**24    # set read_bufsize to 16MB for 30K input/output
    
    def __init__(self, api: ApiBase, headers: Dict=None, debug: bool=False):
        self.api = api
        self.debug = debug
        if debug:
            logging.setLevel(level=logging.DEBUG)
            
        self.session = requests.Session()
        self.headers = {
            "user-agent": "llm-eval",
            "Content-Type": "application/json"
        }
        if headers:
            self.headers.update(headers)
        
        self.url = api.api_url
        
        if platform.system() != 'Windows':
            enhance_soft_limit(self.MAX_FILE_DESCRIPTORS)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.session.close()
        
    
    def parse_event_stream(self, data: str) -> list:
        """Parse raw event stream data into list of events.
        
        Args:
            data: Raw event stream data
            
        Returns:
            list: Parsed events
        """
        events = []
        event_lines = data.strip().split('\n\n')
        for event_data in event_lines:
            event = event_data.strip().split(':', 1)[1]
            if '"status":"running"' in event:
                continue
            event = event.replace('false', 'False')
            event = event.replace('true', 'True')
            try:
                d = eval(event)
                events.append(d)
            except Exception as e:
                logging.warning(f"Failed to parse event: {event}, error: {str(e)}")
        return events

    def post_raw_stream(self, request_input: RequestFuncInput) -> ServerResponses:
        """Post request with raw stream processing, handling non-standard streaming responses.
        
        Args:
            request_input: Same as post method
            
        Returns:
            ServerResponses: Response object with parsed events
        """
        payload = self.api.build_request(request_input)
        server_responses = ServerResponses(
            request=request_input,
            start_time=time.perf_counter(),
            end_time=0,
            success=False,
            chunk_list=[],
            error_msg="",
            status_code=0
        )

        try:
            with self.session.post(
                url=self.url,
                json=payload,
                headers=self.headers,
                stream=True,
                timeout=(self.CONNECT_TIMEOUT, self.TOTAL_TIMEOUT)
            ) as response:
                server_responses.status_code = response.status_code
                
                if response.status_code == 200:
                    server_responses.response_type = "raw_stream"
                    result = None
                    for chunk in response.iter_content(chunk_size=1):
                        if result is None:
                            result = chunk
                        else:
                            result += chunk
                    
                    if result:
                        result = result.decode('utf-8')
                        events = self.parse_event_stream(result)
                        for event in events:
                            # 这里的chunk time就不可信了
                            server_responses.chunk_list.append(Chunk(chunk_time=time.perf_counter(), response=json.dumps(event)))
                        server_responses.end_time = time.perf_counter()
                        server_responses.success = True
                else:
                    server_responses.success = False
                    server_responses.error_msg = response.text
                    logging.error(f"post_raw_stream Exception {server_responses.error_msg}, headers={response.headers}")
                    
        except Exception:
            server_responses.success = False
            exc_info = sys.exc_info()
            server_responses.error_msg = "".join(traceback.format_exception(*exc_info))
            logging.error(f"post_raw_stream Exception {server_responses.error_msg}")
        
        if server_responses.end_time == 0:
            server_responses.end_time = time.perf_counter()
            
        return server_responses

    def post(self, request_input: RequestFuncInput) -> ServerResponses:
        payload = self.api.build_request(request_input)
        server_responses = ServerResponses(
            request=request_input,
            start_time=time.perf_counter(),
            end_time=0,
            success=False,
            chunk_list=[],
            error_msg="",
            status_code=0
        )

        try:
            with self.session.post(
                url=self.url,
                json=payload,
                headers=self.headers,
                stream=True,
                timeout=(self.CONNECT_TIMEOUT, self.TOTAL_TIMEOUT)
            ) as response:
                server_responses.status_code = response.status_code
                content_type = response.headers.get('content-type', '')
                
                if response.status_code == 200 and "text/event-stream" in content_type:
                    server_responses.response_type = "stream"
                    for line in response.iter_lines():
                        if line:
                            if self.debug:
                                logging.debug(line)
                            sse = stream_sse_decoder(line)
                            if sse:
                                if sse.event and sse.event == "error":
                                    server_responses.success = False
                                    server_responses.error_msg = "Encountered event=error while handling Server-Sent Events (SSE)."
                                    logging.error(f"post Exception {server_responses.error_msg}")
                                    break

                                if sse.data:
                                    if sse.data.startswith("[DONE]"):  # openai api completed
                                        server_responses.end_time = time.perf_counter()
                                    else:
                                        server_responses.chunk_list.append(Chunk(chunk_time=time.perf_counter(), response=sse.data))
                                        server_responses.end_time = time.perf_counter()
                    server_responses.success = True
                elif response.status_code == 200 and "application/json" in content_type:
                    server_responses.response_type = "application/json"
                    json_res = response.json()
                    server_responses.chunk_list.append(Chunk(chunk_time=time.perf_counter(), response=json.dumps(json_res)))
                    server_responses.end_time = time.perf_counter()
                    if "object" in json_res and json_res["object"] == "error":
                        server_responses.success = False
                        server_responses.error_msg = f"code={json_res['code']}, message={json_res['message']}"
                        logging.error(f"post Exception {server_responses.error_msg}")
                    else:
                        server_responses.success = True
                else:
                    server_responses.success = False
                    server_responses.error_msg = response.text
                    logging.error(f"post Exception {server_responses.error_msg}, headers={response.headers}")
                    
        except Exception:
            server_responses.success = False
            exc_info = sys.exc_info()
            server_responses.error_msg = "".join(traceback.format_exception(*exc_info))
            logging.error(f"post Exception {server_responses.error_msg}")
        
        if server_responses.end_time == 0:
            server_responses.end_time = time.perf_counter()
            
        return server_responses
    