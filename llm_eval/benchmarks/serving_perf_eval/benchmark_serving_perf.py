import asyncio
import logging
import time
import re
from enum import Enum
import numpy as np
from typing import Optional, Tuple, List, Dict
from tqdm.asyncio import tqdm
from dataclasses import asdict
from llm_eval.utils.registry import dataset_registry, api_registry
from llm_eval.utils.http_client import AioHttpClient
from llm_eval.apis.base_api import ApiBase
from llm_eval.metrics.serving_perf_eval_metrics import InferenceServingPerformanceTags, InferencePerformanceProcess, InferenceServingPerformanceSummary
from llm_eval.utils.report_task import ReportTask
from llm_eval.utils.wandb_task import WandbTask
from llm_eval.data_store.client import TestType
from .save_perf_data import save_result_to_excel, save_process_data_to_json
from llm_eval.datasets.local import CustomBodyDataset
import wandb

_response_process_completed = False

class Backend(Enum):
    OPENAI = "openai"
    VLLM = "vllm"
    MAAS = "maas"

class PerfTestType(Enum):
    SINGLE = "single"
    MULTIPLE = "multiple"
    

async def request_filler_worker(
    cfg, 
    api: ApiBase,
    request_queue: asyncio.Queue,
) -> Tuple[int, int, int]:
    total_count = 0

    if cfg.benchmark.dataset_name:     # Use specificed dataset
        # Initialize dataset
        dataset_class = dataset_registry.get_class(cfg.benchmark.dataset_name)
        if cfg.benchmark.dataset_name == "random":
            dataset = dataset_class(
                tokenizer=api.tokenizer,
                num_requests=cfg.benchmark.num_prompts,
                prefix_len=cfg.benchmark.random_prefix_len,
                range_ratio=cfg.benchmark.random_range_ratio,
                input_len=cfg.benchmark.random_input_len,
                output_len=cfg.benchmark.random_output_len,
            )
        else:
            if cfg.benchmark.dataset_path:
                dataset = dataset_class(path=cfg.benchmark.dataset_path)
            else:
                dataset = dataset_class()
        logging.info(f"Initialize dataset successed. {dataset.dataset}")
        # Parse dataset to request
        if isinstance(dataset, CustomBodyDataset):
            # Custom dataset, the whole body will be replaced
            logging.info("Using CustomBodyDataset, the whole body will be replaced. And the args you set in extra_args will not be effective.")
            for req_data in dataset.req_generator(num_requests=cfg.benchmark.num_prompts):
                request_input = api.create_request_input(**cfg.benchmark, **cfg.extra_args)
                request_input.idx = req_data["idx"]
                request_input.custom_body = req_data["body"]
                await request_queue.put(request_input)
                total_count += 1
        else:
            # Local dataset or other datasets
            for perf_data in dataset.perf_req_generator(num_requests=cfg.benchmark.num_prompts, tokenizer=api.tokenizer):
                request_input = api.create_request_input(**cfg.benchmark, **cfg.extra_args)
                request_input.idx = perf_data.idx
                request_input.prompt = perf_data.prompt
                request_input.prompt_len = perf_data.prompt_len
                for attr in ["max_completion_tokens", "max_tokens"]:
                    if hasattr(request_input, attr):
                        if cfg.extra_args.get("max_completion_tokens", None):
                            setattr(request_input, attr, cfg.extra_args.max_completion_tokens)
                        else:
                            setattr(request_input, attr, perf_data.output_len)
                await request_queue.put(request_input)
                total_count += 1
    else:   # Use same prompt
        prompt = cfg.benchmark.prompt if cfg.benchmark.prompt else "The largest animal in the world is"
        count = 0
        for _ in range(cfg.benchmark.num_prompts):
            request_input = api.create_request_input(**cfg.benchmark, **cfg.extra_args)
            request_input.prompt = prompt
            request_input.idx = count
            count += 1
            for attr in ["max_completion_tokens", "max_tokens"]:
                if hasattr(request_input, attr) and cfg.extra_args.get("max_completion_tokens", None):
                    setattr(request_input, attr, cfg.extra_args.max_completion_tokens)
            await request_queue.put(request_input)
            total_count += 1
    
    return total_count


async def http_client_worker(
    cfg,
    api: ApiBase,
    request_queue: asyncio.Queue, 
    response_queue: asyncio.Queue, 
    pbar: Optional[tqdm] = None
) -> None:
    headers = {}
    if cfg.benchmark.get("api_key", None):
        headers["Authorization"] = f"Bearer {cfg.benchmark.api_key}"
    if cfg.benchmark.get("extra_headers", None):
        headers.update(cfg.benchmark.extra_headers)
    
    # This can be used once the minimum Python version is 3.10 or higher
    semaphore = asyncio.Semaphore(cfg.benchmark.concurrency) if cfg.benchmark.get("limit_max_concurrency", False) else None

    # Initialize client
    client = AioHttpClient(api=api, headers=headers, debug=False)
    sended_request_tasks = []
    
    async with client:
        while True:     # Getting request from queue
            try:
                request = request_queue.get_nowait()
                request_queue.task_done()
                
                async def send_request_task(request):
                    if semaphore is None:
                        # No semaphore, send request directly
                        server_responses = await client.post(request)
                    else:
                        async with semaphore:
                            server_responses = await client.post(request)
                    if pbar:
                        pbar.update(1)
                    # putting response in the queue
                    await response_queue.put(server_responses)

                sending_request_task = asyncio.create_task(send_request_task(request))
                sended_request_tasks.append(sending_request_task)

                # Control the request rate
                # Note: If the request rate is llmty, then we don't need to wait.
                if cfg.benchmark.request_rate in [float("inf"), "inf"]:
                    continue
                # Note: If the request rate is -1, them we send request one by one.
                elif cfg.benchmark.request_rate == -1:
                    await sending_request_task
                # Note: Sample the request interval from the exponential distribution.
                else:
                    interval = np.random.exponential(1.0 / cfg.benchmark.request_rate)
                    await asyncio.sleep(interval)
                    
            except asyncio.CancelledError:
                logging.info("Consumer cancelled.")
                break
            except asyncio.QueueEmpty:
                logging.debug("Request queue empty.")
                await asyncio.gather(*sended_request_tasks, return_exceptions=True)
                break
            except Exception as e:
                logging.error(f"An error occurred: {e}")
    

async def metrics_collector_worker(
    cfg,
    api: ApiBase,
    response_queue: asyncio.Queue,
    wandb_task: WandbTask,
    metrics_tag: InferenceServingPerformanceTags = None,
    pbar: Optional[tqdm] = None
) -> Tuple[InferenceServingPerformanceSummary, Dict]:
    
    total_response = 0
    prompt_input_lens: List[int] = []
    actual_output_lens: List[int] = []
    actual_reasoning_lens: List[int] = []
    completed = 0
    audit_hit_num = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    e2els: List[float] = []
    process_metrics_list = []
    generated_texts = []
    reasoning_texts = []
    status_codes = []
    error_msgs = []
    result_json = {}
    
    while True:
        # collect response
        try:
            server_response = response_queue.get_nowait()
            # print(f"server_response: {server_response}")
            response_queue.task_done()
            total_response += 1
            if pbar:
                pbar.update(1)
        except asyncio.QueueEmpty as e:
            # print(f"Response queue empty, total_response={total_response}")
            if total_response >= cfg.benchmark.num_prompts or _response_process_completed:
                break
            await asyncio.sleep(0.1)
            continue
        
        try:
            request_id = server_response.request.idx if server_response.request.idx else 0
            status_codes.append((request_id, server_response.status_code))

            if server_response.success:
                # Parse output
                parse_output = api.parse_responses(server_responses=server_response)
                
                if not parse_output.success:
                    logging.error(f"Parse response failed, error_msg={parse_output.error}")
                    error_msgs.append((request_id, parse_output.error))
                    continue
                
                # Security Audit Hit Status
                if parse_output.audit_hit:
                    audit_hit_num += 1
                    logging.error(parse_output.generated_text)
                    error_msgs.append((request_id, "Security Audit Hit Status"))
                    continue
                
                if parse_output.output_len > 0:
                    completed += 1
                else:
                    logging.error(f"Generated text is empty, Response={server_response}")
                    error_msgs.append((request_id, "Generated text is empty"))
                    continue
                    
                # Statistics
                prompt_input_lens.append(parse_output.prompt_len)
                actual_output_lens.append(parse_output.output_len)
                actual_reasoning_lens.append(parse_output.reasoning_len)
                latency = parse_output.end_time - parse_output.start_time
                ttft, tpot = 0, 0
                if len(parse_output.chunk_list) > 0:
                    ttft = parse_output.chunk_list[0].chunk_time - parse_output.start_time
                    if parse_output.output_len > 1:
                        tpot = (latency - ttft) / (parse_output.output_len - 1)
                    for earlier, later in zip(parse_output.chunk_list[:-1], parse_output.chunk_list[1:]):
                        delay = later.chunk_time - earlier.chunk_time
                        itls.append(delay)
                ttfts.append(ttft)
                tpots.append(tpot)
                e2els.append(latency)
                
                process_metrics = InferencePerformanceProcess(
                    input_length=parse_output.prompt_len,
                    output_length=parse_output.output_len,
                    reasoning_length=parse_output.reasoning_len,
                    latency=latency,
                    ttft=ttft,
                    tpot=tpot
                    # start_time=parse_output.start_time,
                    # end_time=parse_output.end_time
                    # extra_body=server_response.request.custom_body.get("extra_body", {}) if server_response.request.custom_body else {}
                )
                process_metrics_list.append((request_id, process_metrics))
                generated_texts.append((request_id, parse_output.generated_text))
                reasoning_texts.append((request_id, parse_output.reasoning_text))
                
            else:
                logging.error(f"Getting error response, status_code={server_response.status_code}, error_msg={server_response.error_msg}")  
                error_msgs.append((request_id, server_response.error_msg))
                
        except Exception as e:
            logging.error(f"An error occurred while processing response: {e}")
            error_msgs.append((request_id, str(e)))
    
    
    # Summary
    summary_result = InferenceServingPerformanceSummary(
        duration = 0,     # calculate later
        completed = completed,
        audit_hit = audit_hit_num,
        total_input = sum(prompt_input_lens),
        total_output = sum(actual_output_lens),
        total_resoning_tokens = sum(actual_reasoning_lens),
        request_throughput = 0,    # calculate later
        input_throughput = 0,  # calculate later
        output_throughput = 0, # calculate later
        mean_ttft_ms = np.mean(ttfts or 0) * 1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms = np.median(ttfts or 0) * 1000,
        p99_ttft_ms = np.percentile(ttfts or 0, 99) * 1000,
        p1_ttft_ms = np.percentile(ttfts or 0, 1) * 1000,
        p90_ttft_ms = np.percentile(ttfts or 0, 90) * 1000,
        p10_ttft_ms = np.percentile(ttfts or 0, 10) * 1000,
        mean_tpot_ms = np.mean(tpots or 0) * 1000,
        median_tpot_ms = np.median(tpots or 0) * 1000,
        p99_tpot_ms = np.percentile(tpots or 0, 99) * 1000,
        p1_tpot_ms = np.percentile(tpots or 0, 1) * 1000,
        p90_tpot_ms = np.percentile(tpots or 0, 90) * 1000,
        p10_tpot_ms = np.percentile(tpots or 0, 10) * 1000,
        mean_itl_ms = np.mean(itls or 0) * 1000,
        median_itl_ms = np.median(itls or 0) * 1000,
        p99_itl_ms = np.percentile(itls or 0, 99) * 1000,
        p1_itl_ms = np.percentile(itls or 0, 1) * 1000,
        p90_itl_ms = np.percentile(itls or 0, 90) * 1000,
        p10_itl_ms = np.percentile(itls or 0, 10) * 1000,
        mean_e2el_ms = np.mean(e2els or 0) * 1000,
        median_e2el_ms = np.median(e2els or 0) * 1000,
        p99_e2el_ms = np.percentile(e2els or 0, 99) * 1000,
        p1_e2el_ms = np.percentile(e2els or 0, 1) * 1000,
        p90_e2el_ms = np.percentile(e2els or 0, 90) * 1000,
        p10_e2el_ms = np.percentile(e2els or 0, 10) * 1000,
        prompt_input_lens = prompt_input_lens,
        actual_output_lens = actual_output_lens,
        actual_reasoning_lens = actual_reasoning_lens
    )
    
    
    wandb_writer = wandb_task.get_wandb_writer()
    if wandb_writer:
        # Log process metrics
        for process_metrics in process_metrics_list:
            metric = {**asdict(process_metrics[1]), "concurrency": metrics_tag.concurrency}
            wandb_writer.log(metric)    # (idx, process_metrics)

        # Log dataset summary
        table = wandb.Table(
            columns=["Scenario", "Percentile", "Prompt Input", "Actual Output", "Reasoning"],
            log_mode="MUTABLE"
        )
        # Log length distributions as single table with optimized styling
        percentiles = [0, 25, 50, 75, 100]
        prompt_input_percentiles = np.percentile(summary_result.prompt_input_lens or [0], percentiles)
        actual_output_percentiles = np.percentile(summary_result.actual_output_lens or [0], percentiles)
        actual_reasoning_percentiles = np.percentile(summary_result.actual_reasoning_lens or [0], percentiles)
        
        prompt_input_avg = np.mean(summary_result.prompt_input_lens or [0])
        actual_output_avg = np.mean(summary_result.actual_output_lens or [0])
        actual_reasoning_avg = np.mean(summary_result.actual_reasoning_lens or [0])
        
        scenario_name = f"{metrics_tag.dataset}|Req:{metrics_tag.request_num}|Rate:{metrics_tag.request_rate}|Con:{metrics_tag.concurrency}"
        
        for p, pi, ao, ar in zip(percentiles, prompt_input_percentiles, actual_output_percentiles, actual_reasoning_percentiles):
            table.add_data(scenario_name, f"{p}%", pi, ao, ar)
            
        table.add_data(scenario_name, "Avg", prompt_input_avg, actual_output_avg, actual_reasoning_avg)
        wandb_writer.log({"Lengths Distribution": table})
            
    # sort result msg before save result_json, so you can use result_json check details
    generated_texts.sort(key=lambda x: x[0])
    reasoning_texts.sort(key=lambda x: x[0])
    # process_metrics_list.sort(key=lambda x: x[1].end_time)    # Sort by request end time to determine request priority
    process_metrics_list.sort(key=lambda x: x[0])
    status_codes.sort(key=lambda x: x[0])
    error_msgs.sort(key=lambda x: x[0])

    result_json = {
        "total_response": total_response,
        "process_metrics": process_metrics_list,
        "generated_texts": generated_texts,
        "reasoning_texts": reasoning_texts,
        "status_codes": status_codes,
        "error_msgs": error_msgs
    }
    
    return summary_result, result_json


async def warm_up(
    cfg,
    api: ApiBase,
    warm_up_times: int
) -> None:
    headers = {}
    if cfg.benchmark.get("api_key", None):
        headers["Authorization"] = f"Bearer {cfg.benchmark.api_key}"
    if cfg.benchmark.get("extra_headers", None):
        headers.update(cfg.benchmark.extra_headers)
    headers["Connection"] = "close"

    # Initialize client
    client = AioHttpClient(api=api, headers=headers, debug=False)
    request_input = api.create_request_input(**cfg.benchmark, **cfg.extra_args)
    request_input.prompt = "The biggest animal in the world is"
    for attr in ["max_completion_tokens", "max_tokens"]:
        if hasattr(request_input, attr):
            setattr(request_input, attr, 128)   # prevent freezing
    pbar = tqdm(total=warm_up_times, desc="Warm Up")
    async with client:
        for idx in range(warm_up_times):
            request_input.idx = idx
            response = await client.post(request_input)
            assert response.success, f"Ping {client.url} failed, response={response}."
            pbar.update(1)
    pbar.close


async def benchmark(
    cfg,
    report_task: ReportTask,
    wandb_task: WandbTask
) -> Tuple[InferenceServingPerformanceTags, InferenceServingPerformanceSummary]:
    # Initialize task queue
    request_queue = asyncio.Queue()     # Save all requests that will be sent.
    response_queue = asyncio.Queue()    # Save all benchmark response data.
    http_client_tasks = []
    # Initialize api
    api_class = api_registry.get_class(cfg.benchmark.backend)
    api = api_class(cfg.benchmark.url, cfg.benchmark.tokenizer)
                
    # Initialize metrics tag
    metrics_tag = InferenceServingPerformanceTags(
        request_num=cfg.benchmark.num_prompts,
        request_rate=str(cfg.benchmark.request_rate),
        dataset=cfg.benchmark.dataset_name,
        concurrency=cfg.benchmark.concurrency
    )
    
    # Check Local Dataset
    if cfg.benchmark.dataset_name and cfg.benchmark.dataset_name in ["local", "custom"]:
        assert cfg.benchmark.dataset_path, "Try to use local dataset, but dataset_path is null."
        local_dataset_class = dataset_registry.get_class("local")
        local_dataset_file_name = local_dataset_class.check_local_dataset(cfg.benchmark.dataset_path)
        # Fix metrics_tag
        metrics_tag.dataset = local_dataset_file_name
    
    # Warm up, use default datasets
    await warm_up(cfg, api, 1)
    
    print("Preparing request queue...")
    request_filler_task = asyncio.create_task(request_filler_worker(cfg, api, request_queue))
    total_request_num  = await request_filler_task   # Waiting for all requests fill in queue
    print(f"Requests in the request queue: {total_request_num}")
    
    pbar_http_client = tqdm(total=total_request_num, desc="HTTP Handle Client")
    pbar_metrics_collector = tqdm(total=total_request_num, desc="Metrics Statistics")
    
    print(f"Metrics worker start, listening to the response...")
    metrics_collector_task = asyncio.create_task(
        metrics_collector_worker(
            cfg, 
            api=api, 
            response_queue=response_queue, 
            wandb_task=wandb_task,
            metrics_tag=metrics_tag,
            pbar=pbar_metrics_collector
        )
    )
    
    print(f"Benchmark worker start, user concurrency: {cfg.benchmark.concurrency}, traffic request rate: {cfg.benchmark.request_rate}, limit max concurrency: {cfg.benchmark.get('limit_max_concurrency', False)}")
    if cfg.benchmark.get("limit_max_concurrency", False):
        http_client_worker_num = 1  # Use a single worker to handle all requests, using asyncio.Semaphore to limit the max concurrency.
        logging.info("Using limit max concurrency mode, only one worker will be used to handle all requests.")
    else:
        http_client_worker_num = cfg.benchmark.concurrency
    
    for idx in range(http_client_worker_num):
        logging.info(f"Start concurrency {idx} worker.")
        task = asyncio.create_task(http_client_worker(
            cfg,
            api=api,
            request_queue=request_queue, 
            response_queue=response_queue, 
            pbar=pbar_http_client
        ))
        http_client_tasks.append(task)
    
    benchmark_start_time = time.perf_counter()
    
    await request_queue.join()      # Waiting for all requests have been send
    await asyncio.gather(*http_client_tasks, return_exceptions=True)    # Waiting for all requests have been handle

    pbar_http_client.close()
    benchmark_duration = time.perf_counter() - benchmark_start_time
    
    await response_queue.join()         # Waiting for all responses have been collect
    global _response_process_completed
    _response_process_completed = True  # To prevent some request responses from missing and not being added to the queue
    summary_result, result_json = await metrics_collector_task   # Waiting for all metrics have been statistic
    pbar_metrics_collector.close()
    _response_process_completed = False     # For multiple test
    print(f"Benchmark completed in {benchmark_duration}s. Waiting for all metrics have been statistic...")
    
    summary_result.duration = benchmark_duration
    summary_result.request_throughput = summary_result.completed / benchmark_duration
    summary_result.input_throughput = summary_result.total_input / benchmark_duration
    summary_result.output_throughput = summary_result.total_output / benchmark_duration
    summary_result.pretty_print()
    
    # send metrics
    report_task.produce_statistic_data(summary_result, metrics_tag)
    
    await asyncio.sleep(0.25)
    
    # save process data for debug
    if cfg.benchmark.save_data:
        result_json = {**cfg.benchmark, **result_json, **asdict(summary_result)}
        file_path = save_process_data_to_json(cfg=cfg, result_json=result_json)
        print(f"Save process data to: {file_path}")
    return metrics_tag, summary_result


def run_benchmark(cfg):
    # Check config
    assert cfg.benchmark.benchmark_name == "serving_perf_eval", "Illegal config."
    if cfg.benchmark.dataset_name in ["random", "math-500", "sharegpt", "aime-2024", "lp-resume", "lenovo-mmlu"]:
        assert cfg.benchmark.tokenizer, "You should provide a tokenizer when using random dataset."
    
    # Metrics repost task
    report_task = ReportTask(cfg, TestType.SERVING_INFERENCE_PERFORMANCE)
    
    wandb_task = WandbTask(
        cfg, 
        wandb_exp_name=re.sub(r'[^\w.-]+', '_', '_'.join(str(v) for v in cfg.test_meta.values() if v)) or 'unknown', 
        wandb_project=cfg.benchmark.benchmark_name
    )
    
    result_list = []
    # Benchmark
    if cfg.benchmark.get("perf_test_type", None) == PerfTestType.MULTIPLE.value:
        assert cfg.benchmark.request_rate_list, "You should support the request_rate_list like [-1, 1, 2, 4, 8, inf]."
        assert cfg.benchmark.concurrency_list, "You should support the concurrency_list like [1, 2, 4, 8, 16]."
        for concurrency in cfg.benchmark.concurrency_list:
            for request_rate in cfg.benchmark.request_rate_list:
                cfg.benchmark.concurrency = concurrency
                cfg.benchmark.request_rate = request_rate
                metrics_tag, statistic_result = asyncio.run(benchmark(cfg, report_task, wandb_task))
                result_list.append((metrics_tag, statistic_result))
    else:
        assert cfg.benchmark.request_rate, "You should provide a request_rate for single test."
        assert cfg.benchmark.concurrency, "You should provide a concurrency for single test."
        metrics_tag, statistic_result = asyncio.run(benchmark(cfg, report_task, wandb_task))
        result_list.append((metrics_tag, statistic_result))
    
    print("All finished.")

    file_path = save_result_to_excel(cfg=cfg, result_list=result_list, wandb_task=wandb_task)
    print(f"Save results to: {file_path}")
    
    wandb_task.finish_wandb_writer()