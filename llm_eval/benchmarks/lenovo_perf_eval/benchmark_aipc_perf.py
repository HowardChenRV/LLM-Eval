import requests
import logging
from dataclasses import dataclass, asdict
from typing import Optional
from tqdm import tqdm
from llm_eval.utils.registry import dataset_registry, api_registry
from llm_eval.utils.http_client import HttpClient, ApiBase
from llm_eval.utils.report_task import ReportTask
from llm_eval.utils.wandb_task import WandbTask
from data_store.client import TestType
from llm_eval.metrics.serving_perf_eval_metrics import InferencePerformanceProcess
from .process_lenovo_perf_results import process_results

@dataclass
class LenovoPerfEvalResult:
    success: bool = False
    error_msg: str = ""
    model: str = ""
    prompt: str = ""
    generated_text: str = ""
    process_metrics: Optional["InferencePerformanceProcess"] = None
    prefill_throughput: float = 0.0
    generation_throughput: float = 0.0
    

def load_model_to_gpu(host:str, port:int) -> bool:
    url=f"http://{host}:{port}/app/v1/model/llm/loadTogpu"
    headers = {
        'Accept': 'text/event-stream',
        'Content-Type': 'application/json',
    }
    response = requests.patch(url, headers=headers)
    if response.status_code == 200:
        logging.info(f"Model loaded to GPU successfully.")
        return True
    else:
        logging.error(f"Failed to load model to GPU. Status code: {response.status_code}")
        return False


def unload_model_from_gpu(host:str, port:int) -> bool:
    url=f"http://{host}:{port}/app/v1/model/llm/unloadModel"
    headers = {
        'Accept': 'text/event-stream',
        'Content-Type': 'application/json',
    }
    response = requests.patch(url, headers=headers)
    if response.status_code == 200:
        logging.info(f"Model unloaded from GPU successfully.")
        return True
    else:
        logging.error(f"Failed to unload model from GPU. Status code: {response.status_code}")
        return False


def warm_up_model(api: ApiBase, client: HttpClient, warmup_times: int):
    for i in range(warmup_times):
        warmup_input = api.create_request_input(**{
            "prompt": "What is the capital of China?",
            "stream": True,
            "n_ctx": 512,
            "n_predict": 128
        })
        warmup_input.idx = i
        server_responses = client.post(warmup_input)
        if server_responses.status_code != 200:
            raise RuntimeError(f"Warmup failed with status {server_responses.status_code}")


def lenovo_server_generate(
    api: ApiBase,
    client: HttpClient,
    model,
    req_data,
    wandb_task: WandbTask,
    pbar: Optional[tqdm] = None
) -> LenovoPerfEvalResult:
    # init wandb
    wandb_writer = wandb_task.get_wandb_writer()
    
    # fit chat template
    formatted_text = api.tokenizer.apply_chat_template(
        req_data["messages"],
        tokenize=False,
        add_generation_prompt=True
    )
    # Request parameter
    request_input = api.create_request_input(**{
        "prompt": formatted_text,
        "stream": True,
        "n_ctx": 4096,
        "n_predict": 1024,
        "seed": 10000,
        "temp": 0,
        "top_k":40,
        "top_p":0.9,
        # "repeat_penalty":1.1, 
        "repeat_penalty":1.0    # disable, no repeat penalty
    })
    request_input.idx = req_data["idx"]

    result = LenovoPerfEvalResult(
        model=model,
        prompt=formatted_text
    )

    try:
        server_responses = client.post_raw_stream(request_input)
        if server_responses.status_code == 200:
            output = api.parse_responses(server_responses)
            if output.success:
                # logging.info(f"[idx {request_input.idx}] Generated text: {output.generated_text}")
                # logging.info(f"[idx {request_input.idx}] Prompt length: {output.prompt_len}, Output length: {output.output_len}")
                timings = output.extra["timings"]
                ttft = timings["prompt_ms"]
                prefill_throughput = timings["prompt_per_second"]
                if output.output_len < 2:   # no decode
                    tpot = 0
                    generation_throughput = 0           
                else:
                    tpot = timings["predicted_per_token_ms"]
                    generation_throughput = timings["predicted_per_second"]
                
                result.success = True
                result.generated_text = output.generated_text
                result.process_metrics = InferencePerformanceProcess(
                    input_length=output.prompt_len,
                    output_length=output.output_len,
                    reasoning_length=output.reasoning_len,
                    latency=output.end_time - output.start_time,
                    ttft=ttft,
                    tpot=tpot
                    # start_time=output.start_time,
                    # end_time=output.end_time,
                )
                result.prefill_throughput = prefill_throughput
                result.generation_throughput = generation_throughput
                
                if wandb_writer:
                    metrics_dict = asdict(result.process_metrics)
                    metrics_dict["prefill_throughput"] = result.prefill_throughput
                    metrics_dict["generation_throughput"] = result.generation_throughput
                    wandb_writer.log(metrics_dict)
            else:
                result.success = False
                result.error_msg = f"[idx {request_input.idx}] Failed to generate text: {output.error}"
        else:
            result.success = False
            result.error_msg = f"[idx {request_input.idx}] Failed to generate text. Status code: {server_responses.status_code}"
    except Exception as e:
        result.success = False
        result.error_msg = f"[idx {request_input.idx}] Exception occurred: {str(e)}"

    if pbar:
        pbar.update(1)
         
    return result


def run_benchmark(cfg):
    # init config
    assert cfg.benchmark.benchmark_name == "lenovo_perf_eval", "Illegal config."
    host = "localhost"
    port = cfg.benchmark.port
    model = cfg.benchmark.model
    num_requests = 5
    model_loaded = False
    
    try:
        # Handle output prefix logic
        if hasattr(cfg.benchmark, "output_prefix") and cfg.benchmark.output_prefix:
            output_prefix = f"{cfg.benchmark.output_prefix}"
        elif hasattr(cfg, "test_meta"):
            # Build prefix from test_meta fields
            parts = []
            for field in ["gpu_model", "model", "quantization_method", "spectype"]:
                if hasattr(cfg.test_meta, field) and getattr(cfg.test_meta, field):
                    parts.append(getattr(cfg.test_meta, field))
            
            if parts:
                # Clean invalid filename characters
                import re
                prefix = "-".join(parts)
                prefix = re.sub(r'[\\/*?:"<>| ]', "_", prefix)
                output_prefix = f"lenovo_perf_eval-{prefix}"
            else:
                # Fallback to model path if no valid fields
                model_name = cfg.benchmark.model.split("/")[-1].split("\\")[-1]
                output_prefix = f"lenovo_perf_eval-{model_name}"
        else:
            # Extract last part of model path
            model_name = cfg.benchmark.model.split("/")[-1].split("\\")[-1]
            output_prefix = f"lenovo_perf_eval-{model_name}"
        
        # init report task  
        report_task = ReportTask(cfg, TestType.LENOVO_DEMO_PERFORMANCE)
        # init wandb
        wandb_task = WandbTask(
            cfg,
            wandb_exp_name=output_prefix,
            wandb_project=cfg.benchmark.benchmark_name
        )

        # load model to GPU
        if not load_model_to_gpu(host, port):
            raise RuntimeError("Failed to load model to GPU")
        model_loaded = True
        
        # init dataset and api
        dataset_class = dataset_registry.get_class("lenovo_mmlu")
        dataset = dataset_class()
        api_class = api_registry.get_class("lenovo")
        api = api_class(f"http://{host}:{port}/app/v1/infer/llm/chunked", model)
        headers = {
            'Accept': 'text/event-stream',
            'Content-Type': 'application/json',
        }
        with HttpClient(api=api, headers=headers, debug=False) as client:
            # Warmup with fixed prompt
            logging.info("Warmup with fixed prompt...")
            try:
                warm_up_model(api, client, cfg.benchmark.warmup_times)
            except Exception as e:
                logging.error(f"Warmup failed: {str(e)}")
                raise
            logging.info("Warmup completed.")

            result_list = []
            # generate with repetition
            pbar = tqdm(total=cfg.benchmark.repetition*num_requests, desc="Lenovo Perf Eval", unit="request")
            try:
                for req_data in dataset.req_generator(num_requests=num_requests):
                    for _ in range(cfg.benchmark.repetition):
                        result = lenovo_server_generate(api, client, model, req_data, wandb_task, pbar)
                        result_list.append(result)
                        if not result.success:
                            logging.error(f"Error in request {req_data['idx']}: {result.error_msg}")
            finally:
                pbar.close()
        
        # process results
        benchmark_summary = process_results(result_list, output_prefix)
        
        # upload performance stats to report task
        report_task.produce_statistic_data(benchmark_summary)
        
        try:
            # Upload to WandB if enabled
            if wandb_task and wandb_task.get_wandb_writer():
                wandb_writer = wandb_task.get_wandb_writer()
                # Upload performance stats to WandB summary
                for key, value in benchmark_summary.items():
                    wandb_writer.summary[key] = value
                # upload test_meta to wandb summary if exists
                for key, value in cfg.test_meta.items():
                    wandb_task.get_wandb_writer().summary[key] = value
        except Exception as e:
            logging.error(f"Failed to upload performance stats to WandB: {str(e)}")
        
        wandb_task.finish_wandb_writer()
        
    except Exception as e:
        logging.error(f"Benchmark failed with error: {str(e)}")
        raise
    finally:
        # Ensure model is unloaded from GPU in all cases
        if model_loaded:
            if not unload_model_from_gpu(host, port):
                logging.error("Failed to unload model from GPU")
    
    