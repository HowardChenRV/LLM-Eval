import os
import json
import wandb
import pandas as pd
import numpy as np
from enum import Enum
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict
from llm_eval.utils.wandb_task import WandbTask
from llm_eval.metrics.serving_perf_eval_metrics import InferenceServingPerformanceTags, InferenceServingPerformanceSummary

USER_HOME = os.path.expanduser("~")
PERF_DATA_FOLDER = os.path.join(USER_HOME, '.cache/llm_eval/serving_perf_eval_result/')

def save_result_to_excel(
    cfg,
    result_list: List[Tuple[InferenceServingPerformanceTags, InferenceServingPerformanceSummary]],
    wandb_task: WandbTask
) -> str:
    benchmark = cfg.benchmark
    test_meta = cfg.test_meta
    dynamic_cols = [
        "Hardware Spec", "Model", "Software Framework", "Cluster Spec", "Dataset", "Total Requests",
        "Request Rate", "Request Success Rate(%)", "Audit Hits", "Concurrency", "Total Input Tokens", "Total Output Tokens", "Total Reasoning Tokens", "Total Duration(s)",
        "Total Request Throughput(requests/s)", "Total Input Throughput(tokens/s)", "Total Output Throughput(tokens/s)",
        "TTFT_avg(ms)", "TPOT_avg(ms)", "ITL_avg(ms)", "E2EL_avg(ms)",
        "TTFT_p50(ms)", "TPOT_p50(ms)", "ITL_p50(ms)", "E2EL_p50(ms)",
        "TTFT_p90(ms)", "TPOT_p90(ms)", "ITL_p90(ms)", "E2EL_p90(ms)",
        "TTFT_p10(ms)", "TPOT_p10(ms)", "ITL_p10(ms)", "E2EL_p10(ms)",
        "TTFT_p99(ms)", "TPOT_p99(ms)", "ITL_p99(ms)", "E2EL_p99(ms)",
        "TTFT_p1(ms)" , "TPOT_p1(ms)" , "ITL_p1(ms)" , "E2EL_p1(ms)" ,
    ]
    result_pd = pd.DataFrame(columns=dynamic_cols)
    for tags, metrics in result_list:
        pd_data = dict()
        pd_data["Hardware Spec"] = f"{getattr(test_meta, 'hardware', 'unknow')} x {getattr(test_meta, 'hardware_num', 'unknow')}"
        pd_data["Model"] = f"{getattr(test_meta, 'model', 'unknow')} - {getattr(test_meta, 'quantization_method', 'unknow')}"
        pd_data["Software Framework"] = f"{getattr(test_meta, 'framework', 'unknow')} {getattr(test_meta, 'framework_version', 'unknow')}"
        pd_data["Cluster Spec"] = getattr(test_meta, 'cluster', 'unknow')
        pd_data["Dataset"] = tags.dataset
        pd_data["Total Requests"] = tags.request_num
        pd_data["Request Rate"] = "Serial" if float(tags.request_rate) == -1 else str(tags.request_rate)
        pd_data["Request Success Rate(%)"] = round(metrics.completed / tags.request_num * 100., 2)
        pd_data["Audit Hits"] = metrics.audit_hit
        pd_data["Concurrency"] = tags.concurrency
        pd_data["Total Input Tokens"] = metrics.total_input
        pd_data["Total Output Tokens"] = metrics.total_output
        pd_data["Total Reasoning Tokens"] = metrics.total_resoning_tokens
        pd_data["Total Duration(s)"] = round(metrics.duration, 2)
        pd_data["Total Request Throughput(requests/s)"] = round(metrics.request_throughput, 2)
        pd_data["Total Input Throughput(tokens/s)"] = round(metrics.input_throughput, 2)
        pd_data["Total Output Throughput(tokens/s)"] = round(metrics.output_throughput, 2)
        
        pd_data["TTFT_avg(ms)"] = round(metrics.mean_ttft_ms, 2)
        pd_data["TPOT_avg(ms)"] = round(metrics.mean_tpot_ms, 2)
        pd_data["ITL_avg(ms)"] = round(metrics.mean_itl_ms, 2)
        pd_data["E2EL_avg(ms)"] = round(metrics.mean_e2el_ms, 2)

        pd_data["TTFT_p50(ms)"] = round(metrics.median_ttft_ms, 2)
        pd_data["TPOT_p50(ms)"] = round(metrics.median_tpot_ms, 2)
        pd_data["ITL_p50(ms)"] = round(metrics.median_itl_ms, 2)
        pd_data["E2EL_p50(ms)"] = round(metrics.median_e2el_ms, 2)

        pd_data["TTFT_p90(ms)"] = round(metrics.p90_ttft_ms, 2)
        pd_data["TPOT_p90(ms)"] = round(metrics.p90_tpot_ms, 2)
        pd_data["ITL_p90(ms)"] = round(metrics.p90_itl_ms, 2)
        pd_data["E2EL_p90(ms)"] = round(metrics.p90_e2el_ms, 2)

        pd_data["TTFT_p10(ms)"] = round(metrics.p10_ttft_ms, 2)
        pd_data["TPOT_p10(ms)"] = round(metrics.p10_tpot_ms, 2)
        pd_data["ITL_p10(ms)"] = round(metrics.p10_itl_ms, 2)
        pd_data["E2EL_p10(ms)"] = round(metrics.p10_e2el_ms, 2)

        pd_data["TTFT_p99(ms)"] = round(metrics.p99_ttft_ms, 2)
        pd_data["TPOT_p99(ms)"] = round(metrics.p99_tpot_ms, 2)
        pd_data["ITL_p99(ms)"] = round(metrics.p99_itl_ms, 2)
        pd_data["E2EL_p99(ms)"] = round(metrics.p99_e2el_ms, 2)

        pd_data["TTFT_p1(ms)"] = round(metrics.p1_ttft_ms, 2)
        pd_data["TPOT_p1(ms)"] = round(metrics.p1_tpot_ms, 2)
        pd_data["ITL_p1(ms)"] = round(metrics.p1_itl_ms, 2)
        pd_data["E2EL_p1(ms)"] = round(metrics.p1_e2el_ms, 2)
        
        result_pd.loc[len(result_pd)]= pd_data

    # save result
    base_model_id = benchmark.model.split("/")[-1]
    dataset_name = benchmark.dataset_name
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    if cfg.benchmark.get("result_data_path", ""):
        result_path = Path(cfg.benchmark.result_data_path)
    else:
        result_path = Path(PERF_DATA_FOLDER)
    result_path.mkdir(parents=True, exist_ok=True)
    file_name = f"result_data-{dataset_name}-{base_model_id}-{current_dt}.xlsx"
    file_name = file_name.replace('/', '_')
    result_pd.to_excel(result_path / file_name, index=False)

    # Report to wandb if writer exists
    if wandb_task and hasattr(wandb_task, 'get_wandb_writer'):
        wandb_writer = wandb_task.get_wandb_writer()
        if wandb_writer:
            # Create table with columns matching dynamic_cols
            table = wandb.Table(columns=dynamic_cols, log_mode="MUTABLE")
            
            # Add data rows from result_pd
            for _, row in result_pd.iterrows():
                table.add_data(*[str(row[col]) for col in dynamic_cols])
            
            wandb_writer.log({"benchmark_results": table})

    return os.path.join(result_path, file_name)


def custom_serializer(obj):
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def save_process_data_to_json(cfg, result_json: Dict) -> str:
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_model_id = cfg.benchmark.model.split("/")[-1]
    file_name = f"{cfg.benchmark.backend}-{base_model_id}-{cfg.benchmark.concurrency}concurrency-{cfg.benchmark.request_rate}qps-{current_dt}.json"
        
    base_path = (
        Path(cfg.benchmark.get("result_data_path"))
        if cfg.benchmark.get("result_data_path")
        else Path(PERF_DATA_FOLDER)
    )
    process_path = base_path / 'process'
    process_path.mkdir(parents=True, exist_ok=True)
    
    file_path = process_path / file_name
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, default=custom_serializer, ensure_ascii=False, indent=4)
    
    return str(file_path)