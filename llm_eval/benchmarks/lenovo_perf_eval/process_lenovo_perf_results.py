import json
import os
import numpy as np
import pandas as pd
from typing import List
from datetime import datetime
from dataclasses import asdict

USER_HOME = os.path.expanduser("~")
PERF_DATA_FOLDER = os.path.join(USER_HOME, '.cache/llm_eval/lenovo_perf_eval_result/')

def group_by_input_length(results: List):
    """Group results by input length ranges"""
    bins = [(1, 300), (300, 700), (700, 1500), (1500, 3000), (3000, 4096)]
    groups = {f"{low}-{high}": [] for low, high in bins}
    
    for result in results:
        if not result.process_metrics:
            continue
            
        input_len = result.process_metrics.input_length
        for low, high in bins:
            if low < input_len <= high:
                groups[f"{low}-{high}"].append(result)
                break
                
    return groups

def calculate_group_stats(group_results: List):
    """Calculate statistics for a single group"""
    if not group_results:
        return None
        
    # Basic metrics
    prefill_tp = [r.prefill_throughput for r in group_results if r.prefill_throughput > 0]
    gen_tp = [r.generation_throughput for r in group_results if r.generation_throughput > 0]
    success_rate = sum(1 for r in group_results if r.success) / len(group_results)
    
    # Latency metrics
    ttfts = [r.process_metrics.ttft for r in group_results if r.process_metrics]
    tpots = [r.process_metrics.tpot for r in group_results if r.process_metrics]
    latencies = [r.process_metrics.latency for r in group_results if r.process_metrics]
    
    return {
        "count": len(group_results),
        "success_rate": success_rate,
        "prefill_throughput": {
            "avg": np.mean(prefill_tp) if prefill_tp else 0,
            "p1": np.percentile(prefill_tp, 1) if prefill_tp else 0,
            "p10": np.percentile(prefill_tp, 10) if prefill_tp else 0,
            "p50": np.percentile(prefill_tp, 50) if prefill_tp else 0,
            "p90": np.percentile(prefill_tp, 90) if prefill_tp else 0,
            "p99": np.percentile(prefill_tp, 99) if prefill_tp else 0
        },
        "generation_throughput": {
            "avg": np.mean(gen_tp) if gen_tp else 0,
            "p1": np.percentile(gen_tp, 1) if gen_tp else 0,
            "p10": np.percentile(gen_tp, 10) if gen_tp else 0,
            "p50": np.percentile(gen_tp, 50) if gen_tp else 0,
            "p90": np.percentile(gen_tp, 90) if gen_tp else 0,
            "p99": np.percentile(gen_tp, 99) if gen_tp else 0
        },
        "ttft": {
            "avg": np.mean(ttfts) if ttfts else 0,
            "p1": np.percentile(ttfts, 1) if ttfts else 0,
            "p10": np.percentile(ttfts, 10) if ttfts else 0,
            "p50": np.percentile(ttfts, 50) if ttfts else 0,
            "p90": np.percentile(ttfts, 90) if ttfts else 0,
            "p99": np.percentile(ttfts, 99) if ttfts else 0
        },
        "tpot": {
            "avg": np.mean(tpots) if tpots else 0,
            "p1": np.percentile(tpots, 1) if tpots else 0,
            "p10": np.percentile(tpots, 10) if tpots else 0,
            "p50": np.percentile(tpots, 50) if tpots else 0,
            "p90": np.percentile(tpots, 90) if tpots else 0,
            "p99": np.percentile(tpots, 99) if tpots else 0
        },
        "latency": {
            "avg": np.mean(latencies) if latencies else 0,
            "p1": np.percentile(latencies, 1) if latencies else 0,
            "p10": np.percentile(latencies, 10) if latencies else 0,
            "p50": np.percentile(latencies, 50) if latencies else 0,
            "p90": np.percentile(latencies, 90) if latencies else 0,
            "p99": np.percentile(latencies, 99) if latencies else 0
        }
    }
    
def flatten_stats(stats_dict):
    """Flatten nested stats dictionary into flat structure"""
    flat_stats = {}
    for metric, values in stats_dict.items():
        if isinstance(values, dict):
            for sub_metric, value in values.items():
                flat_stats[f"{metric}_{sub_metric}"] = value
        else:
            flat_stats[metric] = values
    return flat_stats

def process_results(
    results: List,
    output_prefix: str
) -> dict:
    """Main processing function"""
    # Ensure directories exist
    os.makedirs(PERF_DATA_FOLDER, exist_ok=True)
    process_dir = os.path.join(PERF_DATA_FOLDER, "process")
    os.makedirs(process_dir, exist_ok=True)
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save raw results to JSON in process subdirectory with timestamp
    json_path = os.path.join(process_dir, f"{output_prefix}_{timestamp}_raw.json")
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    # Group by input length
    groups = group_by_input_length(results)
    
    # Calculate stats for each group (skip empty groups)
    stats = {}
    empty_groups = []
    for range_name, group_results in groups.items():
        if not group_results:
            empty_groups.append(range_name)
            continue
        stats[range_name] = flatten_stats(calculate_group_stats(group_results))
    
    if empty_groups:
        print(f"Warning: Empty groups found for ranges: {', '.join(empty_groups)}")
    
    # Convert to DataFrame and save to Excel with timestamp
    df = pd.DataFrame.from_dict(stats, orient="index")
    excel_path = os.path.join(PERF_DATA_FOLDER, f"{output_prefix}_{timestamp}_stats.xlsx")
    df.to_excel(excel_path)

    # Prepare WandB summary data (only throughput metrics)
    benchmark_summary = {}
    for range_name, stat in stats.items():
        for metric, value in stat.items():
            if metric.startswith(("prefill_throughput_", "generation_throughput_")):
                benchmark_summary[f"{range_name}/{metric}"] = value
    benchmark_summary["timestamp"] = timestamp
    benchmark_summary["output_prefix"] = output_prefix

    # Print throughput summary with better formatting (avg/p50/p99 only)
    print("Throughput Statistics by Input Length Range (avg/p50/p99):")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    throughput_df = df.filter(regex='throughput.*(avg|p50|p99)')
    print(throughput_df.to_string())
    print(f"Raw results saved to: {json_path}")
    print(f"Statistics saved to: {excel_path}")
    
    return benchmark_summary
    
    
if __name__ == "__main__":
    # Example usage
    # # 构造测试数据
    # test_results = [
    #     # 成功案例 - 短输入 (1-300)
    #     LenovoPerfEvalResult(
    #         success=True,
    #         process_metrics=InferencePerformanceProcess(
    #             input_length=100,
    #             output_length=50,
    #             reasoning_length=0,
    #             latency=1.5,
    #             ttft=0.2,
    #             tpot=0.05,
    #             start_time=0,
    #             end_time=1.5
    #         ),
    #         prefill_throughput=500.0,
    #         generation_throughput=1000.0
    #     ),
    #     # 成功案例 - 中等输入 (300-700)
    #     LenovoPerfEvalResult(
    #         success=True,
    #         process_metrics=InferencePerformanceProcess(
    #             input_length=500,
    #             output_length=200,
    #             reasoning_length=0,
    #             latency=3.0,
    #             ttft=0.5,
    #             tpot=0.1,
    #             start_time=0,
    #             end_time=3.0
    #         ),
    #         prefill_throughput=300.0,
    #         generation_throughput=800.0
    #     ),
    #     # 失败案例 - 长输入 (700-1500)
    #     LenovoPerfEvalResult(
    #         success=False,
    #         error_msg="Timeout error",
    #         process_metrics=InferencePerformanceProcess(
    #             input_length=1000,
    #             output_length=0,
    #             reasoning_length=0,
    #             latency=10.0,
    #             ttft=2.0,
    #             tpot=0,
    #             start_time=0,
    #             end_time=10.0
    #         ),
    #         prefill_throughput=100.0,
    #         generation_throughput=0.0
    #     ),
    #     # 成功案例 - 超长输入 (1500-3072)
    #     LenovoPerfEvalResult(
    #         success=True,
    #         process_metrics=InferencePerformanceProcess(
    #             input_length=2000,
    #             output_length=500,
    #             reasoning_length=0,
    #             latency=8.0,
    #             ttft=1.5,
    #             tpot=0.2,
    #             start_time=0,
    #             end_time=8.0
    #         ),
    #         prefill_throughput=200.0,
    #         generation_throughput=600.0
    #     ),
    #     # 成功案例 - 最大长度输入 (3072-4096)
    #     LenovoPerfEvalResult(
    #         success=True,
    #         process_metrics=InferencePerformanceProcess(
    #             input_length=3500,
    #             output_length=800,
    #             reasoning_length=0,
    #             latency=15.0,
    #             ttft=3.0,
    #             tpot=0.3,
    #             start_time=0,
    #             end_time=15.0
    #         ),
    #         prefill_throughput=150.0,
    #         generation_throughput=500.0
    #     )
    # ]

    # # 运行测试
    # process_results(test_results, "test_output")

    # print("测试数据已生成并处理完成，结果保存在 test_output_raw.json 和 test_output_stats.xlsx")
    pass
