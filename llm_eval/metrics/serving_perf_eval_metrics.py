from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class InferenceServingPerformanceTags:
    request_num: int         # total request number
    request_rate: str        # request rate (like 0.3, 1, 2, 4, inf)
    dataset: str             # dataset name
    concurrency: int         # concurrency


@dataclass
class InferencePerformanceProcess:
    input_length: int                # tokens
    output_length: int               # tokens
    reasoning_length: int            # tokens
    latency: float                   # ms
    ttft: float                      # ms
    tpot: float                      # ms
    # start_time: float                # s timestamp
    # end_time: float                  # s timestamp
    # extra_body: dict = None          # additional request body


@dataclass
class InferenceServingPerformanceSummary:
    duration: float                 # test duration
    completed: int                  # success request
    audit_hit: int                  # audit hit
    total_input: int                # total input tokens
    total_output: int               # total output tokens
    total_resoning_tokens: int      # total resoning tokens
    request_throughput: float       # request/s
    input_throughput: float         # input token/s
    output_throughput: float        # output token/s
    mean_ttft_ms: float             # ttft (ms) 
    median_ttft_ms: float       
    p99_ttft_ms: float
    p1_ttft_ms: float
    p90_ttft_ms: float
    p10_ttft_ms: float
    mean_tpot_ms: float             # tpot (ms)
    median_tpot_ms: float
    p99_tpot_ms: float
    p1_tpot_ms: float
    p90_tpot_ms: float
    p10_tpot_ms: float
    mean_itl_ms: float              # itl (ms)
    median_itl_ms: float
    p99_itl_ms: float
    p1_itl_ms: float
    p90_itl_ms: float
    p10_itl_ms: float
    mean_e2el_ms: float             # e2el (ms)
    median_e2el_ms: float
    p99_e2el_ms: float
    p1_e2el_ms: float
    p90_e2el_ms: float
    p10_e2el_ms: float
    prompt_input_lens: list         # prompt input length
    actual_output_lens: list        # actual output length
    actual_reasoning_lens: list     # actual reasoning length

    def pretty_print(self):
        percentiles = [0, 25, 50, 75, 100]
        prompt_input_percentiles = np.percentile(self.prompt_input_lens or 0, percentiles)
        actual_output_percentiles = np.percentile(self.actual_output_lens or 0, percentiles)
        actual_reasoning_percentiles = np.percentile(self.actual_reasoning_lens or 0, percentiles)
        prompt_input_avg = np.mean(self.prompt_input_lens or 0)
        actual_output_avg = np.mean(self.actual_output_lens or 0)
        actual_reasoning_avg = np.mean(self.actual_reasoning_lens or 0)
        print("-" * 100)
        print("Datasets Summary: ")
        df = pd.DataFrame({
            'Percentile': [f'{p}%' for p in percentiles] + ['Average'],
            'Prompt Input Lengths': np.append(prompt_input_percentiles, prompt_input_avg),
            'Actual Output Lengths': np.append(actual_output_percentiles, actual_output_avg),
            'Actual Reasoning Lengths': np.append(actual_reasoning_percentiles, actual_reasoning_avg)
        })
        print(df.to_string(index=False, float_format="%.0f"))
           
        print("-" * 100)
        print("Inference Serving Performance Summary: ")
        print(f"Test Duration: {self.duration:.2f} seconds")
        print(f"Completed Requests: {self.completed}")
        print(f"Audit hit Requests: {self.audit_hit}")
        print(f"Total Input Tokens: {self.total_input}")
        print(f"Total Output Tokens: {self.total_output}")
        print(f"Total Reasoning Tokens: {self.total_resoning_tokens}")
        print(f"Request Throughput: {self.request_throughput:.2f} requests/s (RPM:{self.request_throughput*60:.2f})")
        print(f"Input Throughput: {self.input_throughput:.2f} tokens/s")
        print(f"Output Throughput: {self.output_throughput:.2f} tokens/s (TPM:{self.output_throughput*60:.2f})")
        
        print("Latency Statistics: (ms)")
        stats = [
            ('TTFT', self.mean_ttft_ms, self.p1_ttft_ms, self.p10_ttft_ms, self.median_ttft_ms, self.p90_ttft_ms, self.p99_ttft_ms),
            ('TPOT', self.mean_tpot_ms, self.p1_tpot_ms, self.p10_tpot_ms, self.median_tpot_ms, self.p90_tpot_ms, self.p99_tpot_ms),
            ('ITL', self.mean_itl_ms, self.p1_itl_ms, self.p10_itl_ms, self.median_itl_ms, self.p90_itl_ms, self.p99_itl_ms),
            ('E2EL', self.mean_e2el_ms, self.p1_e2el_ms, self.p10_e2el_ms, self.median_e2el_ms, self.p90_e2el_ms, self.p99_e2el_ms)
        ]
        df = pd.DataFrame(stats, columns=['Statistic', 'Avg', 'P1', 'P10', 'P50', 'P90', 'P99'])
        print(df.to_string(index=False, float_format="%.2f"))
        print("-" * 100)
        print("(Please note that if the API response contains ‘reasoning_content’, and the length of ‘reasoning_content’ is not included in the returned output token length, the data statistics may be inaccurate.)")