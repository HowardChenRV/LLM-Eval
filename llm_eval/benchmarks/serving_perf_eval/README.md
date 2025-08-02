# Serving Performance Evaluation Benchmark

## Overview

This benchmark tool is designed to evaluate the performance of Large Language Model (LLM) inference services. It supports multiple backends (OpenAI, VLLM, MAAS) and allows testing with various datasets. The tool provides comprehensive metrics for analyzing service performance under different load conditions.

## Features

- Support for multiple inference backends (OpenAI, VLLM, MAAS)
- Flexible dataset handling (random generation, local datasets, ShareGPT, etc.)
- Configurable concurrency and request rate settings
- Detailed performance metrics collection
- Results export to Excel and JSON formats
- Integration with Weights & Biases for experiment tracking

## Installation

Ensure you have Python 3.8+ installed. Install the package in development mode:

```bash
cd LLM-Eval
pip install -e .
```

## Usage

### Basic Command Line Interface

To run the benchmark, use the following command structure:

```bash
llm-eval serving_perf_eval [OPTIONS]
```

### Key Options

#### Inference Server Configuration
- `--backend`: Inference API protocol for different backends (choices: openai, vllm, maas) (default: openai)
- `--url`: URL for the inference server (default: "http://127.0.0.1:8000/v1/chat/completions")
- `--api-key`: API key for authentication
- `--model`: Model name to use
- `--tokenizer`: Tokenizer name or local path (required when using random dataset)
- `--extra-headers`: Comma-separated string or JSON formatted arguments for extra headers

#### Dataset Configuration
- `--dataset-name`: Name of the dataset (choices from registered datasets)
- `--dataset-path`: Path to the dataset file
- `--num-prompts`: Number of prompts to use (default: 10)
- `--prompt`: Prompt to use (mutually exclusive with dataset_path and dataset_name)

##### Random Dataset Options
- `--random-input-len`: Number of input tokens per request (default: 1024)
- `--random-output-len`: Number of output tokens per request (default: 128)
- `--random-range-ratio`: Range ratio for sampling input/output length (default: 0.0)
- `--random-prefix-len`: Number of fixed prefix tokens before the random context (default: 0)

#### Performance Test Configuration
- `--perf-test-type`: Type of performance test (choices: single, multiple) (default: single)
- `--concurrency`: Number of concurrent requests (default: 1)
- `--request-rate`: Request rate in requests per second (default: inf)
- `--request-rate-list`: List of request rates for multiple tests (e.g., -1,4,8,16,inf)
- `--concurrency-list`: List of concurrency levels for multiple tests (e.g., 1,2,3,4,5)
- `--limit-max-concurrency`: Limit maximum concurrency using asyncio.Semaphore

#### Data Saving Configuration
- `--save-data`: Save generated process data to JSON file
- `--result-data-path`: Path to save the result data file (default: ~/.cache/llm_eval/serving_perf_eval_result/)

### Examples

#### Single Test with Random Dataset
```bash
llm-eval serving_perf_eval \
  --backend openai \
  --url http://localhost:8000/v1/chat/completions \
  --model my-model \
  --tokenizer my-tokenizer \
  --dataset-name random \
  --num-prompts 100 \
  --concurrency 10 \
  --request-rate 1.0 \
  --save-data
```

#### Multiple Tests with Different Concurrency Levels
```bash
llm-eval serving_perf_eval \
  --backend vllm \
  --url http://localhost:8000/v1/chat/completions \
  --model my-model \
  --tokenizer my-tokenizer \
  --dataset-name sharegpt \
  --dataset-path /path/to/sharegpt.json \
  --num-prompts 100 \
  --perf-test-type multiple \
  --concurrency-list 1,2,4,8 \
  --request-rate 1.0 \
  --save-data
```

#### Multiple Tests with Different Request Rates
```bash
llm-eval serving_perf_eval \
  --backend maas \
  --url http://my-maas-service.com/api/v1/chat/completions \
  --model my-model \
  --api-key my-api-key \
  --dataset-name random \
  --tokenizer my-tokenizer \
  --num-prompts 100 \
  --perf-test-type multiple \
  --concurrency 4 \
  --request-rate-list -1,0.5,1,2,4,inf \
  --save-data
```

## Output Metrics

The benchmark collects and reports the following metrics:

### Summary Metrics
- Duration (seconds)
- Completed requests
- Audit hits
- Total input tokens
- Total output tokens
- Total reasoning tokens
- Request throughput (requests/second)
- Input throughput (tokens/second)
- Output throughput (tokens/second)

### Latency Metrics
- TTFT (Time To First Token): Average, Median, P99, P1, P90, P10
- TPOT (Time Per Output Token): Average, Median, P99, P1, P90, P10
- ITL (Inter-Token Latency): Average, Median, P99, P1, P90, P10
- E2EL (End-to-End Latency): Average, Median, P99, P1, P90, P10

### Data Export

Results are saved in two formats:
1. Excel file containing summary metrics for all test configurations
2. JSON files containing detailed process data for debugging (when --save-data is enabled)

The default save location is `~/.cache/llm_eval/serving_perf_eval_result/`, but this can be customized using the `--result-data-path` option.

## Architecture

The benchmark consists of three main components:

1. **CLI Interface** ([`llm_eval/cli/serving_perf_eval.py`](file:///Users/howardchen/Dev/LLM-Eval/llm_eval/cli/serving_perf_eval.py)): Handles command-line argument parsing and configuration management.
2. **Benchmark Core** ([`llm_eval/benchmarks/serving_perf_eval/benchmark_serving_perf.py`](file:///Users/howardchen/Dev/LLM-Eval/llm_eval/benchmarks/serving_perf_eval/benchmark_serving_perf.py)): Implements the core benchmark logic including request generation, HTTP client handling, and metrics collection.
3. **Data Saving** ([`llm_eval/benchmarks/serving_perf_eval/save_perf_data.py`](file:///Users/howardchen/Dev/LLM-Eval/llm_eval/benchmarks/serving_perf_eval/save_perf_data.py)): Manages exporting results to Excel and JSON formats.

The benchmark uses asynchronous programming (asyncio) to efficiently handle multiple concurrent requests and collect performance metrics.