# llama-bench-test Tool

## Overview

llama-bench-test is an automated testing tool for running llama-bench performance tests and collecting results. It supports multi-model test configurations and sends test results to Kafka for centralized statistics and analysis.

## Features

- Support loading test cases from YAML configuration files
- Can test multiple different models simultaneously
- Real-time output of test logs and results
- Test results automatically reported to Kafka
- Detailed error handling and logging
- Support for recording Git branch and commit information

## Usage Instructions

### Runtime Environment
Windows AIPC, refer to document xxx

### Dependency Installation

```bash
# Requires Python 3.10~3.12
pip install -r requirement.txt
```

### Configuration File

Configuration file `llama_bench_case.yaml` example:

```yaml
test_case:
  model_list:
    - model_name: Qwen2.5-1.5B-Instruct
      quantization_method: Q4_0
      model_path: \\path\to\model.gguf
    - model_name: Qwen2.5-3B-Instruct
      quantization_method: Q4_0
      model_path: \\path\to\model.gguf
  prompt_tokens: "64,128"  # Number of input tokens
  gen_tokens: "32,64"      # Number of generated tokens

metrics:
  kafka:
    host: 8.140.201.231    # Kafka server address
    port: 9094             # Kafka port

logging:
  level: INFO              # Log level
  file_path: llama_bench_test_log  # Log directory
  format: "%(asctime)s [%(levelname)s] - %(message)s - %(filename)s:%(lineno)d"

meta_info:
  tester: CI/CD            # Tester information
```

### Running Tests

Basic usage:
```bash
python auto_run_llama_bench.py --llama-bench-path path/to/llama-bench.exe
```

Running with Git information:
```bash
python auto_run_llama_bench.py \
  --llama-bench-path path/to/llama-bench.exe \
  --branch main \
  --commit abc123
```

### Parameter Description

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| --llama-bench-path | Path to llama-bench executable | llama-bench.exe |
| --branch | Git branch name | None |
| --commit | Git commit hash | None |

## Log Description

Test logs will be saved in the `llama_bench_test_log` directory, with filenames in the format `runtime_YYYYMMDD_HHMMSS.log`.

Logs contain the following information:
- Test start/end time
- Test results for each model
- Error information (if any)
- Performance data

## Notes

1. Ensure model file paths are accessible
2. Kafka server needs to be pre-configured
3. Test duration may be long, it is recommended to run in a performance-stable environment
4. Log directory will be automatically created, no manual creation required