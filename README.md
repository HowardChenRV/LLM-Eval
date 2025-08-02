## Installation
### Install from Source
1. Download the source code
   ```shell
   git clone https://github.com/HowardChenRV/LLM-Eval.git
   ```

2. Create a conda environment (optional)
   We recommend using conda to manage your environment:
   ```shell
   # It is recommended to use Python 3.10 ~ 3.11
   conda create -n llm-eval python=3.10
   # Activate the conda environment
   conda activate llm-eval
   ```

3. Install dependencies
   ```shell
   cd LLM-Eval/
   pip install -e .                  # Install Default Dependencies
   # Additional options
   pip install -e '.[test]'
   pip install -e '.[all]'           
   ```


## Quick Start
   ```shell
   llm-eval -h
   # LLM serving performance evaluation
   llm-eval serving_perf_eval -h
   # Edge(AIPC) LLM performance evaluation for lenovo-demo server
   llm-eval lenovo_perf_eval -h
   ```

## Project Modules

This project contains several modules for evaluating different aspects of LLM performance:

### Serving Performance Evaluation (serving_perf_eval)

Benchmark tool designed to evaluate the performance of Large Language Model (LLM) inference services. It supports multiple backends (OpenAI, VLLM, MAAS) and allows testing with various datasets.

For detailed information, please refer to [llm_eval/benchmarks/serving_perf_eval/README.md](llm_eval/benchmarks/serving_perf_eval/README.md).

### Lenovo Performance Evaluation (lenovo_perf_eval)

Edge (AIPC) LLM performance evaluation specifically for Lenovo demo servers.

For detailed information, please refer to [llm_eval/benchmarks/lenovo_perf_eval/README.md](llm_eval/benchmarks/lenovo_perf_eval/README.md).

### llama-bench Performance Testing (llama_bench_performance)

Automated testing tool for running llama-bench performance tests and collecting results. It supports multi-model test configurations and sends test results to Kafka for centralized statistics and analysis.

For detailed information, please refer to [examples/llama_bench_performance/README.md](examples/llama_bench_performance/README.md).

### Base Evaluation (base_eval)

Cluster basic performance evaluation including GEMM, MEMCPY, P2P, NCCL, GPFS, and other tests.

For detailed information, please refer to [examples/base_eval/README.md](examples/base_eval/README.md).

### Data Platform (data_platform)

For storing and processing various test data, displaying test processes and results.

Environment startup:
```bash
cd docker
docker-compose up -d
```

Middleware addresses:

| **Middleware**    | **Access Address**           | **Username/Password**  | **Remarks**              |
|-------------------|------------------------------|------------------------|--------------------------|
| Kafka             | localhost:9094               |                        |                          |
| Kafka-UI          | http://localhost:8080/       |                        | Kafka Web Admin Console  |
| MongoDB           | localhost:27017              | admin/admin            |                          |
| Mongo Express     | http://localhost:8081/       |                        | Mongo Web Admin Console  |
| InfluxDB          | http://localhost:8086/       |                        |                          |
| Chronograf        | http://localhost:8888/       |                        | InfluxDB Visualization Plugin |
| Grafana           | http://localhost:3000/       | admin/admin            |                          |
| Apache NiFi       | https://localhost:8443/nifi/ | admin/a12345678910     | Data ETL Tool            |

## Windows EXE Packaging

To package the application as a Windows executable:

1. Install PyInstaller:
```shell
pip install pyinstaller
```

2. Run the build script:
```shell
python build_windows_exe.py
```

This will generate a single executable file in the `dist` folder.

Note:
- The build script requires Python 3.10+
- Make sure all dependencies are installed before building
- The executable is built for Windows but can be created on macOS/Linux using Wine

## Development Guide

### How to Add a New Benchmark

To add a new benchmark, you need to complete the following steps:

1. Create a new benchmark directory under `llm_eval/benchmarks/`, for example `my_new_benchmark/`
2. Implement your benchmark logic in that directory, referencing existing implementations like `llm_eval/benchmarks/serving_perf_eval/`
3. Create a corresponding CLI command file under `llm_eval/cli/`, for example `my_new_benchmark.py`
4. In the CLI command file:
   - Define argument parsing function `add_argument(parser)`
   - Create a command class inheriting from `CLICommand` and implement `define_args` and `execute` methods
   - Use `@dataclass` to define benchmark configuration parameters
5. Register your new command in `llm_eval/cli/main.py`:
   - Import your command class: `from llm_eval.cli.my_new_benchmark import MyNewBenchmarkCMD`
   - Add in the `run_command` function: `MyNewBenchmarkCMD.define_args(subparsers)`
6. Ensure your benchmark implementation correctly uses the `ConfigManager` for configuration handling

### How to Add a New API

To add a new API, you need to complete the following steps:

1. Create a new API implementation file under `llm_eval/apis/`, for example `my_new_api.py`
2. Inherit from the `ApiBase` base class and implement the required methods:
   - `create_request_input`: Create request input object
   - `build_request`: Build HTTP request
   - `parse_responses`: Parse server responses
3. Define a data class inheriting from `RequestFuncInput` containing parameters specific to your API
4. Register your API using the `@register_api("api_name")` decorator
5. Import your API module in the CLI command file that uses it to ensure proper registration
6. Refer to existing implementations like `llm_eval/apis/openai_api.py` or `llm_eval/apis/lenovo_api.py`

### How to Add a New Dataset

To add a new dataset, you need to complete the following steps:

1. Create a new dataset implementation file under `llm_eval/datasets/`, for example `my_new_dataset.py`
2. Inherit from the `DatasetBase` base class and implement the required methods:
   - `load`: Static method for loading the dataset
   - `perf_req_generator`: Method for generating performance test requests (optional, if customization is needed)
3. Register your dataset using the `@register_dataset("dataset_name")` decorator
4. In your dataset implementation, ensure proper data format handling and return a list of `PerfReq` objects
5. Import your dataset module in the CLI command file that uses it to ensure proper registration
6. Refer to existing implementations like `llm_eval/datasets/random.py` or `llm_eval/datasets/sharegpt.py`