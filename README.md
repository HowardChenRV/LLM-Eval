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