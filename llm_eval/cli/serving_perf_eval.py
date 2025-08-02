import argparse
from dataclasses import dataclass
from omegaconf import OmegaConf
from typing import List, Optional
from llm_eval.benchmarks.serving_perf_eval.benchmark_serving_perf import run_benchmark, Backend, PerfTestType
from llm_eval.cli.base import CLICommand, add_global_arguments, try_parse_json
from llm_eval.utils.configs import ConfigManager, simple_parse_args_string
from llm_eval.utils.registry import dataset_registry, api_registry

# for registry
import importlib
PLUGIN_REGISTRY = [
    # dataset
    "llm_eval.datasets.random",
    "llm_eval.datasets.local",
    "llm_eval.datasets.sharegpt",
    "llm_eval.datasets.math_500",
    "llm_eval.datasets.aime_2024",
    "llm_eval.datasets.tencent_yuanbao.tencent_yuanbao",
    "llm_eval.datasets.lp_resume.resume_screening",
    "llm_eval.datasets.lenovo.lenovo_mmlu",
    "llm_eval.datasets.multi_modal.mmbench_cn",
    # api
    "llm_eval.apis.openai_api",
    "llm_eval.apis.vllm_api",
    "llm_eval.apis.maas_api",
]
for module in PLUGIN_REGISTRY:
    importlib.import_module(module)

BENCHMARK_NAME = "serving_perf_eval"

@dataclass
class ServingPerfEvalArgument:
    benchmark_name: str = BENCHMARK_NAME
    # Inference server config 
    backend: str = Backend.OPENAI.value
    url: str = "http://127.0.0.1:8000/v1/chat/completions"
    api_key: str = ""
    model: str = ""
    tokenizer: str = ""
    extra_headers: Optional[dict] = None  # Extra headers for the request, if needed
    # Dataset config
    dataset_name: str = ""
    dataset_path: str = ""
    num_prompts: int = 1
    prompt: str = ""   # Mutually exclusive with dataset_path and dataset_name
    # random dataset options
    random_input_len: int = 1024
    random_output_len: int = 128
    random_range_ratio: float = 0.0
    random_prefix_len: int = 0
    # Perf test config
    perf_test_type: str = PerfTestType.SINGLE.value
    concurrency: int = 1
    request_rate: float = float("inf")  # Value of -1 indicates sending requests sequentially.
    request_rate_list: Optional[List[float]] = None  # Only for PerfTestType.multiple
    concurrency_list: Optional[List[int]] = None  # Only for PerfTestType.multiple
    limit_max_oncurrency: bool = False  # If True, use asyncio.Semaphore to limit the max concurrency.
    # Save data config
    save_data: bool = False
    result_data_path: str = ""  # Path to save the result data file. If not specified, the data will be saved to cache dir.

def parse_request_rates(rate_string):
    rates = rate_string.split(',')
    float_rates = []

    for rate in rates:
        if rate.lower() == 'inf':
            float_rates.append(float('inf'))
        else:
            try:
                float_rates.append(float(rate))
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid request rate: {rate}")

    return float_rates


def parse_concurrency_list(rate_string):
    rates = rate_string.split(',')
    float_rates = []

    for rate in rates:
        try:
            float_rates.append(int(rate))
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid concurrency: {rate}")

    return float_rates


def add_argument(parser: argparse.ArgumentParser):
    # Inference server config
    parser.add_argument('--backend', type=str, choices=[api for api in api_registry.all_classes()], 
                        default=Backend.OPENAI.value, help='Inference api protocol for different backends')
    parser.add_argument('--url', type=str, default="http://127.0.0.1:8000/v1/chat/completions", 
                        help='URL for the inference server')
    parser.add_argument('--api-key', type=str, default="", help='API key for authentication')
    parser.add_argument('--model', type=str, default="", help='Model name to use')
    parser.add_argument('--tokenizer', type=str, default="", help='Tokenizer name or local path; only required (and used) when --dataset-name random.')
    parser.add_argument(
        "--extra-headers",
        default="",
        type=try_parse_json,
        help="""Comma separated string or JSON formatted arguments for extra headers, e.g. `header1=value1,header2=value2,` or '{"header1": "value1", "header2": "value2"}'.""",
    )

    # Dataset config
    parser.add_argument('--dataset-name', type=str, default="", choices=[dataset for dataset in dataset_registry.all_classes()], help='Name of the dataset')
    parser.add_argument('--dataset-path', type=str, default="", help='Path to the dataset file')
    parser.add_argument('--num-prompts', type=int, default=10, help='Number of prompts to use')
    parser.add_argument('--prompt', type=str, default="Hello!", help='Prompt to use (mutually exclusive with dataset_path and dataset_name)')

    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help="Number of output tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range ratio for sampling input/output length, "
        "used only for random sampling. Must be in the range [0, 1) to define "
        "a symmetric sampling range"
        "[length * (1 - range_ratio), length * (1 + range_ratio)].",
    )
    random_group.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help=(
            "Number of fixed prefix tokens before the random context "
            "in a request. "
            "The total input length is the sum of `random-prefix-len` and "
            "a random "
            "context length sampled from [input_len * (1 - range_ratio), "
            "input_len * (1 + range_ratio)]."
        ),
    )

    # Perf test config
    parser.add_argument('--perf-test-type', type=str, choices=[perf_test.value for perf_test in PerfTestType], 
                        default=PerfTestType.SINGLE.value, help='Type of performance test')
    parser.add_argument('--concurrency', type=int, default=1, help='Number of concurrent requests')
    parser.add_argument('--request-rate', type=float, default=float("inf"), help='Request rate (requests per second)')
    parser.add_argument('--request-rate-list', type=parse_request_rates, default=None,
                        help='List of request rate for multiple tests (e.g. -1,4,8,16,inf)')
    parser.add_argument('--concurrency-list', type=parse_concurrency_list, default=None,
                        help='List of concurrency for multiple tests (e.g. 1,2,3,4,5)')
    parser.add_argument('--limit-max-oncurrency', action='store_true', help='If set, use asyncio.Semaphore to limit the max concurrency. ')
    # Save data config
    parser.add_argument('--save-data', action='store_true', help='Save generate process data to json file.')
    parser.add_argument('--result-data-path', type=str, default='', help='Path to save the result data file. If not specified, the data will be saved to cache dir.')


class ServingPerfEvalCMD(CLICommand):
    name = BENCHMARK_NAME

    def __init__(self, args):
        self.args = args

    @staticmethod
    def define_args(parsers: argparse.ArgumentParser):
        parser = parsers.add_parser(ServingPerfEvalCMD.name)
        add_global_arguments(parser)
        add_argument(parser)
        parser.set_defaults(func=lambda args: ServingPerfEvalCMD(args))

    def execute(self):
        config_manager = ConfigManager(env=self.args.env, extra_args=self.args.extra_args, meta_data=self.args.meta_data)
        
        # Test logging
        import logging
        logging.debug("DEBUG test message - should appear if level=DEBUG")
        logging.info("INFO test message - should always appear")
        
        if self.args.use_yaml:
            config_manager.add_benchmark_by_yaml(case_path=self.args.yaml_path)
        else:
            eval_args = ServingPerfEvalArgument(
                backend=self.args.backend,
                url=self.args.url,
                api_key=self.args.api_key,
                model=self.args.model,
                tokenizer=self.args.tokenizer,
                extra_headers=simple_parse_args_string(self.args.extra_headers) if isinstance(self.args.extra_headers, str) else self.args.extra_headers if isinstance(self.args.extra_headers, dict) else {},
                dataset_name=self.args.dataset_name,
                dataset_path=self.args.dataset_path,
                num_prompts=self.args.num_prompts,
                prompt=self.args.prompt,
                random_input_len=self.args.random_input_len,
                random_output_len=self.args.random_output_len,
                random_range_ratio=self.args.random_range_ratio,
                random_prefix_len=self.args.random_prefix_len,
                perf_test_type=self.args.perf_test_type,
                concurrency=self.args.concurrency,
                request_rate=self.args.request_rate,
                request_rate_list=self.args.request_rate_list,
                concurrency_list=self.args.concurrency_list,
                limit_max_oncurrency=self.args.limit_max_oncurrency,
                save_data=self.args.save_data,
                result_data_path=self.args.result_data_path,
            )
            config_manager.config.benchmark = OmegaConf.structured(eval_args)
        
        config_manager.print_config()
        run_benchmark(config_manager.config)
    
    